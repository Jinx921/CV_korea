"""Runtime patches embedded in the unified-RAG Kanana champion Colab notebook.

This module is not imported by the local project.  The notebook builder embeds its
source after cloning the pinned AIVQA repository in Colab.
"""

from __future__ import annotations

import copy
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import LogitsProcessor, LogitsProcessorList

import aivqa.data as aivqa_data
import aivqa.metrics as aivqa_metrics
import rag_db.augmentation as rag_augmentation
import rag_db.infer_with_rag as rag_infer
import rag_db.prompts as rag_prompts
import run_rag_pipeline as rag_pipeline
import train_lora
import type_adapters.modeling as type_modeling
import type_adapters.train as type_train


# ---------------------------------------------------------------------------
# 1. Colab에서 검증된 Kanana revision + 내부 vision SDPA 강제 로더
# ---------------------------------------------------------------------------


def _kanana_model_class():
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    model_class = get_class_from_dynamic_module(
        "modeling.KananaVForConditionalGeneration",
        MODEL_ID,
        revision=KANANA_REVISION,
    )
    vision_class = model_class.__init__.__globals__["CustomQwen2VLVE"]
    if not getattr(vision_class, "_champion_sdpa_patched", False):
        original_from_config = vision_class._from_config.__func__

        @classmethod
        def from_config_with_sdpa(cls, config, **kwargs):
            kwargs["attn_implementation"] = "sdpa"
            return original_from_config(cls, config, **kwargs)

        vision_class._from_config = from_config_with_sdpa
        vision_class._champion_sdpa_patched = True
    return model_class


def _processor_for(args):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        revision=KANANA_REVISION,
        trust_remote_code=True,
        cache_dir=str(MODEL_CACHE_DIR),
    )
    train_lora.configure_image_pixel_limits(
        processor, args.min_pixels, args.max_pixels
    )
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return processor


def _base_model_for(args, dtype):
    from transformers import BitsAndBytesConfig

    model_kwargs = {
        "revision": KANANA_REVISION,
        "dtype": dtype,
        "device_map": {"": torch.cuda.current_device()},
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
        "cache_dir": str(MODEL_CACHE_DIR),
    }
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    return _kanana_model_class().from_pretrained(args.model_id, **model_kwargs)


def safe_build_model_and_processor(args):
    from peft import LoraConfig, TaskType, get_peft_model

    if not torch.cuda.is_available():
        raise RuntimeError("Kanana 학습에는 CUDA GPU가 필요합니다.")
    dtype = getattr(torch, args.dtype)
    processor = _processor_for(args)
    model = _base_model_for(args, dtype)
    model.requires_grad_(False)
    model.config.use_cache = False
    model.language_model.config.use_cache = False
    llm = train_lora._prepare_llm_for_training(
        model,
        load_in_4bit=args.load_in_4bit,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    target_modules = train_lora.find_adapter_target_modules(
        name for name, _ in llm.named_modules()
    )
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    peft_llm = get_peft_model(llm, config)
    if hasattr(llm, "_require_grads_hook"):
        llm.disable_input_require_grads()
    model.language_model = peft_llm
    train_lora._verify_only_llm_adapters_are_trainable(model)
    model.language_model.print_trainable_parameters()
    return model, processor, dtype


def safe_load_base_model_and_processor(args, *, for_training):
    if not torch.cuda.is_available():
        raise RuntimeError("Kanana 실행에는 CUDA GPU가 필요합니다.")
    dtype = getattr(torch, args.dtype)
    processor = _processor_for(args)
    model = _base_model_for(args, dtype)
    model.requires_grad_(False)
    if for_training:
        model.config.use_cache = False
        model.language_model.config.use_cache = False
        model.language_model = train_lora._prepare_llm_for_training(
            model,
            load_in_4bit=args.load_in_4bit,
            gradient_checkpointing=args.gradient_checkpointing,
        )
    else:
        model.eval()
    return model, processor, dtype


rag_pipeline.build_model_and_processor = safe_build_model_and_processor
rag_pipeline.load_base_model_and_processor = safe_load_base_model_and_processor
rag_infer.load_base_model_and_processor = safe_load_base_model_and_processor
type_modeling.load_base_model_and_processor = safe_load_base_model_and_processor
type_train.load_base_model_and_processor = safe_load_base_model_and_processor


# ---------------------------------------------------------------------------
# 2. KURE/CLIP score를 그대로 더하지 않는 고신뢰 selective retrieval
#    JSONL에 실제 이미지 파일이 없으면 README 설계대로 text-only로 동작합니다.
# ---------------------------------------------------------------------------


def safe_retriever_init(
    self,
    client,
    collection_name,
    encoders,
    threshold,
    retrieval_page_size,
    local_image_search=False,
):
    self.client = client
    self.collection_name = collection_name
    self.encoders = encoders
    self.threshold = threshold
    self.retrieval_page_size = retrieval_page_size
    self.title_index = rag_infer.load_title_index(client, collection_name)
    self.local_image_index = None
    self.image_search_enabled = False
    if local_image_search:
        try:
            self.local_image_index = rag_infer.LocalImageIndex.load(
                client, collection_name
            )
            self.image_search_enabled = True
        except ValueError as error:
            if "no image vectors" not in str(error).lower():
                raise
            print(
                "Qdrant에 image vector가 없어 KURE text-only RAG로 전환합니다."
            )
    else:
        # Server mode에서는 native image query가 가능하다고 가정합니다.
        self.image_search_enabled = True


rag_infer.QdrantRetriever.__init__ = safe_retriever_init


def _candidate_gate(candidate):
    text_score = float(candidate.text_score)
    image_score = float(candidate.image_score)
    exact_title = text_score >= 1.99
    dual = text_score >= DUAL_TEXT_THRESHOLD and image_score >= DUAL_IMAGE_THRESHOLD
    strong_text = text_score >= STRONG_TEXT_THRESHOLD
    near_duplicate_image = image_score >= STRONG_IMAGE_THRESHOLD

    if exact_title:
        return "exact_title", 1.20 + min(image_score, 1.0) * 0.20
    if near_duplicate_image:
        return "near_duplicate_image", 1.10 + image_score * 0.10
    if dual:
        return "dual_channel", 0.55 * text_score + 0.45 * image_score + 0.10
    if strong_text:
        return "strong_text", text_score
    return None


def selective_retrieve(self, search_terms: Sequence[str], image):
    candidates = {}
    for term in search_terms:
        exact_doc_ids = self.title_index.get(rag_infer.normalize_exact_text(term), [])
        if exact_doc_ids:
            for payload in self._retrieve_payloads(exact_doc_ids):
                self._merge(candidates, payload, text_score=2.0)
            continue
        for point in self._query_all(
            self.encoders.embed_text(term), rag_infer.TEXT_VECTOR_NAME
        ):
            self._merge(
                candidates,
                rag_infer.validate_payload(point.payload),
                text_score=float(point.score),
            )

    if self.image_search_enabled:
        image_vector = self.encoders.embed_image(image)
        if self.local_image_index is not None:
            image_hits = self.local_image_index.search(image_vector, self.threshold)
            for payload, score in image_hits:
                self._merge(candidates, payload, image_score=score)
        else:
            for point in self._query_all(
                [image_vector], rag_infer.IMAGE_VECTOR_NAME, require_vector=True
            ):
                self._merge(
                    candidates,
                    rag_infer.validate_payload(point.payload),
                    image_score=float(point.score),
                )

    accepted = []
    for candidate in candidates.values():
        gate = _candidate_gate(candidate)
        if gate is None:
            continue
        reason, confidence = gate
        payload = dict(candidate.payload)
        payload["_retrieval"] = {
            "reason": reason,
            "confidence": float(confidence),
            "text_score": float(candidate.text_score),
            "image_score": float(candidate.image_score),
        }
        accepted.append(
            (
                confidence,
                rag_prompts.Candidate(
                    doc_id=candidate.doc_id,
                    payload=payload,
                    text_score=float(candidate.text_score),
                    image_score=float(candidate.image_score),
                ),
            )
        )
    accepted.sort(key=lambda item: (-item[0], item[1].doc_id))
    return [candidate for _, candidate in accepted[:RAG_TOP_K]]


rag_infer.QdrantRetriever.retrieve = selective_retrieve


# ---------------------------------------------------------------------------
# 3. 질문 관련 문장만 넣는 짧은 top-1 RAG prompt
# ---------------------------------------------------------------------------


_GENERIC_TERMS = {
    "사진",
    "이미지",
    "그림",
    "대상",
    "설명",
    "내용",
    "이름",
    "무엇",
    "해당",
    "다음",
    "우리나라",
    "한국",
    "답하시오",
    "고르시오",
}


def _clean_rag_text(text):
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", str(text))
    text = re.sub(r"\r\n?", "\n", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _query_terms(question, options):
    text = " ".join([str(question), *map(str, options)])
    terms = re.findall(r"[0-9A-Za-z가-힣]{2,}", text)
    return [term for term in terms if term not in _GENERIC_TERMS]


def _relevant_excerpt(description, question, options, title, limit):
    description = _clean_rag_text(description)
    if len(description) <= limit:
        return description
    pieces = [
        piece.strip()
        for piece in re.split(r"\n+|(?<=[.!?。])\s+", description)
        if piece.strip()
    ]
    if not pieces:
        return description[:limit].rstrip()
    terms = _query_terms(question, options)
    title_terms = re.findall(r"[0-9A-Za-z가-힣]{2,}", str(title))
    scored = []
    for index, piece in enumerate(pieces):
        compact = piece.casefold()
        score = sum(2.0 for term in title_terms if term.casefold() in compact)
        score += sum(1.0 for term in terms if term.casefold() in compact)
        if index == 0:
            score += 0.25
        scored.append((score, index, piece))
    selected = sorted(scored, key=lambda item: (-item[0], item[1]))[:4]
    selected.sort(key=lambda item: item[1])
    output = " ".join(piece for _, _, piece in selected).strip()
    return output[:limit].rstrip()


def champion_build_answer_feature(
    sample,
    question,
    options,
    candidates,
    max_rag_chars=None,
):
    question_form = sample["question_form"]
    system_prompt = (
        f"{aivqa_data.SYSTEM_PROMPT}\n\n"
        f"{aivqa_data.QUESTION_FORM_INSTRUCTIONS[question_form]}\n\n"
        "참고자료는 검색 신뢰 기준을 통과한 경우에만 제공됩니다. "
        "그래도 이미지와 질문에 맞지 않으면 참고자료를 무시하세요."
    )
    parts = [aivqa_data.format_question(question_form, question, options)]
    if candidates:
        candidate = candidates[0]
        payload = candidate.payload
        limit = RAG_CONTEXT_LIMITS[question_form]
        excerpt = _relevant_excerpt(
            payload.get("description", ""),
            question,
            options,
            payload.get("title", ""),
            limit,
        )
        if excerpt:
            retrieval = payload.get("_retrieval", {})
            parts.append(
                "[검색 신뢰 기준을 통과한 참고자료]\n"
                f"제목: {payload.get('title', '')}\n"
                f"출처: {payload.get('source', '')}\n"
                f"관련 설명: {excerpt}\n"
                f"검색 근거: {retrieval.get('reason', 'verified')}"
            )
    return {
        "conversation": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "<image>"},
            {"role": "user", "content": "\n\n".join(parts)},
        ],
        "image": sample["image"],
    }


rag_prompts.build_answer_feature = champion_build_answer_feature
rag_augmentation.build_answer_feature = champion_build_answer_feature
rag_infer.build_answer_feature = champion_build_answer_feature


# ---------------------------------------------------------------------------
# 4. 검색 cache를 20건마다 Drive에 저장하고 재실행 시 재사용
# ---------------------------------------------------------------------------


def _candidate_to_dict(candidate):
    return {
        "doc_id": candidate.doc_id,
        "text_score": float(candidate.text_score),
        "image_score": float(candidate.image_score),
        "payload": candidate.payload,
    }


def _candidate_from_dict(value):
    return rag_prompts.Candidate(
        doc_id=str(value["doc_id"]),
        payload=dict(value["payload"]),
        text_score=float(value.get("text_score", 0.0)),
        image_score=float(value.get("image_score", 0.0)),
    )


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def resumable_retrieve_dataset_candidates(
    model,
    processor,
    dataset,
    retriever,
    *,
    max_length,
    search_max_new_tokens,
    dtype,
    description,
    cache_path=None,
):
    cache_path = Path(cache_path) if cache_path is not None else None
    cache_rows = []
    if cache_path is not None and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("signature") == RETRIEVAL_SIGNATURE:
                cache_rows = list(cached.get("rows", []))
        except Exception as error:
            print("검색 cache를 무시합니다:", error)

    all_candidates = []
    completed_since_save = 0
    for index in tqdm(range(len(dataset)), desc=description, unit="sample"):
        sample = dataset[index]
        question_id = str(sample.get("question_id", index))
        cached_row = cache_rows[index] if index < len(cache_rows) else None
        if (
            isinstance(cached_row, dict)
            and cached_row.get("question_id") == question_id
        ):
            all_candidates.append(
                [_candidate_from_dict(item) for item in cached_row.get("candidates", [])]
            )
            continue

        search_output = rag_infer.generate_one(
            model,
            processor,
            rag_prompts.build_search_feature(sample, sample["question"]),
            max_length,
            search_max_new_tokens,
            dtype,
        )
        search_terms = rag_infer.parse_search_terms(search_output)
        candidates = retriever.retrieve(search_terms, sample["image"])
        row = {
            "question_id": question_id,
            "search_terms": search_terms,
            "candidates": [_candidate_to_dict(item) for item in candidates],
        }
        if index < len(cache_rows):
            cache_rows[index] = row
        else:
            cache_rows.append(row)
        all_candidates.append(candidates)
        completed_since_save += 1
        if cache_path is not None and completed_since_save >= 20:
            _atomic_json(
                cache_path,
                {"signature": RETRIEVAL_SIGNATURE, "rows": cache_rows},
            )
            completed_since_save = 0

    if cache_path is not None:
        _atomic_json(
            cache_path,
            {"signature": RETRIEVAL_SIGNATURE, "rows": cache_rows},
        )
    return all_candidates


rag_augmentation.retrieve_dataset_candidates = resumable_retrieve_dataset_candidates
rag_pipeline.retrieve_dataset_candidates = resumable_retrieve_dataset_candidates


def fixed_run_output_dir(_output_root):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    return RUN_DIR


rag_pipeline.create_run_output_dir = fixed_run_output_dir


# ---------------------------------------------------------------------------
# 5. SA 음절·어절 hard retry + 유형별 안전한 출력 정규화
# ---------------------------------------------------------------------------


_COUNT_RE = re.compile(r"(\d+)\s*(음절|어절)")
_TOKEN_STATS_CACHE = {}


def parse_sa_length_constraint(question_text):
    matches = _COUNT_RE.findall(str(question_text))
    if len(matches) != 1:
        return None
    count_text, unit_kr = matches[0]
    target = int(count_text)
    if target < 1:
        return None
    return ("syllable" if unit_kr == "음절" else "eojeol", target)


def _token_stats(tokenizer):
    cache_key = id(tokenizer)
    if cache_key in _TOKEN_STATS_CACHE:
        return _TOKEN_STATS_CACHE[cache_key]
    vocab_size = len(tokenizer)
    syllables = torch.zeros(vocab_size, dtype=torch.long)
    chunks = torch.zeros(vocab_size, dtype=torch.long)
    starts_ws = torch.zeros(vocab_size, dtype=torch.bool)
    ends_ws = torch.zeros(vocab_size, dtype=torch.bool)
    all_ws = torch.zeros(vocab_size, dtype=torch.bool)
    for start in tqdm(range(0, vocab_size, 4096), desc="SA 길이 토큰 통계"):
        ids = list(range(start, min(start + 4096, vocab_size)))
        texts = tokenizer.batch_decode(
            [[token_id] for token_id in ids],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for offset, text in enumerate(texts):
            if not text:
                continue
            token_id = start + offset
            syllables[token_id] = sum("가" <= char <= "힣" for char in text)
            chunks[token_id] = len(text.split())
            starts_ws[token_id] = text[0].isspace()
            ends_ws[token_id] = text[-1].isspace()
            all_ws[token_id] = not text.strip()
    value = {
        "syllables": syllables,
        "chunks": chunks,
        "starts_ws": starts_ws,
        "ends_ws": ends_ws,
        "all_ws": all_ws,
    }
    _TOKEN_STATS_CACHE[cache_key] = value
    return value


class KoreanLengthLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, spec, eos_token_id):
        self.spec = spec
        self.eos_token_id = int(eos_token_id)
        stats = _token_stats(tokenizer)
        self.syllables = stats["syllables"]
        self.chunks = stats["chunks"]
        self.starts_ws = stats["starts_ws"]
        self.ends_ws = stats["ends_ws"]
        self.all_ws = stats["all_ws"]
        self.prefix_length = None
        self.processed_length = 0
        self.word_count = 0
        self.at_boundary = True

    def _to(self, device):
        if self.syllables.device == device:
            return
        self.syllables = self.syllables.to(device)
        self.chunks = self.chunks.to(device)
        self.starts_ws = self.starts_ws.to(device)
        self.ends_ws = self.ends_ws.to(device)
        self.all_ws = self.all_ws.to(device)

    def __call__(self, input_ids, scores):
        self._to(scores.device)
        if self.prefix_length is None:
            self.prefix_length = input_ids.shape[1]
        generated = input_ids[0, self.prefix_length :]
        unit, target = self.spec
        if unit == "syllable":
            current = int(self.syllables[generated].sum().item())
            remaining = target - current
            if remaining <= 0:
                scores[0, :] = float("-inf")
                scores[0, self.eos_token_id] = 0.0
            else:
                scores[0, self.eos_token_id] = float("-inf")
                scores[0, self.syllables > remaining] = float("-inf")
            return scores

        for token_tensor in generated[self.processed_length :]:
            token_id = int(token_tensor.item())
            token_chunks = int(self.chunks[token_id].item())
            if token_chunks == 0:
                if bool(self.all_ws[token_id].item()):
                    self.at_boundary = True
                continue
            merge = self.word_count > 0 and not self.at_boundary and not bool(
                self.starts_ws[token_id].item()
            )
            self.word_count += token_chunks - 1 if merge else token_chunks
            self.at_boundary = bool(self.ends_ws[token_id].item())
        self.processed_length = len(generated)
        if self.word_count < target:
            scores[0, self.eos_token_id] = float("-inf")
        if self.word_count >= target:
            if self.word_count > 0 and not self.at_boundary:
                effective = torch.where(
                    self.starts_ws, self.chunks, self.chunks - 1
                )
            else:
                effective = self.chunks
            scores[0, effective > 0] = float("-inf")
        return scores


def _strip_thinking(text):
    text = str(text).strip()
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return re.sub(
        r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE
    ).strip()


def _normalize_mc(text):
    text = _strip_thinking(text)
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    match = re.search(r"(?<!\d)([1-5](?:\s*[/,]\s*[1-5])*)(?!\d)", first)
    if not match:
        return "", False
    numbers = sorted(set(re.findall(r"[1-5]", match.group(1))), key=int)
    return "/".join(numbers), True


def _normalize_sa(text):
    text = _strip_thinking(text)
    text = re.sub(r"^(?:정답은|정답|답은|답|answer)\s*[:：]?\s*", "", text, flags=re.I)
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    first = re.sub(r"\s+", " ", first).strip(" \t\"'“”‘’`")
    return first.rstrip("。.!?")


def _normalize_la(text):
    text = _strip_thinking(text)
    text = re.sub(r"^(?:정답|답변|answer)\s*[:：]?\s*", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip(" \t\"'“”‘’`")[:250].rstrip()


def _sa_requirements(question):
    return [
        (int(length), unit)
        for length, unit in re.findall(r"(\d+)\s*(음절|글자|어절)", str(question))
    ]


def _sa_length_match(answer, question):
    requirements = _sa_requirements(question)
    if not requirements:
        return None
    segments = [segment.strip() for segment in str(answer).split("/")]
    if len(requirements) > 1 and len(segments) != len(requirements):
        return False
    if len(requirements) == 1:
        segments = [str(answer).strip()]
    for segment, (target, unit) in zip(segments, requirements):
        actual = (
            len([token for token in segment.split() if token])
            if unit == "어절"
            else len(re.findall(r"[0-9A-Za-z가-힣]", segment))
        )
        if actual != target:
            return False
    return True


def _retry_feature(feature):
    copied = dict(feature)
    conversation = copy.deepcopy(feature["conversation"])
    for message in reversed(conversation):
        if message.get("role") == "user":
            message["content"] += (
                "\n\n이전 출력이 형식 또는 길이 조건을 지키지 못했습니다. "
                "정답을 다시 판단하고 출력 전에 음절·어절 수를 확인한 뒤 "
                "정답만 출력하세요."
            )
            break
    copied["conversation"] = conversation
    return copied


def _generate_once(
    model,
    processor,
    feature,
    max_length,
    dtype,
    *,
    retry=False,
    enforce_length=True,
):
    form = feature["question_form"]
    used_feature = _retry_feature(feature) if retry else feature
    batch = train_lora._move_batch_to_device(
        rag_infer.collate_generation_feature(processor, used_feature, max_length),
        train_lora._model_input_device(model),
    )
    kwargs = {
        "max_new_tokens": {"MC": 16, "SA": 32, "LA": 256}[form],
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "pad_token_id": processor.tokenizer.pad_token_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
    }
    spec = (
        parse_sa_length_constraint(feature.get("question", ""))
        if retry and enforce_length and form == "SA"
        else None
    )
    if spec is not None:
        kwargs["logits_processor"] = LogitsProcessorList(
            [
                KoreanLengthLogitsProcessor(
                    processor.tokenizer,
                    spec,
                    processor.tokenizer.eos_token_id,
                )
            ]
        )
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
        generated = model.generate(**batch, **kwargs)
    return processor.batch_decode(
        generated.detach().cpu(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def generate_champion_answer(model, processor, feature, max_length, dtype):
    form = feature["question_form"]
    primary_raw = _generate_once(
        model, processor, feature, max_length, dtype, retry=False
    )
    if form == "MC":
        primary, valid = _normalize_mc(primary_raw)
        if not valid:
            retry_raw = _generate_once(
                model, processor, feature, max_length, dtype, retry=True
            )
            retry, retry_valid = _normalize_mc(retry_raw)
            if retry_valid:
                return retry
        return primary if valid else "1"
    if form == "SA":
        primary = _normalize_sa(primary_raw)
        length_match = _sa_length_match(primary, feature.get("question", ""))
        if not primary or length_match is False:
            try:
                retry_raw = _generate_once(
                    model, processor, feature, max_length, dtype, retry=True
                )
            except (RuntimeError, TypeError, ValueError) as error:
                print("SA 길이 hard retry를 soft retry로 대체합니다:", error)
                retry_raw = _generate_once(
                    model,
                    processor,
                    feature,
                    max_length,
                    dtype,
                    retry=True,
                    enforce_length=False,
                )
            retry = _normalize_sa(retry_raw)
            retry_match = _sa_length_match(retry, feature.get("question", ""))
            if retry and retry_match is True:
                return retry
            if not primary and retry:
                return retry
        return primary or "확인 불가"
    return _normalize_la(primary_raw) or "확인 불가"


VALIDATION_EVAL_LOG = []
_EVAL_CALLS = Counter()


def champion_evaluate_generation(
    model,
    processor,
    dataset,
    batch_size,
    max_length,
    max_new_tokens,
    dtype,
):
    predictions = []
    references = []
    forms = []
    question_ids = []
    for index in tqdm(range(len(dataset)), desc="champion validation", leave=False):
        sample = dataset[index]
        predictions.append(
            generate_champion_answer(model, processor, sample, max_length, dtype)
        )
        references.append(str(sample["answer"]))
        forms.append(sample["question_form"])
        question_ids.append(sample.get("question_id", str(index)))
    metrics = aivqa_metrics.compute_vqa_metrics(predictions, references, forms)
    form = forms[0] if forms else "UNKNOWN"
    _EVAL_CALLS[form] += 1
    VALIDATION_EVAL_LOG.append(
        {
            "question_form": form,
            "epoch_call": _EVAL_CALLS[form],
            "metrics": metrics,
            "question_ids": question_ids,
            "predictions": predictions,
            "references": references,
        }
    )
    return predictions, metrics


def champion_generate_rag_predictions(
    model,
    processor,
    dataset,
    *,
    max_length,
    max_new_tokens,
    dtype,
    description,
):
    return [
        generate_champion_answer(model, processor, dataset[index], max_length, dtype)
        for index in tqdm(range(len(dataset)), desc=description, unit="sample")
    ]


type_train.evaluate_generation = champion_evaluate_generation
rag_pipeline.generate_rag_predictions = champion_generate_rag_predictions
rag_augmentation.generate_rag_predictions = champion_generate_rag_predictions


# SA는 더 오래 탐색하되 validation exact-match 기준 최고 epoch만 저장합니다.
_original_train_question_form = type_train.train_question_form


def train_question_form_with_budget(args, question_form, *positional, **keyword):
    local_args = copy.copy(args)
    local_args.epochs = TYPE_EPOCHS_BY_FORM[question_form]
    local_args.early_stopping_patience = TYPE_EPOCHS_BY_FORM[question_form]
    return _original_train_question_form(
        local_args, question_form, *positional, **keyword
    )


rag_pipeline.train_question_form = train_question_form_with_budget


print("Champion runtime patch 완료")
print("- Kanana revision/SDPA:", KANANA_REVISION)
print("- Selective RAG top-k:", RAG_TOP_K)
print("- 유형별 epochs:", TYPE_EPOCHS_BY_FORM)
print("- SA length mode: retry-only hard logits")
