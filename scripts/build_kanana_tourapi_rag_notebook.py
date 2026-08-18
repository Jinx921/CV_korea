import json
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / "notebooks" / "kanana15v_3b_lora_paddle_crop_typed.ipynb"
OUTPUT_PATH = ROOT / "notebooks" / "kanana15v_3b_lora_tourapi_rag_typed.ipynb"


def source_lines(text):
    return text.strip("\n").splitlines(keepends=True)


def set_cell(notebook, number, text):
    notebook["cells"][number - 1]["source"] = source_lines(text)


notebook = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
notebook["metadata"]["colab"]["name"] = OUTPUT_PATH.name


set_cell(
    notebook,
    1,
    r"""
## Kanana-1.5-V 3B 유형별 LoRA + TourAPI 멀티모달 RAG

현재 구축한 TourAPI Qdrant DB 2,793건만 사용해 RAG 효과를 확인하는 첫 실험입니다.

1. 대회 질문은 BGE-M3로 텍스트 검색
2. 대회 이미지는 SigLIP2로 이미지 유사도 검색
3. SigLIP2 텍스트→이미지 검색을 추가하고 RRF로 결과 결합
4. 강한 검색 근거가 있는 문항에만 상위 3개 제목·설명을 Kanana 프롬프트에 추가
5. train·validation·test 모두 동일한 검색 방식을 적용
6. 검색 캐시 생성 후 임베딩 모델을 GPU에서 내리고 Kanana 학습·추론 진행
7. 기존 Kanana 유형별 LoRA 기준 점수 `44.8395439`와 비교

이번 버전은 검색된 외부 이미지를 Kanana에 직접 추가하지 않고, 이미지 검색으로 찾은 문서의 텍스트 설명만 사용합니다. 첫 실험에서 메모리 사용량과 검색 노이즈를 낮추기 위한 설정입니다.
""",
)


set_cell(
    notebook,
    2,
    r"""
### 0. Colab 런타임 준비

L4 또는 A100 GPU 런타임에서 실행하세요. 첫 실행에서는 패키지를 설치한 뒤 런타임이 한 번 자동 재시작됩니다.
재연결되면 `런타임 → 모두 실행`을 다시 누르세요.
""",
)


set_cell(
    notebook,
    3,
    r"""
import os
import signal
import subprocess
import sys
from pathlib import Path


restart_marker = Path("/content/.kanana15v_tourapi_rag_env_v1")

if not restart_marker.exists():
    def run_pip(*args):
        subprocess.check_call([sys.executable, "-m", "pip", *args])

    run_pip("uninstall", "-q", "-y", "gradio", "gradio_client", "torchao", "torchaudio", "Pillow")
    run_pip(
        "install", "-q", "-U",
        "transformers>=4.57.0,<5",
        "huggingface-hub==0.36.2",
        "sentence-transformers>=3.4,<6",
        "qdrant-client>=1.14,<2",
        "accelerate>=1.2",
        "peft>=0.14",
        "datasets>=3.2",
        "timm>=1.0",
        "omegaconf>=2.3",
        "sentencepiece",
        "safetensors",
        "pandas>=2.2",
        "tqdm>=4.66",
    )
    run_pip("install", "-q", "--no-cache-dir", "--force-reinstall", "Pillow==11.3.0")
    restart_marker.write_text("ready\n", encoding="utf-8")
    print("패키지 설치 완료. 런타임을 자동 재시작합니다.")
    print("재연결된 뒤 '모두 실행'을 다시 누르세요.")
    os.kill(os.getpid(), signal.SIGKILL)

import PIL
from PIL import Image, ImageOps

print("Pillow 정상 로드:", PIL.__version__)
""",
)


set_cell(notebook, 4, "### 1. Drive 연결 및 실행 설정")


set_cell(
    notebook,
    6,
    r"""
import copy
import gc
import hashlib
import json
import os
import random
import re
import shutil
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from tqdm.auto import tqdm


# 기본 경로
DRIVE_PROJECT_DIR = Path("/content/drive/MyDrive/CV_korea")
DRIVE_ZIP_DIR = DRIVE_PROJECT_DIR / "data"
RAG_DB_DIR = DRIVE_PROJECT_DIR / "tourapi_image_db"
QDRANT_ARCHIVE = RAG_DB_DIR / "tourapi_multimodal_qdrant.zip"
QDRANT_STATE_PATH = RAG_DB_DIR / "tourapi_multimodal_qdrant_state.json"

MODEL_ID = "kakaocorp/kanana-1.5-v-3b-instruct"
# 현재 transformers의 Qwen2-VL position_embeddings 호출 방식과 호환되는 공식 revision입니다.
# 아래 모델 로드 셀에서 이 revision의 FlashAttention2 강제 시도만 SDPA로 우회합니다.
KANANA_REVISION = "2e00ef13ccec2e99459a8eada18a1bfd05bff44b"
TEXT_MODEL_NAME = "BAAI/bge-m3"
IMAGE_MODEL_NAME = "google/siglip2-base-patch16-224"
COLLECTION_NAME = "tourapi_multimodal"
TEXT_VECTOR_NAME = "text_vector"
IMAGE_VECTOR_NAME = "image_vector"

RUN_NAME = "kanana15v_3b_lora_tourapi_rag_typed"
TRAINING_METHOD = "shared_then_typed_lora_with_selective_tourapi_rag"

# True: RAG-aware Shared + 유형별 Adapter 학습 / False: 저장 Adapter로 추론만 수행
DO_TRAIN = True
REUSE_COMPLETED_ADAPTERS = True

TRAIN_MAX_SAMPLES = None
VALIDATION_MAX_SAMPLES = 200
SHARED_EPOCHS = 1
TYPED_EPOCHS = 1
LEARNING_RATE = 1e-4
GRADIENT_ACCUMULATION_STEPS = 16
TRAIN_MAX_IMAGE_SIDE = 896
INFERENCE_MAX_IMAGE_SIDE = 1344
MAX_SEQUENCE_LENGTH = 4096

# RAG 설정
RAG_TOP_K = 3
RAG_CANDIDATE_K = 20
RAG_DOC_CHAR_LIMIT = 700
RAG_RRF_K = 60
RAG_QUERY_TEXT_BATCH_SIZE = 32
RAG_QUERY_IMAGE_BATCH_SIZE = 32

# 약한 검색 결과가 모든 프롬프트에 들어가는 것을 막습니다.
RAG_REQUIRE_STRONG_MATCH = True
RAG_MIN_BGE_SCORE = 0.50
RAG_MIN_IMAGE_SCORE = 0.92
RAG_MIN_SIGLIP_TEXT_SCORE = 0.20

RESET_RAG_CACHE = False
RAG_CACHE_SAVE_EVERY = 20
RESET_PREDICTION_CACHE = False
SAVE_EVERY_N_PREDICTIONS = 20
SEED = 42
REFERENCE_BASELINE_SCORE = 44.8395439

COLAB_WORK_DIR = Path("/content/kculture_kanana15v_tourapi_rag")
EXTRACT_DIR = COLAB_WORK_DIR / "extracted"
QDRANT_WORK_DIR = COLAB_WORK_DIR / "qdrant"
CHECKPOINT_ROOT = DRIVE_PROJECT_DIR / "checkpoints" / RUN_NAME
RUN_OUTPUT_DIR = DRIVE_PROJECT_DIR / "outputs" / RUN_NAME
RAG_CACHE_PATH = RUN_OUTPUT_DIR / "rag_retrieval_cache.json"
SHARED_ADAPTER_DIR = RUN_OUTPUT_DIR / "adapter_shared"
FORM_ADAPTER_DIRS = {
    form: RUN_OUTPUT_DIR / f"adapter_{form.lower()}" for form in ["MC", "SA", "LA"]
}

for path in [
    COLAB_WORK_DIR,
    EXTRACT_DIR,
    CHECKPOINT_ROOT,
    RUN_OUTPUT_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("대회 ZIP 폴더:", DRIVE_ZIP_DIR)
print("RAG DB ZIP:", QDRANT_ARCHIVE)
print("결과 폴더:", RUN_OUTPUT_DIR)
print("모델:", MODEL_ID)
print("RAG: BGE-M3 + SigLIP2 + Qdrant, 선택적 top-k", RAG_TOP_K)
print("학습: Shared", SHARED_EPOCHS, "epoch + 유형별", TYPED_EPOCHS, "epoch")
""",
)


set_cell(
    notebook,
    7,
    r"""
if not torch.cuda.is_available():
    raise RuntimeError("GPU 런타임이 아닙니다. Colab 런타임 유형을 GPU로 변경하세요.")

gpu_name = torch.cuda.get_device_name(0)
gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
bf16_supported = torch.cuda.is_bf16_supported()

print(f"GPU: {gpu_name}")
print(f"VRAM: {gpu_mem_gb:.1f} GiB")
print(f"BF16 지원: {bf16_supported}")

if DO_TRAIN and gpu_mem_gb < 20:
    raise RuntimeError(
        "Kanana-V 3B 일반 LoRA에는 20GB 이상의 VRAM을 권장합니다. "
        "Colab L4 또는 A100 런타임으로 변경하세요."
    )

print("GPU 점검 통과")
""",
)


set_cell(notebook, 12, "### 4. TourAPI Qdrant DB 복원")


set_cell(
    notebook,
    13,
    r"""
from qdrant_client import QdrantClient


def atomic_write_json(data, path):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


assert QDRANT_ARCHIVE.exists(), f"Qdrant ZIP이 없습니다: {QDRANT_ARCHIVE}"
assert QDRANT_STATE_PATH.exists(), f"Qdrant 상태 JSON이 없습니다: {QDRANT_STATE_PATH}"

qdrant_state = json.loads(QDRANT_STATE_PATH.read_text(encoding="utf-8"))
expected_state = {
    "collection_name": COLLECTION_NAME,
    "text_model": TEXT_MODEL_NAME,
    "image_model": IMAGE_MODEL_NAME,
}
state_mismatch = {
    key: (qdrant_state.get(key), expected)
    for key, expected in expected_state.items()
    if qdrant_state.get(key) != expected
}
if state_mismatch:
    raise RuntimeError(f"RAG DB 설정이 노트북 설정과 다릅니다: {state_mismatch}")

if QDRANT_WORK_DIR.exists():
    shutil.rmtree(QDRANT_WORK_DIR)
QDRANT_WORK_DIR.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(QDRANT_ARCHIVE, "r") as archive:
    archive.extractall(QDRANT_WORK_DIR)

qdrant_client = QdrantClient(path=str(QDRANT_WORK_DIR))
if not qdrant_client.collection_exists(COLLECTION_NAME):
    raise RuntimeError(f"Qdrant 컬렉션을 찾지 못했습니다: {COLLECTION_NAME}")

collection_info = qdrant_client.get_collection(COLLECTION_NAME)
print("Qdrant DB 복원 완료")
print("컬렉션:", COLLECTION_NAME)
print("points_count:", collection_info.points_count)
print("상태 JSON indexed_doc_id_count:", qdrant_state["indexed_doc_id_count"])
""",
)


set_cell(notebook, 14, "### 5. Train·Validation·Test RAG 검색 캐시 생성")


set_cell(
    notebook,
    15,
    r"""
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoProcessor


def rag_cache_key(record, split):
    return f"{split}:{record['metadata']['question_id']}"


def rag_query_text(record):
    model_input = record["model_input"]
    parts = [str(model_input.get("question", "")).strip()]
    options = model_input.get("options") or []
    if options:
        parts.append("선택지: " + " / ".join(str(option) for option in options))
    return "\n".join(part for part in parts if part)


rag_config_for_cache = {
    "db_saved_at": qdrant_state.get("saved_at"),
    "indexed_doc_id_count": qdrant_state.get("indexed_doc_id_count"),
    "text_model": TEXT_MODEL_NAME,
    "image_model": IMAGE_MODEL_NAME,
    "top_k": RAG_TOP_K,
    "candidate_k": RAG_CANDIDATE_K,
    "rrf_k": RAG_RRF_K,
    "require_strong_match": RAG_REQUIRE_STRONG_MATCH,
    "min_bge_score": RAG_MIN_BGE_SCORE,
    "min_image_score": RAG_MIN_IMAGE_SCORE,
    "min_siglip_text_score": RAG_MIN_SIGLIP_TEXT_SCORE,
}

if RESET_RAG_CACHE and RAG_CACHE_PATH.exists():
    RAG_CACHE_PATH.unlink()

if RAG_CACHE_PATH.exists():
    rag_cache = json.loads(RAG_CACHE_PATH.read_text(encoding="utf-8"))
    if rag_cache.get("config") != rag_config_for_cache:
        print("DB 또는 RAG 설정이 바뀌어 기존 검색 캐시를 초기화합니다.")
        rag_cache = {"config": rag_config_for_cache, "items": {}}
else:
    rag_cache = {"config": rag_config_for_cache, "items": {}}


pending = []
for split in ["train", "validation", "test"]:
    for record in records[split]:
        key = rag_cache_key(record, split)
        query_hash = hashlib.sha256(
            (rag_query_text(record) + "\n" + record["model_input"]["image_name"]).encode("utf-8")
        ).hexdigest()
        cached = rag_cache["items"].get(key)
        if cached is None or cached.get("query_hash") != query_hash:
            pending.append((split, record, query_hash))

print(f"재사용 RAG 검색 캐시: {3000 - len(pending):,} / 3,000")
print(f"이번 검색 대상: {len(pending):,}")


def ensure_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "pooler_output") and value.pooler_output is not None:
        return value.pooler_output
    if isinstance(value, (tuple, list)) and value:
        return value[0]
    raise TypeError(f"지원하지 않는 임베딩 반환 형식: {type(value)}")


def open_rgb_image(path):
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB").copy()


def encode_siglip_images(paths, processor, model, batch_size):
    chunks = []
    for start in tqdm(range(0, len(paths), batch_size), desc="질의 이미지 임베딩"):
        batch_paths = paths[start : start + batch_size]
        images = [open_rgb_image(path) for path in batch_paths]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        with torch.inference_mode():
            features = ensure_tensor(model.get_image_features(**inputs))
            features = torch.nn.functional.normalize(features.float(), p=2, dim=-1)
        chunks.append(features.cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def encode_siglip_texts(texts, processor, model, batch_size):
    chunks = []
    for start in tqdm(range(0, len(texts), batch_size), desc="SigLIP2 질의 텍스트 임베딩"):
        batch_texts = texts[start : start + batch_size]
        inputs = processor(
            text=batch_texts,
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        with torch.inference_mode():
            features = ensure_tensor(model.get_text_features(**inputs))
            features = torch.nn.functional.normalize(features.float(), p=2, dim=-1)
        chunks.append(features.cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def search_channel(vector, vector_name):
    return qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=np.asarray(vector, dtype=np.float32).tolist(),
        using=vector_name,
        limit=RAG_CANDIDATE_K,
        with_payload=True,
        with_vectors=False,
    ).points


def merge_rag_results(bge_points, siglip_text_points, image_points):
    channel_points = {
        "bge_text": bge_points,
        "siglip_text_to_image": siglip_text_points,
        "siglip_image": image_points,
    }
    merged = {}
    for channel, points in channel_points.items():
        for rank, point in enumerate(points, start=1):
            payload = dict(point.payload or {})
            doc_id = str(payload.get("doc_id", point.id))
            if doc_id not in merged:
                merged[doc_id] = {
                    "doc_id": doc_id,
                    "title": str(payload.get("title", "")),
                    "description": str(payload.get("description", "")),
                    "image_path": str(payload.get("image_path", "")),
                    "image_url": str(payload.get("image_url", "")),
                    "source": str(payload.get("source", "tourapi")),
                    "rrf_score": 0.0,
                    "channels": [],
                    "channel_scores": {},
                }
            merged[doc_id]["rrf_score"] += 1.0 / (RAG_RRF_K + rank)
            merged[doc_id]["channels"].append(channel)
            merged[doc_id]["channel_scores"][channel] = float(point.score)

    ranked = sorted(merged.values(), key=lambda row: row["rrf_score"], reverse=True)
    top_bge = float(bge_points[0].score) if bge_points else -1.0
    top_siglip_text = float(siglip_text_points[0].score) if siglip_text_points else -1.0
    top_image = float(image_points[0].score) if image_points else -1.0
    multi_channel_match = bool(ranked and len(set(ranked[0]["channels"])) >= 2)
    strong_match = (
        top_bge >= RAG_MIN_BGE_SCORE
        or top_siglip_text >= RAG_MIN_SIGLIP_TEXT_SCORE
        or top_image >= RAG_MIN_IMAGE_SCORE
        or multi_channel_match
    )
    applied = bool(ranked) and (strong_match or not RAG_REQUIRE_STRONG_MATCH)

    return {
        "rag_applied": applied,
        "strong_match": strong_match,
        "top_bge_score": top_bge,
        "top_siglip_text_score": top_siglip_text,
        "top_image_score": top_image,
        "documents": ranked[:RAG_TOP_K] if applied else [],
    }


if pending:
    text_encoder = SentenceTransformer(TEXT_MODEL_NAME, device="cuda")
    siglip_processor = AutoProcessor.from_pretrained(IMAGE_MODEL_NAME)
    siglip_encoder = AutoModel.from_pretrained(
        IMAGE_MODEL_NAME,
        dtype=torch.float16,
    ).to("cuda").eval()

    pending_texts = [rag_query_text(record) for _, record, _ in pending]
    bge_vectors = text_encoder.encode(
        pending_texts,
        batch_size=RAG_QUERY_TEXT_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)
    siglip_text_vectors = encode_siglip_texts(
        pending_texts,
        siglip_processor,
        siglip_encoder,
        RAG_QUERY_TEXT_BATCH_SIZE,
    )
    pending_image_paths = [
        str(image_dirs[split] / record["model_input"]["image_name"])
        for split, record, _ in pending
    ]
    image_vectors = encode_siglip_images(
        pending_image_paths,
        siglip_processor,
        siglip_encoder,
        RAG_QUERY_IMAGE_BATCH_SIZE,
    )

    completed_since_save = 0
    for index, (split, record, query_hash) in enumerate(tqdm(pending, desc="Qdrant RAG 검색")):
        bge_points = search_channel(bge_vectors[index], TEXT_VECTOR_NAME)
        siglip_text_points = search_channel(siglip_text_vectors[index], IMAGE_VECTOR_NAME)
        image_points = search_channel(image_vectors[index], IMAGE_VECTOR_NAME)
        item = merge_rag_results(bge_points, siglip_text_points, image_points)
        item["query_hash"] = query_hash
        item["split"] = split
        item["question_id"] = record["metadata"]["question_id"]
        rag_cache["items"][rag_cache_key(record, split)] = item
        completed_since_save += 1
        if completed_since_save >= RAG_CACHE_SAVE_EVERY:
            atomic_write_json(rag_cache, RAG_CACHE_PATH)
            completed_since_save = 0

    atomic_write_json(rag_cache, RAG_CACHE_PATH)
    del text_encoder, siglip_processor, siglip_encoder
    del bge_vectors, siglip_text_vectors, image_vectors

qdrant_client.close()
gc.collect()
torch.cuda.empty_cache()


def rag_item_for(record, split):
    key = rag_cache_key(record, split)
    if key not in rag_cache["items"]:
        raise KeyError(f"RAG 검색 캐시가 없습니다: {key}")
    return rag_cache["items"][key]


def rag_context_for(record, split):
    item = rag_item_for(record, split)
    if not item.get("rag_applied"):
        return ""
    blocks = []
    for rank, document in enumerate(item.get("documents", [])[:RAG_TOP_K], start=1):
        description = str(document.get("description", "")).strip()[:RAG_DOC_CHAR_LIMIT]
        blocks.append(
            "\n".join(
                [
                    f"[검색 자료 {rank}]",
                    f"제목: {document.get('title', '')}",
                    f"설명: {description}",
                    f"출처: {document.get('source', 'tourapi')}",
                ]
            )
        )
    return "\n\n".join(blocks)


rag_summary_rows = []
for split in ["train", "validation", "test"]:
    split_items = [rag_item_for(record, split) for record in records[split]]
    rag_summary_rows.append(
        {
            "split": split,
            "rows": len(split_items),
            "rag_applied": sum(bool(item["rag_applied"]) for item in split_items),
            "mean_top_bge": float(np.mean([item["top_bge_score"] for item in split_items])),
            "mean_top_image": float(np.mean([item["top_image_score"] for item in split_items])),
        }
    )

display(pd.DataFrame(rag_summary_rows))
print("RAG 검색 캐시:", RAG_CACHE_PATH)
print("검색 모델 GPU 메모리 해제 완료")
""",
)


set_cell(notebook, 16, "### 6. RAG 근거를 포함한 유형별 프롬프트")


set_cell(
    notebook,
    17,
    r"""
SYSTEM_PROMPT = (
    "당신은 이미지와 검색 자료를 함께 이해하는 한국문화 질의응답 모델입니다. "
    "검색 자료는 자동 검색 결과이므로 관련 없는 내용일 수 있습니다. "
    "이미지와 질문을 우선하고, 도움이 되는 검색 자료만 사용하세요. "
    "사용자가 요구한 정답 형식만 출력하고 풀이 과정은 출력하지 마세요."
)


def build_user_text(record, split, retry=False):
    form = record["metadata"]["question_form"]
    model_input = record["model_input"]
    question = model_input["question"].strip()
    options = model_input.get("options") or []
    rag_context = rag_context_for(record, split)

    parts = []
    if rag_context:
        parts.append(
            "다음은 자동 검색된 참고자료입니다. 질문과 관련 있는 내용만 사용하세요.\n\n"
            + rag_context
        )
    parts.append("질문:\n" + question)
    if options:
        parts.append("선택지:\n" + "\n".join(str(option) for option in options))

    if form == "MC":
        instruction = (
            "선다형입니다. 정답 선택지 번호만 출력하세요. "
            "복수 정답이면 번호를 오름차순으로 /로 연결하세요(예: 1/3/5). "
            "정답 이외의 설명은 쓰지 마세요."
        )
    elif form == "SA":
        instruction = (
            "단답형입니다. 질문에 명시된 음절·글자·어절 수를 지키고 정답만 출력하세요. "
            "복수 요소를 요구하면 질문의 순서대로 /로 연결하세요."
        )
    elif form == "LA":
        instruction = (
            "서술형입니다. 이미지 정보와 필요한 한국문화 지식을 반영해 한 문장 이상, "
            "250자 이하로 답하세요. 답변 본문만 출력하세요."
        )
    else:
        raise ValueError(f"알 수 없는 question_form: {form}")

    if retry:
        instruction += " 이전 출력 형식이 잘못되었습니다. 이번에는 반드시 지정 형식만 출력하세요."
    parts.append(instruction)
    return "\n\n".join(parts)


def build_conversation(record, split, image_count, answer=None, add_answer=False, retry=False):
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": " ".join(["<image>"] * image_count)},
        {"role": "user", "content": build_user_text(record, split, retry=retry)},
    ]
    if add_answer:
        conversation.append({"role": "assistant", "content": str(answer)})
    return conversation


for form in ["MC", "SA", "LA"]:
    example = next(row for row in records["train"] if row["metadata"]["question_form"] == form)
    print("=" * 80)
    print(form, example["metadata"]["question_id"])
    print("RAG 적용:", rag_item_for(example, "train")["rag_applied"])
    print(build_user_text(example, "train"))
    print("정답:", example["model_output"]["answer"])
""",
)


set_cell(notebook, 18, "### 7. Kanana-1.5-V 3B BF16/FP16 로드")


set_cell(
    notebook,
    19,
    r"""
from transformers import AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module


compute_dtype = torch.bfloat16 if bf16_supported else torch.float16

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    revision=KANANA_REVISION,
    trust_remote_code=True,
)
processor.tokenizer.padding_side = "right"
if processor.tokenizer.pad_token_id is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token


def load_base_model():
    # Kanana 공식 remote code는 내부 vision encoder를 FlashAttention2로 먼저
    # 만들도록 고정되어 있습니다. Colab 기본 환경에는 flash_attn이 없으므로,
    # 동적 모델 클래스를 먼저 불러온 뒤 vision encoder의 _from_config 호출만
    # PyTorch SDPA로 고정합니다. 모델 forward 코드는 공식 revision 그대로 씁니다.
    model_class = get_class_from_dynamic_module(
        "modeling.KananaVForConditionalGeneration",
        MODEL_ID,
        revision=KANANA_REVISION,
    )
    vision_class = model_class.__init__.__globals__["CustomQwen2VLVE"]
    original_from_config = vision_class._from_config.__func__

    @classmethod
    def from_config_with_sdpa(cls, config, **kwargs):
        kwargs["attn_implementation"] = "sdpa"
        return original_from_config(cls, config, **kwargs)

    vision_class._from_config = from_config_with_sdpa
    return model_class.from_pretrained(
        MODEL_ID,
        revision=KANANA_REVISION,
        dtype=compute_dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )


def set_model_cache(enabled, target_model=None):
    target = target_model if target_model is not None else model
    if hasattr(target.config, "use_cache"):
        target.config.use_cache = enabled
    if hasattr(target.config, "text_config"):
        target.config.text_config.use_cache = enabled
    language_model = getattr(target, "language_model", None)
    if language_model is not None and hasattr(language_model, "config"):
        language_model.config.use_cache = enabled


gc.collect()
torch.cuda.empty_cache()
model = load_base_model()
set_model_cache(not DO_TRAIN)
torch.set_float32_matmul_precision("high")

print("모델 로드 완료:", MODEL_ID)
print("Kanana revision:", KANANA_REVISION)
print("학습 방식: 일반 LoRA (양자화 없음)")
print("원본 모델 dtype:", compute_dtype)
print("Attention backend: PyTorch SDPA")
print("모델 입력 장치:", next(model.parameters()).device)
""",
)


set_cell(notebook, 20, "### 8. 이미지 로더·응답 정규화·Kanana 추론 함수")


set_cell(
    notebook,
    21,
    r"""
Image.MAX_IMAGE_PIXELS = None


def image_path_for(record, split):
    return image_dirs[split] / record["model_input"]["image_name"]


def load_rgb_image(path, max_side=None):
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        if max_side is not None and max(image.size) > max_side:
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return image.copy()


def build_input_images(record, split, max_image_side):
    # 첫 실험에서는 대회 원본 이미지만 Kanana에 넣습니다.
    # TourAPI 이미지 벡터는 관련 설명을 검색하는 데 사용됩니다.
    return [load_rgb_image(image_path_for(record, split), max_image_side)]


def strip_thinking(text):
    text = str(text).strip()
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def normalize_mc(text):
    cleaned = strip_thinking(text)
    cleaned = re.sub(r"^(정답|답|answer)\s*[:：]?\s*", "", cleaned, flags=re.IGNORECASE)
    first_line = next((line.strip() for line in cleaned.splitlines() if line.strip()), "")
    match = re.search(r"(?<!\d)([1-5](?:\s*[/,]\s*[1-5])*)(?!\d)", first_line)
    if not match:
        return "", False
    numbers = sorted(set(re.findall(r"[1-5]", match.group(1))), key=int)
    answer = "/".join(numbers)
    return answer, bool(re.fullmatch(r"[1-5](?:/[1-5])*", answer))


def normalize_short_answer(text):
    cleaned = strip_thinking(text)
    cleaned = re.sub(r"^(정답|답|answer)\s*[:：]?\s*", "", cleaned, flags=re.IGNORECASE)
    first_line = next((line.strip() for line in cleaned.splitlines() if line.strip()), "")
    first_line = re.sub(r"\s+", " ", first_line).strip(" \t\"'“”‘’")
    return first_line.rstrip("。")


def normalize_long_answer(text):
    cleaned = strip_thinking(text)
    cleaned = re.sub(r"^(정답|답변|answer)\s*[:：]?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t\"'“”‘’")
    return cleaned[:250].rstrip()


def normalize_prediction(text, form):
    if form == "MC":
        return normalize_mc(text)
    if form == "SA":
        answer = normalize_short_answer(text)
        return answer, bool(answer)
    if form == "LA":
        answer = normalize_long_answer(text)
        return answer, bool(answer) and len(answer) <= 250
    raise ValueError(form)


def max_new_tokens_for(form):
    return {"MC": 16, "SA": 64, "LA": 256}[form]


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


INFERENCE_USES_TYPED_ADAPTERS = False
ACTIVE_ADAPTER_NAME = None


def activate_adapter(form):
    global ACTIVE_ADAPTER_NAME
    if not INFERENCE_USES_TYPED_ADAPTERS:
        return
    if ACTIVE_ADAPTER_NAME != form:
        model.language_model.set_adapter(form)
        ACTIVE_ADAPTER_NAME = form


@torch.inference_mode()
def generate_raw(record, split, retry=False):
    form = record["metadata"]["question_form"]
    activate_adapter(form)
    images = build_input_images(record, split, INFERENCE_MAX_IMAGE_SIDE)
    sample = {
        "image": images,
        "conv": build_conversation(record, split, len(images), retry=retry),
    }
    inputs = processor.batch_encode_collate(
        [sample],
        padding_side="left",
        add_generation_prompt=True,
        max_length=MAX_SEQUENCE_LENGTH,
    )
    device = next(model.parameters()).device
    inputs = move_to_device(inputs, device)
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens_for(form),
        temperature=0,
        top_p=1.0,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    raw_text = processor.tokenizer.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    del images, sample, inputs, generated
    return raw_text


def predict_one(record, split):
    form = record["metadata"]["question_form"]
    raw_text = generate_raw(record, split, retry=False)
    answer, valid = normalize_prediction(raw_text, form)
    retried = False

    if not valid:
        retried = True
        retry_raw = generate_raw(record, split, retry=True)
        retry_answer, retry_valid = normalize_prediction(retry_raw, form)
        raw_text = raw_text + "\n[RETRY]\n" + retry_raw
        if retry_valid:
            answer, valid = retry_answer, True

    if not valid:
        answer = "1" if form == "MC" else "확인 불가"

    return {"answer": answer, "raw": raw_text, "retried": retried, "valid": valid}
""",
)


set_cell(
    notebook,
    23,
    r"""
smoke_record = records["validation"][0]
smoke_rag = rag_item_for(smoke_record, "validation")

print("ID:", smoke_record["metadata"]["question_id"])
print("질문:", smoke_record["model_input"]["question"])
print("정답:", smoke_record["model_output"]["answer"])
print("RAG 적용:", smoke_rag["rag_applied"])
print("RAG 문서:", [doc["title"] for doc in smoke_rag["documents"]])
print("RAG 점수:", {
    "bge": smoke_rag["top_bge_score"],
    "siglip_text": smoke_rag["top_siglip_text_score"],
    "image": smoke_rag["top_image_score"],
})

smoke_result = predict_one(smoke_record, "validation")
print("예측:", smoke_result["answer"])
print("원본 출력:", smoke_result["raw"])
""",
)


# Collator에서 split 인수를 전달하도록 수정합니다.
cell_25 = "".join(notebook["cells"][24]["source"])
old_call = "build_conversation(\n                record, len(images), answer=answer, add_answer=True\n            )"
new_call = "build_conversation(\n                record, self.split, len(images), answer=answer, add_answer=True\n            )"
if old_call not in cell_25:
    raise RuntimeError("Cell 25 full conversation 교체 대상을 찾지 못했습니다.")
cell_25 = cell_25.replace(old_call, new_call, 1)
cell_25 = cell_25.replace(
    "build_conversation(record, len(images), add_answer=False)",
    "build_conversation(record, self.split, len(images), add_answer=False)",
    1,
)
set_cell(notebook, 25, cell_25)


set_cell(
    notebook,
    30,
    r"""
### 11. Validation RAG 진단

유형과 RAG 적용 여부별 exact match를 확인합니다. LA는 공식 평가 방식과 다를 수 있으므로 길이와 예시를 함께 확인하고, 최종 판단은 리더보드 점수를 기준으로 합니다.
""",
)


set_cell(
    notebook,
    31,
    r"""
def comparable_answer(text, form):
    if form == "MC":
        normalized, valid = normalize_mc(text)
        return normalized if valid else str(text).strip()
    return re.sub(r"\s+", " ", str(text)).strip()


validation_subset = records["validation"][:VALIDATION_MAX_SAMPLES]
validation_rows = []

for record in tqdm(validation_subset, desc="validation RAG 추론"):
    result = predict_one(record, "validation")
    form = record["metadata"]["question_form"]
    target = record["model_output"]["answer"]
    rag_item = rag_item_for(record, "validation")
    validation_rows.append(
        {
            "question_id": record["metadata"]["question_id"],
            "question_form": form,
            "rag_applied": rag_item["rag_applied"],
            "rag_doc_ids": "/".join(doc["doc_id"] for doc in rag_item["documents"]),
            "rag_titles": " / ".join(doc["title"] for doc in rag_item["documents"]),
            "top_bge_score": rag_item["top_bge_score"],
            "top_siglip_text_score": rag_item["top_siglip_text_score"],
            "top_image_score": rag_item["top_image_score"],
            "prediction": result["answer"],
            "target": target,
            "exact_match": comparable_answer(result["answer"], form)
            == comparable_answer(target, form),
            "prediction_length": len(result["answer"]),
            "retried": result["retried"],
            "raw": result["raw"],
        }
    )

validation_df = pd.DataFrame(validation_rows)
display(validation_df.groupby(["question_form", "rag_applied"])["exact_match"].agg(["count", "mean"]))
display(validation_df.head(10))

validation_path = RUN_OUTPUT_DIR / "validation_predictions.csv"
validation_df.to_csv(validation_path, index=False, encoding="utf-8-sig")
print("validation 결과 저장:", validation_path)
""",
)


set_cell(
    notebook,
    32,
    r"""
### 12. Test 800건 RAG 추론

20건마다 Google Drive에 예측 캐시를 저장합니다. 연결이 끊겨도 같은 설정으로 다시 실행하면 완료된 question_id를 건너뜁니다.

모델, Adapter, RAG DB 또는 프롬프트를 변경하면 RUN_NAME을 바꾸거나 RESET_PREDICTION_CACHE를 True로 한 번 실행하세요.
""",
)


set_cell(
    notebook,
    33,
    r"""
prediction_cache_path = RUN_OUTPUT_DIR / "test_prediction_cache.json"


if RESET_PREDICTION_CACHE and prediction_cache_path.exists():
    prediction_cache_path.unlink()
    print("기존 test 예측 캐시를 삭제했습니다.")

if prediction_cache_path.exists():
    prediction_cache = json.loads(prediction_cache_path.read_text(encoding="utf-8"))
else:
    prediction_cache = {}

test_ids = {row["metadata"]["question_id"] for row in records["test"]}
prediction_cache = {
    question_id: value
    for question_id, value in prediction_cache.items()
    if question_id in test_ids
}

print(f"재사용 가능한 기존 예측: {len(prediction_cache)} / {len(records['test'])}")

completed_since_save = 0
for record in tqdm(records["test"], desc="test TourAPI RAG 추론"):
    question_id = record["metadata"]["question_id"]
    if question_id in prediction_cache:
        continue

    result = predict_one(record, "test")
    rag_item = rag_item_for(record, "test")
    prediction_cache[question_id] = {
        "answer": result["answer"],
        "raw": result["raw"],
        "question_form": record["metadata"]["question_form"],
        "rag_applied": rag_item["rag_applied"],
        "rag_doc_ids": [doc["doc_id"] for doc in rag_item["documents"]],
        "rag_titles": [doc["title"] for doc in rag_item["documents"]],
        "retried": result["retried"],
        "valid": result["valid"],
    }
    completed_since_save += 1

    if completed_since_save >= SAVE_EVERY_N_PREDICTIONS:
        atomic_write_json(prediction_cache, prediction_cache_path)
        completed_since_save = 0

atomic_write_json(prediction_cache, prediction_cache_path)
print("test 예측 완료:", len(prediction_cache))
print("예측 캐시:", prediction_cache_path)
""",
)


set_cell(notebook, 34, "### 13. 제출 JSON·ZIP 생성 및 최종 검증")


set_cell(
    notebook,
    35,
    r"""
if len(prediction_cache) != len(records["test"]):
    missing_ids = [
        row["metadata"]["question_id"]
        for row in records["test"]
        if row["metadata"]["question_id"] not in prediction_cache
    ]
    raise RuntimeError(f"예측이 완료되지 않은 test 문항: {missing_ids[:10]}")

submission = copy.deepcopy(records["test"])
preview_rows = []

for record in submission:
    question_id = record["metadata"]["question_id"]
    form = record["metadata"]["question_form"]
    cached = prediction_cache[question_id]
    answer = str(cached["answer"]).strip()
    record["model_output"] = {"answer": answer}
    preview_rows.append(
        {
            "question_id": question_id,
            "question_form": form,
            "image_name": record["model_input"]["image_name"],
            "rag_applied": cached["rag_applied"],
            "rag_titles": " / ".join(cached["rag_titles"]),
            "answer": answer,
            "answer_length": len(answer),
        }
    )

assert len(submission) == 800
assert [row["metadata"]["question_id"] for row in submission] == [
    row["metadata"]["question_id"] for row in records["test"]
]
for original, completed in zip(records["test"], submission):
    assert original["metadata"] == completed["metadata"]
    assert original["model_input"] == completed["model_input"]
    answer = completed["model_output"]["answer"]
    form = completed["metadata"]["question_form"]
    assert isinstance(answer, str) and answer.strip()
    if form == "MC":
        assert re.fullmatch(r"[1-5](?:/[1-5])*", answer), (form, answer)
    if form == "LA":
        assert len(answer) <= 250, (form, len(answer))

submission_json_path = RUN_OUTPUT_DIR / "submission_kanana15v_3b_lora_tourapi_rag_typed.json"
submission_zip_path = RUN_OUTPUT_DIR / "submission_kanana15v_3b_lora_tourapi_rag_typed.zip"
submission_preview_path = RUN_OUTPUT_DIR / "submission_preview.csv"

atomic_write_json(submission, submission_json_path)
pd.DataFrame(preview_rows).to_csv(
    submission_preview_path, index=False, encoding="utf-8-sig"
)
with zipfile.ZipFile(submission_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    archive.write(submission_json_path, arcname=submission_json_path.name)

reloaded_submission = json.loads(submission_json_path.read_text(encoding="utf-8"))
assert len(reloaded_submission) == 800
assert all("model_output" in row for row in reloaded_submission)

print("제출 파일 생성 완료")
print("JSON:", submission_json_path)
print("ZIP :", submission_zip_path)
print("미리보기:", submission_preview_path)
display(pd.DataFrame(preview_rows).groupby(["question_form", "rag_applied"]).size())
display(pd.DataFrame(preview_rows).head(10))
""",
)


set_cell(notebook, 36, "### 14. RAG 실험 요약 저장")


set_cell(
    notebook,
    37,
    r"""
rag_experiment_summary = {
    "run_name": RUN_NAME,
    "model_id": MODEL_ID,
    "reference_baseline_score": REFERENCE_BASELINE_SCORE,
    "rag_db_saved_at": qdrant_state.get("saved_at"),
    "rag_db_document_count": qdrant_state.get("indexed_doc_id_count"),
    "rag_config": rag_config_for_cache,
    "retrieval_summary": rag_summary_rows,
    "train_method": TRAINING_METHOD,
    "shared_epochs": SHARED_EPOCHS,
    "typed_epochs": TYPED_EPOCHS,
    "submission_json": str(submission_json_path),
    "submission_zip": str(submission_zip_path),
    "validation_csv": str(validation_path),
    "rag_cache": str(RAG_CACHE_PATH),
}

rag_experiment_summary_path = RUN_OUTPUT_DIR / "rag_experiment_summary.json"
atomic_write_json(rag_experiment_summary, rag_experiment_summary_path)
print(json.dumps(rag_experiment_summary, ensure_ascii=False, indent=2))
print("RAG 실험 요약:", rag_experiment_summary_path)
""",
)


set_cell(
    notebook,
    38,
    r"""
### 실행 결과 확인 순서

1. Qdrant의 `points_count`가 2,793인지 확인
2. train·validation·test RAG 검색 캐시가 총 3,000건 생성됐는지 확인
3. 검색 모델 GPU 메모리 해제 후 Kanana가 정상 로드되는지 확인
4. 프롬프트 예시에서 강한 근거가 있는 문항에만 검색 자료가 추가됐는지 확인
5. 학습 전 loss와 LoRA gradient 스모크 테스트 통과 확인
6. Shared Adapter와 MC·SA·LA Adapter 저장 확인
7. validation에서 유형·RAG 적용 여부별 exact match 확인
8. `submission_kanana15v_3b_lora_tourapi_rag_typed.json`을 리더보드에 제출
9. 기존 Kanana 유형별 Adapter 점수 `44.8395439`와 비교

첫 비교에서는 현재 TourAPI DB만 사용합니다. 이후 DB가 추가되면 Qdrant를 갱신한 뒤 이 노트북을 다시 실행하면 DB의 `saved_at` 변경을 감지하여 RAG 검색 캐시를 자동으로 다시 만듭니다.
""",
)


for cell in notebook["cells"]:
    cell.setdefault("id", uuid.uuid4().hex[:8])
    cell.setdefault("metadata", {})
    if cell["cell_type"] == "markdown":
        cell.pop("execution_count", None)
        cell.pop("outputs", None)
    elif cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []


OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
    encoding="utf-8",
)
print(f"작성 완료: {OUTPUT_PATH}")
