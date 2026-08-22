from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "notebooks" / "kanana15v_3b_lora_verified_mapping_rag_typed.ipynb"
OCR_REFERENCE_PATH = ROOT / "notebooks" / "kanana15v_3b_lora_paddle_crop_typed.ipynb"
OUTPUT_PATH = ROOT / "notebooks" / "kanana15v_3b_lora_verified_rag_sa_ocr_exact.ipynb"


nb = nbf.read(BASE_PATH, as_version=4)
ocr_reference = nbf.read(OCR_REFERENCE_PATH, as_version=4)
nb.metadata["colab"]["name"] = OUTPUT_PATH.name

nb.cells[0].source = """
## Kanana-1.5-V 3B 검증 매핑 RAG + SA Exact Match 개선

이 노트북은 RAG-only 실험을 기준으로 SA exact match를 개선합니다.

- 검증된 이미지↔TourAPI 매핑 RAG 유지
- 글자 판독 단서가 있는 **SA 문항에만** PP-OCRv5 crop 추가
- OCR 인식 문자열은 프롬프트에 주입하지 않음
- SA 유형 Adapter만 2 epoch, MC·LA는 기존 1 epoch 유지
- SA 생성 길이를 32 token으로 축소
- 요구 음절·어절 길이를 어긴 경우 한 번만 형식 강화 재생성
- 재생성 시 단일 `N음절`/`N어절` 조건을 logits 단계에서 강제

Validation 분석에서는 SA 52건 중 20건이 exact match였고, 32개 오답 중 19건이
요구 길이를 지키지 못했습니다. 단순 형식 정리로 복구 가능한 건은 2건뿐이므로
학습·시각 판독·생성 제약을 함께 바꿉니다.

Colab GPU(L4 이상 권장)에서 위에서부터 실행하세요. 첫 설치 셀 실행 후 런타임이
자동 재시작되면 다시 `모두 실행`을 눌러야 합니다.
""".strip()

# PaddleOCR와 Kanana가 함께 동작한 기존 환경 설치 셀을 재사용하되 실행 마커를 분리한다.
install_source = ocr_reference.cells[2].source.replace(
    "/content/.kanana15v_paddle_crop_env_v1",
    "/content/.kanana15v_verified_rag_sa_ocr_env_v1",
)
nb.cells[2].source = install_source

config = nb.cells[5].source
config = config.replace(
    'from PIL import Image, ImageOps',
    'from PIL import Image, ImageDraw, ImageOps',
)
config = config.replace(
    'RUN_NAME = "kanana15v_3b_lora_verified_mapping_rag_typed"',
    'RUN_NAME = "kanana15v_3b_lora_verified_rag_sa_ocr_exact_v2"',
)
config = config.replace(
    'TRAINING_METHOD = "shared_then_typed_lora_with_manual_verified_mapping_rag"',
    'TRAINING_METHOD = "verified_mapping_rag_sa_typed_lora_selective_ocr_and_retry_length_logits"',
)
config = config.replace(
    'TYPED_EPOCHS = 1',
    'TYPED_EPOCHS_BY_FORM = {"MC": 1, "SA": 2, "LA": 1}',
)
config = config.replace(
    'OCR_MODE = "disabled_for_rag_ablation"',
    '''# SA exact-match용 선택적 OCR crop
MAX_OCR_CROPS = 1
OCR_CROP_MAX_SIDE = 672
OCR_VERSION = "PP-OCRv5"
OCR_DETECTION_MODEL = "PP-OCRv5_server_det"
OCR_RECOGNITION_MODEL = "korean_PP-OCRv5_mobile_rec"
OCR_MAX_IMAGE_SIDE = 4000
OCR_MIN_SCORE = 0.30
OCR_CROP_MIN_SCORE = 0.80
OCR_CROP_MIN_HANGUL_CHARS = 2
OCR_APPLY_MODE = "sa_question_cue_only"
RESET_OCR_CACHE = False
OCR_CACHE_SAVE_EVERY = 20
OCR_MODE = "selective_sa_crop_without_text_prompt"

# 첫 생성 답변은 자유롭게 두고, 길이 위반으로 재시도할 때만 강제합니다.
# 검증 데이터의 표기 예외와 의미 훼손 위험을 줄이기 위한 설정입니다.
SA_LENGTH_CONSTRAINT_MODE = "retry_only"''',
)
config = config.replace(
    'print("OCR:", OCR_MODE)',
    'print("OCR:", OCR_MODE, OCR_DETECTION_MODEL, "+", OCR_RECOGNITION_MODEL)',
)
config = config.replace(
    'print("학습: Shared", SHARED_EPOCHS, "epoch + 유형별", TYPED_EPOCHS, "epoch")',
    'print("학습: Shared", SHARED_EPOCHS, "epoch + 유형별", TYPED_EPOCHS_BY_FORM)',
)
nb.cells[5].source = config

# 검증된 RAG 프롬프트는 유지하고 길이 재시도 지시만 강화한다.
nb.cells[16].source = nb.cells[16].source.replace(
    '이전 출력 형식이 잘못되었습니다. 이번에는 반드시 지정 형식만 출력하세요.',
    '이전 출력이 요구한 음절·글자·어절 수 또는 형식을 지키지 못했습니다. '
    '정답 내용을 다시 판단한 뒤 출력 전에 길이를 세고, 반드시 정답만 출력하세요.',
)

# PP-OCRv5 초기화·캐시 코드는 검증된 기존 노트북에서 가져와 SA-only로 제한한다.
ocr_init = ocr_reference.cells[12].source
ocr_init = ocr_init.replace(
    'if OCR_APPLY_MODE != "selective_mc_sa":',
    'if OCR_APPLY_MODE != "sa_question_cue_only":',
)
ocr_init = ocr_init.replace(
    'if record["metadata"]["question_form"] not in {"MC", "SA"}:',
    'if record["metadata"]["question_form"] != "SA":',
)
ocr_preprocess = ocr_reference.cells[14].source.replace(
    '선택적 OCR 대상 MC·SA train 문항:',
    '선택적 OCR 대상 SA train 문항:',
)

# 기존 RAG 구성 다음, 프롬프트 구성 전에 OCR 셀을 삽입한다.
insert_at = 15
nb.cells.insert(insert_at, nbf.v4.new_markdown_cell(
    "### 6. PP-OCRv5 한국어 초기화 및 대표 이미지 검수"
))
nb.cells.insert(insert_at + 1, nbf.v4.new_code_cell(ocr_init.strip()))
nb.cells.insert(insert_at + 2, nbf.v4.new_markdown_cell(
    "### 7. 학습용 SA OCR 후보 이미지 사전 처리"
))
nb.cells.insert(insert_at + 3, nbf.v4.new_code_cell(ocr_preprocess.strip()))

# 삽입 후 기존 셀 인덱스는 4씩 증가한다.
image_cell_index = 24
image_source = nb.cells[image_cell_index].source
length_processor_source = r'''
from transformers import LogitsProcessor, LogitsProcessorList


_SA_COUNT_RE = re.compile(r"(\d+)\s*(음절|어절)")
_SA_TOKEN_STATS_CACHE = {}


def parse_sa_length_constraint(question_text):
    """안전하게 처리 가능한 단일 N음절/N어절 조건만 반환합니다."""
    question_text = str(question_text).split("\n\n[참고 정보")[0]
    matches = _SA_COUNT_RE.findall(question_text)
    if len(matches) != 1:
        return None
    count_text, unit_kr = matches[0]
    target = int(count_text)
    if target <= 0:
        return None
    unit = "syllable" if unit_kr == "음절" else "eojeol"
    return unit, target


def _is_hangul_syllable(character):
    return "가" <= character <= "힣"


def build_sa_token_stats(tokenizer):
    """토큰별 한글 음절 수와 공백/어절 경계 정보를 한 번만 계산합니다."""
    cache_key = id(tokenizer)
    if cache_key in _SA_TOKEN_STATS_CACHE:
        return _SA_TOKEN_STATS_CACHE[cache_key]

    vocab_size = len(tokenizer)
    syllable_count = torch.zeros(vocab_size, dtype=torch.long)
    chunk_count = torch.zeros(vocab_size, dtype=torch.long)
    starts_with_space = torch.zeros(vocab_size, dtype=torch.bool)
    ends_with_space = torch.zeros(vocab_size, dtype=torch.bool)
    whitespace_only = torch.zeros(vocab_size, dtype=torch.bool)

    for start in tqdm(range(0, vocab_size, 4096), desc="SA 길이 토큰 통계"):
        token_ids = list(range(start, min(start + 4096, vocab_size)))
        decoded = tokenizer.batch_decode(
            [[token_id] for token_id in token_ids],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for offset, text in enumerate(decoded):
            if not text:
                continue
            token_id = start + offset
            syllable_count[token_id] = sum(
                _is_hangul_syllable(character) for character in text
            )
            chunk_count[token_id] = len(text.split())
            starts_with_space[token_id] = text[0].isspace()
            ends_with_space[token_id] = text[-1].isspace()
            whitespace_only[token_id] = not text.strip()

    stats = {
        "syllable_count": syllable_count,
        "chunk_count": chunk_count,
        "starts_with_space": starts_with_space,
        "ends_with_space": ends_with_space,
        "whitespace_only": whitespace_only,
    }
    _SA_TOKEN_STATS_CACHE[cache_key] = stats
    return stats


class KoreanLengthLogitsProcessor(LogitsProcessor):
    """단일 SA 답변이 지정 음절/어절 수를 넘거나 일찍 끝나지 않게 제한합니다.

    팀원 구현의 핵심 방식을 사용하되, Kanana generate가 prompt input_ids를
    포함하는 경우에도 안전하도록 첫 호출의 prefix 길이를 자동으로 제외합니다.
    """

    def __init__(self, tokenizer, specs, eos_token_id):
        self.specs = specs
        self.eos_token_id = int(eos_token_id)
        stats = build_sa_token_stats(tokenizer)
        self.syllable_count = stats["syllable_count"]
        self.chunk_count = stats["chunk_count"]
        self.starts_with_space = stats["starts_with_space"]
        self.ends_with_space = stats["ends_with_space"]
        self.whitespace_only = stats["whitespace_only"]
        self._prefix_length = None
        self._processed_lengths = [0] * len(specs)
        self._word_counts = [0] * len(specs)
        self._at_boundaries = [True] * len(specs)

    def _move_stats_once(self, device):
        if self.syllable_count.device == device:
            return
        self.syllable_count = self.syllable_count.to(device)
        self.chunk_count = self.chunk_count.to(device)
        self.starts_with_space = self.starts_with_space.to(device)
        self.ends_with_space = self.ends_with_space.to(device)
        self.whitespace_only = self.whitespace_only.to(device)

    def __call__(self, input_ids, scores):
        self._move_stats_once(scores.device)
        current_length = input_ids.shape[1]
        if self._prefix_length is None:
            # 첫 호출은 첫 생성 토큰을 고르기 전이므로 현재 길이가 prompt prefix입니다.
            self._prefix_length = current_length

        for row, spec in enumerate(self.specs):
            if spec is None:
                continue
            unit, target = spec
            generated_ids = input_ids[row, self._prefix_length :]

            if unit == "syllable":
                current = int(self.syllable_count[generated_ids].sum().item())
                remaining = target - current
                if remaining <= 0:
                    scores[row, :] = float("-inf")
                    scores[row, self.eos_token_id] = 0.0
                else:
                    scores[row, self.eos_token_id] = float("-inf")
                    scores[row, self.syllable_count > remaining] = float("-inf")
                continue

            # 어절 상태는 새로 생성된 토큰만 증분 처리합니다.
            processed = self._processed_lengths[row]
            new_ids = generated_ids[processed:]
            for token_tensor in new_ids:
                token_id = int(token_tensor.item())
                chunks = int(self.chunk_count[token_id].item())
                if chunks == 0:
                    if bool(self.whitespace_only[token_id].item()):
                        self._at_boundaries[row] = True
                    continue

                prior_mid_word = (
                    not self._at_boundaries[row] and self._word_counts[row] > 0
                )
                begins_with_space = bool(self.starts_with_space[token_id].item())
                merges = prior_mid_word and not begins_with_space
                self._word_counts[row] += chunks - 1 if merges else chunks
                self._at_boundaries[row] = bool(self.ends_with_space[token_id].item())
            self._processed_lengths[row] = len(generated_ids)

            word_count = self._word_counts[row]
            at_boundary = self._at_boundaries[row]
            if word_count < target:
                scores[row, self.eos_token_id] = float("-inf")
            if word_count >= target:
                if not at_boundary and word_count > 0:
                    effective_chunks = torch.where(
                        self.starts_with_space,
                        self.chunk_count,
                        self.chunk_count - 1,
                    )
                else:
                    effective_chunks = self.chunk_count
                scores[row, effective_chunks > 0] = float("-inf")
        return scores


def sa_length_logits_processors(record, retry):
    if SA_LENGTH_CONSTRAINT_MODE != "retry_only" or not retry:
        return None
    if record["metadata"]["question_form"] != "SA":
        return None
    spec = parse_sa_length_constraint(record["model_input"].get("question", ""))
    if spec is None:
        return None
    return LogitsProcessorList([
        KoreanLengthLogitsProcessor(
            processor.tokenizer,
            specs=[spec],
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    ])
'''.strip()
image_source = length_processor_source + "\n\n\n" + image_source
image_source = image_source.replace(
    '''def build_input_images(record, split, max_image_side):
    # 첫 실험에서는 대회 원본 이미지만 Kanana에 넣습니다.
    # TourAPI 이미지 벡터는 관련 설명을 검색하는 데 사용됩니다.
    return [load_rgb_image(image_path_for(record, split), max_image_side)]''',
    '''def build_input_images(record, split, max_image_side):
    original = load_rgb_image(image_path_for(record, split), max_image_side)
    images = [original]
    if should_apply_ocr(record):
        crops = make_ocr_crops(record, split, max_crops=MAX_OCR_CROPS)
        images.extend(crop["image"] for crop in crops)
    return images''',
)
image_source = image_source.replace(
    '''def normalize_long_answer(text):''',
    '''def sa_length_requirements(record):
    question = str(record["model_input"].get("question", ""))
    return [
        (int(length), unit)
        for length, unit in re.findall(r"(\\d+)\\s*(음절|글자|어절)", question)
    ]


def measure_sa_segment(text, unit):
    cleaned = normalize_short_answer(text)
    if unit == "어절":
        return len([token for token in cleaned.split() if token])
    return len(re.findall(r"[0-9A-Za-z가-힣]", cleaned))


def sa_length_matches(answer, record):
    requirements = sa_length_requirements(record)
    if not requirements:
        return None
    segments = [segment.strip() for segment in str(answer).split("/")]
    if len(requirements) > 1:
        if len(segments) != len(requirements):
            return False
        return all(
            measure_sa_segment(segment, unit) == expected
            for segment, (expected, unit) in zip(segments, requirements)
        )
    expected, unit = requirements[0]
    return measure_sa_segment(answer, unit) == expected


def normalize_long_answer(text):''',
)
image_source = image_source.replace(
    'return {"MC": 16, "SA": 64, "LA": 256}[form]',
    'return {"MC": 16, "SA": 32, "LA": 256}[form]',
)
image_source = image_source.replace(
    '''    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens_for(form),
        temperature=0,
        top_p=1.0,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )''',
    '''    generation_kwargs = {
        "max_new_tokens": max_new_tokens_for(form),
        "temperature": 0,
        "top_p": 1.0,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "pad_token_id": processor.tokenizer.pad_token_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
    }
    length_processors = sa_length_logits_processors(record, retry=retry)
    if length_processors is not None:
        generation_kwargs["logits_processor"] = length_processors
    generated = model.generate(**inputs, **generation_kwargs)''',
)
if 'generation_kwargs["logits_processor"]' not in image_source:
    raise RuntimeError("SA 길이 LogitsProcessor 연결 대상 코드를 찾지 못했습니다.")
old_predict = '''def predict_one(record, split):
    form = record["metadata"]["question_form"]
    raw_text = generate_raw(record, split, retry=False)
    answer, valid = normalize_prediction(raw_text, form)
    retried = False

    if not valid:
        retried = True
        retry_raw = generate_raw(record, split, retry=True)
        retry_answer, retry_valid = normalize_prediction(retry_raw, form)
        raw_text = raw_text + "\\n[RETRY]\\n" + retry_raw
        if retry_valid:
            answer, valid = retry_answer, True

    if not valid:
        answer = "1" if form == "MC" else "확인 불가"

    return {"answer": answer, "raw": raw_text, "retried": retried, "valid": valid}'''
new_predict = '''def predict_one(record, split):
    form = record["metadata"]["question_form"]
    raw_text = generate_raw(record, split, retry=False)
    answer, valid = normalize_prediction(raw_text, form)
    retried = False
    primary_length_match = sa_length_matches(answer, record) if form == "SA" else None
    hard_length_retry_applied = False

    needs_retry = not valid or (form == "SA" and primary_length_match is False)
    if needs_retry:
        retried = True
        hard_length_retry_applied = (
            form == "SA"
            and parse_sa_length_constraint(record["model_input"].get("question", ""))
            is not None
        )
        retry_raw = generate_raw(record, split, retry=True)
        retry_answer, retry_valid = normalize_prediction(retry_raw, form)
        retry_length_match = (
            sa_length_matches(retry_answer, record) if form == "SA" else None
        )
        raw_text = raw_text + "\\n[RETRY]\\n" + retry_raw

        if form == "SA":
            # 길이 조건은 soft gate입니다. 데이터 자체의 예외를 고려해 원답을 강제로 자르지 않습니다.
            if retry_valid and retry_length_match is True:
                answer, valid = retry_answer, True
            elif not valid and retry_valid:
                answer, valid = retry_answer, True
        elif retry_valid:
            answer, valid = retry_answer, True

    if not valid:
        answer = "1" if form == "MC" else "확인 불가"

    return {
        "answer": answer,
        "raw": raw_text,
        "retried": retried,
        "valid": valid,
        "sa_length_match": sa_length_matches(answer, record) if form == "SA" else None,
        "hard_length_retry_applied": hard_length_retry_applied,
    }'''
if old_predict not in image_source:
    raise RuntimeError("predict_one 교체 대상 코드를 찾지 못했습니다.")
image_source = image_source.replace(old_predict, new_predict)
nb.cells[image_cell_index].source = image_source

# 학습 설정: SA Adapter만 2 epoch.
training_cell_index = 32
training_source = nb.cells[training_cell_index].source
training_source = training_source.replace(
    '''            TYPED_EPOCHS,
        )''',
    '''            TYPED_EPOCHS_BY_FORM[form],
        )''',
)
if "TYPED_EPOCHS," in training_source:
    raise RuntimeError("유형별 epoch 교체가 완료되지 않았습니다.")
nb.cells[training_cell_index].source = training_source

# Validation: RAG·OCR·SA 길이 조건을 함께 기록한다.
validation_index = 34
nb.cells[validation_index].source = r'''
def comparable_answer(text, form):
    if form == "MC":
        normalized, valid = normalize_mc(text)
        return normalized if valid else str(text).strip()
    return re.sub(r"\s+", " ", str(text)).strip()


validation_subset = records["validation"][:VALIDATION_MAX_SAMPLES]
ensure_ocr_records("validation", validation_subset, only_routed=True)
validation_rows = []

for record in tqdm(validation_subset, desc="validation RAG+SA OCR 추론"):
    result = predict_one(record, "validation")
    form = record["metadata"]["question_form"]
    target = record["model_output"]["answer"]
    rag_item = rag_item_for(record, "validation")
    ocr_applied = should_apply_ocr(record)
    validation_rows.append({
        "question_id": record["metadata"]["question_id"],
        "question_form": form,
        "rag_applied": rag_item["rag_applied"],
        "rag_doc_ids": "/".join(doc["doc_id"] for doc in rag_item["documents"]),
        "rag_titles": " / ".join(doc["title"] for doc in rag_item["documents"]),
        "ocr_crop_applied": ocr_applied,
        "ocr_crop_count": len(make_ocr_crops(record, "validation")) if ocr_applied else 0,
        "sa_length_match": result["sa_length_match"],
        "hard_length_retry_applied": result["hard_length_retry_applied"],
        "prediction": result["answer"],
        "target": target,
        "exact_match": comparable_answer(result["answer"], form)
        == comparable_answer(target, form),
        "prediction_length": len(result["answer"]),
        "retried": result["retried"],
        "raw": result["raw"],
    })

validation_df = pd.DataFrame(validation_rows)
display(validation_df.groupby(["question_form", "ocr_crop_applied"])["exact_match"].agg(["count", "mean"]))
display(validation_df[validation_df["question_form"] == "SA"])

validation_path = RUN_OUTPUT_DIR / "validation_predictions.csv"
validation_df.to_csv(validation_path, index=False, encoding="utf-8-sig")
print("validation 결과 저장:", validation_path)
'''.strip()

# Test 추론 전에 선택적 SA OCR 캐시를 만들고 기록한다.
test_index = 36
test_source = nb.cells[test_index].source
test_source = 'ensure_ocr_records("test", records["test"], only_routed=True)\n\n' + test_source
test_source = test_source.replace(
    'for record in tqdm(records["test"], desc="test 검증 매핑 RAG 추론"):',
    'for record in tqdm(records["test"], desc="test RAG+SA OCR 추론"):',
)
test_source = test_source.replace(
    '''        "rag_titles": [doc["title"] for doc in rag_item["documents"]],
        "retried": result["retried"],''',
    '''        "rag_titles": [doc["title"] for doc in rag_item["documents"]],
        "ocr_crop_applied": should_apply_ocr(record),
        "sa_length_match": result["sa_length_match"],
        "hard_length_retry_applied": result["hard_length_retry_applied"],
        "retried": result["retried"],''',
)
nb.cells[test_index].source = test_source

# 제출 파일명과 미리보기 필드를 새 실험명으로 변경한다.
submission_index = 38
submission_source = nb.cells[submission_index].source
submission_source = submission_source.replace(
    '"rag_titles": " / ".join(cached["rag_titles"]),',
    '"rag_titles": " / ".join(cached["rag_titles"]),\n            "ocr_crop_applied": cached["ocr_crop_applied"],',
)
submission_source = submission_source.replace(
    "submission_kanana15v_3b_lora_verified_mapping_rag_typed.json",
    "submission_kanana15v_3b_lora_verified_rag_sa_ocr_exact_v2.json",
)
submission_source = submission_source.replace(
    "submission_kanana15v_3b_lora_verified_mapping_rag_typed.zip",
    "submission_kanana15v_3b_lora_verified_rag_sa_ocr_exact_v2.zip",
)
nb.cells[submission_index].source = submission_source

summary_index = 40
summary_source = nb.cells[summary_index].source
summary_source = summary_source.replace(
    '"typed_epochs": TYPED_EPOCHS,',
    '"typed_epochs_by_form": TYPED_EPOCHS_BY_FORM,',
)
summary_source = summary_source.replace(
    '"ocr_mode": OCR_MODE,',
    '''"ocr_mode": OCR_MODE,
    "ocr_detection_model": OCR_DETECTION_MODEL,
    "ocr_recognition_model": OCR_RECOGNITION_MODEL,
    "ocr_cache": str(OCR_CACHE_PATH),
    "sa_length_constraint_mode": SA_LENGTH_CONSTRAINT_MODE,''',
)
nb.cells[summary_index].source = summary_source

nb.cells[41].source = """
### 실행 결과 확인 순서

1. 최종 매핑 DB가 2,000건, matched 12건인지 확인
2. PP-OCRv5 대표 이미지에서 한글이 정상적으로 인식되는지 확인
3. 실제 SA OCR crop 대상 수와 생성된 crop 수 확인
4. Shared 1 epoch, MC 1·SA 2·LA 1 epoch 학습 완료 확인
5. Validation SA exact match와 길이 조건 준수율을 직전 38.46%와 비교
6. `submission_kanana15v_3b_lora_verified_rag_sa_ocr_exact_v2.json` 제출
7. 리더보드 exact_match 28.6408 및 전체 점수 43.7227과 비교

이 실험은 OCR 문자열을 프롬프트에 넣지 않습니다. PaddleOCR은 글자 영역을 찾고
고신뢰도 crop을 Kanana의 두 번째 이미지로 제공하는 역할만 합니다.
단일 음절·어절 조건은 첫 답변이 조건을 어긴 경우의 재시도에만 강제됩니다.
""".strip()

for cell in nb.cells:
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUTPUT_PATH)
print(OUTPUT_PATH)
