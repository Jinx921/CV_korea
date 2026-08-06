import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / "notebooks" / "qwen3vl_8b_lora_paddle_crop.ipynb"
OUTPUT_PATH = ROOT / "notebooks" / "qwen3vl_8b_lora_paddle_crop_v2_selective.ipynb"


def source_lines(text):
    return text.strip("\n").splitlines(keepends=True)


def cell_text(cell):
    return "".join(cell["source"])


def replace_once(text, old, new):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"교체 대상이 {count}개입니다: {old[:80]!r}")
    return text.replace(old, new, 1)


notebook = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))

# 제목과 실험 설명
notebook["cells"][0]["source"] = source_lines(
    r"""
## Qwen3-VL 8B LoRA + PaddleOCR 선택적 crop v2

이 노트북은 OCR팀의 두 번째 리더보드 실험입니다.

1. Qwen3-VL 8B 일반 LoRA를 처음부터 2 epoch 학습
2. OCR 후보 중 MC·SA 문항에만 PaddleOCR crop 적용
3. 신뢰도 0.80 이상이며 한글이 2글자 이상인 영역 중 가장 좋은 1개만 확대
4. LA는 OCR crop 없이 원본 이미지만 사용
5. PaddleOCR가 인식한 문자열은 Qwen 프롬프트에 넣지 않음
6. 기존 v1 PaddleOCR 캐시는 복사하여 재사용
7. validation 200건 확인 후 test 800건 추론 및 제출 파일 생성

v1과 비교하면 학습 epoch, crop 개수·크기, crop 채택 기준, 적용 문항 유형이 달라집니다.
"""
)

# 설정 셀
config_index = next(
    index
    for index, cell in enumerate(notebook["cells"])
    if cell["cell_type"] == "code" and "RUN_NAME =" in cell_text(cell)
)
config = cell_text(notebook["cells"][config_index])
config = replace_once(
    config,
    'RUN_NAME = "qwen3vl_8b_lora_paddle_crop_v1"',
    'RUN_NAME = "qwen3vl_8b_lora_paddle_crop_v2_selective"',
)
config = replace_once(config, "VALIDATION_MAX_SAMPLES = 30", "VALIDATION_MAX_SAMPLES = 200")
config = replace_once(config, "NUM_EPOCHS = 1", "NUM_EPOCHS = 2")
config = replace_once(config, "MAX_OCR_CROPS = 2", "MAX_OCR_CROPS = 1")
config = replace_once(config, "OCR_CROP_MAX_SIDE = 448", "OCR_CROP_MAX_SIDE = 672")
config = replace_once(
    config,
    "OCR_MIN_SCORE = 0.30\nOCR_RAG_SCORE = 0.50",
    "OCR_MIN_SCORE = 0.30\n"
    "OCR_CROP_MIN_SCORE = 0.80\n"
    "OCR_CROP_MIN_HANGUL_CHARS = 2\n"
    "OCR_RAG_SCORE = 0.50",
)
config = replace_once(
    config,
    '# OCR crop 이미지 토큰이 추가되므로 입력 상한만 2048에서 3072로 확장합니다.',
    '# 고해상도 단일 crop 이미지 토큰을 위해 입력 상한을 3072로 유지합니다.',
)
config = replace_once(
    config,
    'OCR_APPLY_MODE = "routed"  # OCR 후보 질문에만 crop 적용',
    'OCR_APPLY_MODE = "selective_mc_sa"  # OCR 후보 MC·SA에만 고신뢰도 crop 적용',
)
notebook["cells"][config_index]["source"] = source_lines(config)

# OCR 캐시는 v1 결과를 새 v2 폴더로 복사해 사용합니다.
ocr_index = next(
    index
    for index, cell in enumerate(notebook["cells"])
    if cell["cell_type"] == "code" and "OCR_CACHE_PATH =" in cell_text(cell)
)
ocr_code = cell_text(notebook["cells"][ocr_index])
ocr_code = replace_once(
    ocr_code,
    'OCR_CACHE_PATH = RUN_OUTPUT_DIR / "paddleocr_cache.json"',
    'OCR_CACHE_PATH = RUN_OUTPUT_DIR / "paddleocr_cache.json"\n'
    'V1_OCR_CACHE_PATH = (\n'
    '    DRIVE_PROJECT_DIR / "outputs" / "qwen3vl_8b_lora_paddle_crop_v1"\n'
    '    / "paddleocr_cache.json"\n'
    ')\n\n'
    'if not OCR_CACHE_PATH.exists() and V1_OCR_CACHE_PATH.exists():\n'
    '    shutil.copy2(V1_OCR_CACHE_PATH, OCR_CACHE_PATH)\n'
    '    print("v1 PaddleOCR 캐시를 v2 실행 폴더로 복사했습니다:", OCR_CACHE_PATH)',
)

old_router = r'''def should_apply_ocr(record):
    if OCR_APPLY_MODE == "all":
        return True
    if OCR_APPLY_MODE != "routed":
        raise ValueError(OCR_APPLY_MODE)
    model_input = record["model_input"]
    parts = [model_input.get("question", "")]
    parts.extend(str(option) for option in (model_input.get("options") or []))
    return bool(OCR_CUE_RE.search(" ".join(parts)))'''
new_router = r'''def should_apply_ocr(record):
    if OCR_APPLY_MODE != "selective_mc_sa":
        raise ValueError(OCR_APPLY_MODE)
    if record["metadata"]["question_form"] not in {"MC", "SA"}:
        return False
    model_input = record["model_input"]
    parts = [model_input.get("question", "")]
    parts.extend(str(option) for option in (model_input.get("options") or []))
    return bool(OCR_CUE_RE.search(" ".join(parts)))'''
ocr_code = replace_once(ocr_code, old_router, new_router)

old_entries = '''    entries = ocr_cache["items"][key]["entries"]
    ranked = sorted(entries, key=lambda entry: crop_rank(entry, original.size), reverse=True)'''
new_entries = '''    entries = ocr_cache["items"][key]["entries"]
    eligible_entries = [
        entry for entry in entries
        if entry["score"] >= OCR_CROP_MIN_SCORE
        and len(re.findall(r"[가-힣]", entry["text"])) >= OCR_CROP_MIN_HANGUL_CHARS
    ]
    ranked = sorted(
        eligible_entries,
        key=lambda entry: crop_rank(entry, original.size),
        reverse=True,
    )'''
ocr_code = replace_once(ocr_code, old_entries, new_entries)
notebook["cells"][ocr_index]["source"] = source_lines(ocr_code)

# 학습 전 집계 문구를 v2 조건에 맞게 명확히 합니다.
train_ocr_index = next(
    index
    for index, cell in enumerate(notebook["cells"])
    if cell["cell_type"] == "code" and "routed_train_count" in cell_text(cell)
)
train_ocr_code = cell_text(notebook["cells"][train_ocr_index])
train_ocr_code = replace_once(
    train_ocr_code,
    '''crop_train_count = sum(
    bool(ocr_cache["items"][image_cache_key(record, "train")]["entries"])
    for record in train_rows_for_ocr if should_apply_ocr(record)
)''',
    '''crop_train_count = sum(
    bool(ocr_cache["items"][image_cache_key(record, "train")]["entries"])
    for record in train_rows_for_ocr if should_apply_ocr(record)
)
selective_crop_train_count = sum(
    any(
        entry["score"] >= OCR_CROP_MIN_SCORE
        and len(re.findall(r"[가-힣]", entry["text"])) >= OCR_CROP_MIN_HANGUL_CHARS
        for entry in ocr_cache["items"][image_cache_key(record, "train")]["entries"]
    )
    for record in train_rows_for_ocr if should_apply_ocr(record)
)''',
)
train_ocr_code = replace_once(
    train_ocr_code,
    'print("OCR 후보 train 문항:", routed_train_count)\nprint("실제 crop 생성 가능 문항:", crop_train_count)',
    'print("선택적 OCR 대상 MC·SA train 문항:", routed_train_count)\n'
    'print("OCR 인식 결과가 존재하는 문항:", crop_train_count)\n'
    'print("실제 고신뢰도 한글 crop 적용 문항:", selective_crop_train_count)',
)
notebook["cells"][train_ocr_index]["source"] = source_lines(train_ocr_code)

# 제출 파일명을 v2로 분리합니다.
submission_index = next(
    index
    for index, cell in enumerate(notebook["cells"])
    if cell["cell_type"] == "code" and "submission_json_path" in cell_text(cell)
)
submission_code = cell_text(notebook["cells"][submission_index])
submission_code = submission_code.replace(
    "submission_qwen3vl_8b_lora_paddle_crop.json",
    "submission_qwen3vl_8b_lora_paddle_crop_v2_selective.json",
)
submission_code = submission_code.replace(
    "submission_qwen3vl_8b_lora_paddle_crop.zip",
    "submission_qwen3vl_8b_lora_paddle_crop_v2_selective.zip",
)
notebook["cells"][submission_index]["source"] = source_lines(submission_code)

# 마지막 실행 안내도 v2 파일명과 설정으로 교체합니다.
for cell in notebook["cells"]:
    if cell["cell_type"] != "markdown":
        continue
    text = cell_text(cell)
    text = text.replace(
        "submission_qwen3vl_8b_lora_paddle_crop.json",
        "submission_qwen3vl_8b_lora_paddle_crop_v2_selective.json",
    )
    text = text.replace(
        "대표 3개 이미지에서 한글 OCR이 제대로 추출되는지 확인",
        "대표 3개 이미지에서 한글 OCR이 제대로 추출되는지 확인",
    )
    cell["source"] = source_lines(text)

notebook["metadata"]["colab"]["name"] = OUTPUT_PATH.name
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

OUTPUT_PATH.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
    encoding="utf-8",
)
print(OUTPUT_PATH)
