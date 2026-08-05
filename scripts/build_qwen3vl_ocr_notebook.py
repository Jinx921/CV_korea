import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "notebooks" / "qwen3vl_8b_lora_ppocrv5.ipynb"


def source_lines(text):
    return text.strip("\n").splitlines(keepends=True)


def markdown(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines(text),
    }


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines(text),
    }


cells = [
    markdown(
        r"""
## Qwen3-VL 8B LoRA + PP-OCRv5 Korean 실험

이 노트북은 기존 **Qwen3-VL 8B 일반 LoRA adapter**를 그대로 불러와 OCR 추가 효과만 비교합니다.

1. Google Drive의 `CV_korea/data` ZIP 4개를 Colab 로컬 디스크에 해제
2. 원본 해상도 이미지에 PP-OCRv5 Korean 적용 및 Drive 캐시 저장
3. 전체 이미지와 확대 글자 영역, OCR 텍스트를 Qwen3-VL에 함께 입력
4. validation에서 `baseline / OCR 전체 적용 / OCR 후보 문항만 적용` 비교
5. 선택한 방식으로 test 800건을 추론하고 제출 JSON·ZIP 생성

기존 LoRA 학습 노트북과 adapter는 수정하지 않습니다. 이번 실험은 **재학습 없이 OCR만 추가하는 1차 ablation**입니다.

대회 규정상 외부 모델 API는 제출에 사용할 수 없으므로 PP-OCRv5를 Colab 안에서 로컬 실행합니다.
"""
    ),
    markdown(
        r"""
### 0. Colab 런타임 준비

Colab 메뉴에서 **런타임 → 런타임 유형 변경 → GPU**를 선택하세요. Qwen3-VL 8B 추론에는 L4 24GB 또는 A100을 권장합니다.

첫 설치 후 Pillow를 깨끗하게 다시 불러오기 위해 런타임이 한 번 자동 재시작됩니다. 재연결되면 첫 셀부터 다시 실행하세요.
"""
    ),
    code(
        r"""
# PaddleOCR는 CPU에서 실행하고 Qwen3-VL만 GPU를 사용해 VRAM 충돌을 피합니다.
%pip uninstall -q -y gradio gradio_client torchao paddlepaddle-gpu

# PaddleOCR 공식 CPU 설치 조합입니다.
%pip install -q "paddlepaddle==3.2.0" -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
%pip install -q "paddleocr==3.5.0"

# Colab에는 LangChain 일부 패키지만 설치된 경우가 있어 PaddleX의 선택 기능
# import가 실패할 수 있습니다. OCR에서 직접 쓰지는 않지만 import 정합성을 맞춥니다.
%pip install -q "langchain-text-splitters"

# Qwen3-VL 실행 버전을 마지막에 고정해 PaddleOCR 설치 과정의 의존성 변경을 방지합니다.
%pip install -q -U \
  "transformers>=4.57.0,<5" \
  "huggingface-hub==0.36.2" \
  "accelerate>=1.2" \
  "peft>=0.14" \
  "safetensors" \
  "pandas>=2.2" \
  "tqdm>=4.66"

%pip install -q --no-cache-dir --force-reinstall "Pillow==12.3.0"

import os
from pathlib import Path

restart_marker = Path("/content/.qwen3vl_ppocrv5_pillow_ready")
if not restart_marker.exists():
    restart_marker.write_text("ready", encoding="utf-8")
    print("설치 완료. 런타임을 재시작합니다. 재연결 후 첫 셀부터 다시 실행하세요.")
    os.kill(os.getpid(), 9)

import PIL
import paddle
try:
    import paddleocr
except RuntimeError as error:
    if "PDX has already been initialized" in str(error):
        print("이전 PaddleOCR import가 반쯤 초기화되어 런타임을 한 번 재시작합니다.")
        os.kill(os.getpid(), 9)
    raise
from PIL import Image

assert PIL.__version__ == "12.3.0", PIL.__version__
print("Pillow:", PIL.__version__)
print("PaddlePaddle:", paddle.__version__)
print("PaddleOCR:", paddleocr.__version__)
"""
    ),
    markdown("### 1. Google Drive 연결 및 실행 설정"),
    code(
        r"""
from google.colab import drive

drive.mount("/content/drive")
"""
    ),
    code(
        r"""
import copy
import gc
import json
import math
import random
import re
import unicodedata
import zipfile
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from tqdm.auto import tqdm


# 경로 설정
DRIVE_PROJECT_DIR = Path("/content/drive/MyDrive/CV_korea")
DRIVE_ZIP_DIR = DRIVE_PROJECT_DIR / "data"

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
ADAPTER_SOURCE_RUN_NAME = "qwen3vl_8b_lora_v1"
ADAPTER_DIR = DRIVE_PROJECT_DIR / "outputs" / ADAPTER_SOURCE_RUN_NAME / "adapter"
BASELINE_TEST_CACHE_PATH = (
    DRIVE_PROJECT_DIR / "outputs" / ADAPTER_SOURCE_RUN_NAME / "test_prediction_cache.json"
)

RUN_NAME = "qwen3vl_8b_lora_ppocrv5_v1"
RUN_OUTPUT_DIR = DRIVE_PROJECT_DIR / "outputs" / RUN_NAME
COLAB_WORK_DIR = Path("/content/kculture_qwen3vl_ppocrv5")
EXTRACT_DIR = COLAB_WORK_DIR / "extracted"

# 처음부터 전체 validation 200건을 비교합니다. 빠른 점검만 할 때 30으로 낮추세요.
VALIDATION_MAX_SAMPLES = 200

# 이미지 및 OCR 설정
INFERENCE_MAX_IMAGE_SIDE = 1344
OCR_MAX_IMAGE_SIDE = 4000
OCR_SCORE_THRESHOLD = 0.55
OCR_ROUTING_SCORE_THRESHOLD = 0.80
MAX_OCR_LINES = 30
MAX_OCR_TEXT_CHARS = 1200
MAX_OCR_CROPS = 3
OCR_CROP_MAX_SIDE = 896
OCR_CACHE_SAVE_EVERY = 20
OCR_SMOKE_COUNT = 5

# test 제출 모드: "baseline", "ocr_all", "ocr_routed" 중 하나
# validation 표를 확인한 뒤 필요하면 변경하세요.
FINAL_INFERENCE_MODE = "ocr_routed"
REUSE_BASELINE_TEST_CACHE = True
RESET_VALIDATION_PREDICTION_CACHE = False
RESET_TEST_PREDICTION_CACHE = False
SAVE_EVERY_N_PREDICTIONS = 20
SEED = 42

for path in [COLAB_WORK_DIR, EXTRACT_DIR, RUN_OUTPUT_DIR]:
    path.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("Adapter:", ADAPTER_DIR)
print("결과 저장:", RUN_OUTPUT_DIR)
print("최종 추론 모드:", FINAL_INFERENCE_MODE)
"""
    ),
    code(
        r"""
if not torch.cuda.is_available():
    raise RuntimeError("GPU 런타임이 아닙니다. Colab 런타임 유형을 GPU로 변경하세요.")

gpu_name = torch.cuda.get_device_name(0)
gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
bf16_supported = torch.cuda.is_bf16_supported()

print(f"GPU: {gpu_name}")
print(f"VRAM: {gpu_mem_gb:.1f} GiB")
print(f"BF16 지원: {bf16_supported}")

if gpu_mem_gb < 20:
    raise RuntimeError(
        "Qwen3-VL 8B BF16/FP16 추론에는 약 20GB 이상의 VRAM을 권장합니다. "
        "L4 또는 A100 런타임으로 변경하세요."
    )

if not (ADAPTER_DIR / "adapter_config.json").exists():
    raise FileNotFoundError(
        f"저장된 Qwen LoRA adapter를 찾지 못했습니다: {ADAPTER_DIR}\n"
        "기존 Qwen LoRA 노트북의 RUN_NAME과 Drive 저장 경로를 확인하세요."
    )
"""
    ),
    markdown("### 2. ZIP 탐색·압축 해제 및 데이터 검증"),
    code(
        r"""
EXPECTED_ARCHIVES = [
    "한국문화 멀티모달 질의응답.zip",
    "train.zip",
    "validation.zip",
    "test.zip",
]


def normalized_name(text):
    return unicodedata.normalize("NFC", str(text))


def resolve_archive(directory, expected_name):
    expected_nfc = normalized_name(expected_name)
    matches = [
        path for path in directory.glob("*.zip")
        if normalized_name(path.name) == expected_nfc
    ]
    if len(matches) != 1:
        found = [path.name for path in directory.glob("*.zip")]
        raise FileNotFoundError(f"{expected_name!r}을 찾지 못했습니다. ZIP 목록: {found}")
    return matches[0]


archive_paths = [resolve_archive(DRIVE_ZIP_DIR, name) for name in EXPECTED_ARCHIVES]
archive_state = {
    path.name: {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
    for path in archive_paths
}
state_path = EXTRACT_DIR / ".archive_state.json"
previous_state = (
    json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
)

if previous_state == archive_state:
    print("동일한 ZIP이 이미 해제되어 있어 건너뜁니다.")
else:
    for archive_path in archive_paths:
        print("압축 해제:", archive_path.name)
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(EXTRACT_DIR)
    state_path.write_text(
        json.dumps(archive_state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("압축 해제 완료:", EXTRACT_DIR)
"""
    ),
    code(
        r"""
def find_split_json(split):
    matches = [
        path for path in EXTRACT_DIR.rglob("*.json")
        if normalized_name(path.stem).endswith(f"_{split}")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"{split} JSON 후보가 1개가 아닙니다: {matches}")
    return matches[0]


def read_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def find_split_image_dir(split_records):
    sample_names = [row["model_input"]["image_name"] for row in split_records[:10]]
    scores = Counter()
    for image_name in sample_names:
        for candidate in EXTRACT_DIR.rglob(image_name):
            if candidate.is_file():
                scores[candidate.parent] += 1
    if not scores:
        raise FileNotFoundError("이미지 디렉터리를 찾지 못했습니다.")
    return scores.most_common(1)[0][0]


json_paths = {split: find_split_json(split) for split in ["train", "validation", "test"]}
records = {split: read_json(path) for split, path in json_paths.items()}
image_dirs = {split: find_split_image_dir(rows) for split, rows in records.items()}

expected_counts = {"train": 1000, "validation": 200, "test": 800}
summary_rows = []
for split, rows in records.items():
    missing = [
        row["model_input"]["image_name"] for row in rows
        if not (image_dirs[split] / row["model_input"]["image_name"]).is_file()
    ]
    forms = Counter(row["metadata"]["question_form"] for row in rows)
    assert len(rows) == expected_counts[split], (split, len(rows))
    assert not missing, (split, missing[:5])
    summary_rows.append({
        "split": split,
        "rows": len(rows),
        "MC": forms["MC"],
        "SA": forms["SA"],
        "LA": forms["LA"],
        "image_dir": str(image_dirs[split]),
    })

display(pd.DataFrame(summary_rows))
"""
    ),
    markdown(
        r"""
### 3. OCR 후보 문항 분류

기존 EDA의 331개는 질문 문구만 검사한 휴리스틱이었습니다. 여기서는 질문과 선택지를 함께 검사하고 숫자·기관명·장소명처럼 OCR 가능성이 있는 표현을 추가합니다.

이 값 역시 정답 라벨이 아니라 `ocr_routed` 모드의 라우팅 기준이며, validation 결과로 유효성을 확인해야 합니다.
"""
    ),
    code(
        r"""
OCR_CUE_RE = re.compile(
    r"적혀|쓰여|써\s*있|문구|글자|텍스트|안내|표지|간판|현수막|포스터|메뉴|게시판|"
    r"명판|상호|로고|화면|앱|책|연도|날짜|가격|요금|번호|숫자|기관|부처|장소의\s*이름|"
    r"무엇이라\s*하|몇\s*(?:개|명|시|분|원|년|월|일)|읽(?:고|어|으)"
)


def text_for_ocr_routing(record):
    model_input = record["model_input"]
    parts = [model_input.get("question", "")]
    parts.extend(str(option) for option in (model_input.get("options") or []))
    return " ".join(parts)


def is_ocr_candidate(record):
    return bool(OCR_CUE_RE.search(text_for_ocr_routing(record)))


routing_rows = []
for split in ["train", "validation", "test"]:
    for row in records[split]:
        routing_rows.append({
            "split": split,
            "question_form": row["metadata"]["question_form"],
            "ocr_candidate": is_ocr_candidate(row),
        })

routing_df = pd.DataFrame(routing_rows)
display(
    routing_df.groupby(["split", "question_form"])["ocr_candidate"]
    .agg(["count", "sum", "mean"])
)
"""
    ),
    markdown("### 4. PP-OCRv5 Korean 초기화 및 OCR 캐시"),
    code(
        r"""
from paddleocr import PaddleOCR


# PP-OCRv6은 Korean을 지원하지 않으므로 ocr_version="PP-OCRv5"와
# lang="korean"을 함께 명시합니다. OCR는 CPU, Qwen은 GPU에서 실행합니다.
ocr_engine = PaddleOCR(
    lang="korean",
    ocr_version="PP-OCRv5",
    text_detection_model_name="PP-OCRv5_mobile_det",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True,
    text_det_limit_side_len=OCR_MAX_IMAGE_SIDE,
    text_det_limit_type="max",
    text_rec_score_thresh=OCR_SCORE_THRESHOLD,
    device="cpu",
)

print("PP-OCRv5 Korean 초기화 완료")
"""
    ),
    code(
        r"""
Image.MAX_IMAGE_PIXELS = None
OCR_CACHE_PATH = RUN_OUTPUT_DIR / "ppocrv5_cache.json"


def atomic_write_json(data, path):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_path.replace(path)


def image_path_for(record, split):
    return image_dirs[split] / record["model_input"]["image_name"]


def load_original_rgb(path):
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB").copy()


def load_qwen_image(path):
    image = load_original_rgb(path)
    if max(image.size) > INFERENCE_MAX_IMAGE_SIDE:
        image.thumbnail(
            (INFERENCE_MAX_IMAGE_SIDE, INFERENCE_MAX_IMAGE_SIDE),
            Image.Resampling.LANCZOS,
        )
    return image


def cache_key(record, split):
    return f"{split}:{record['metadata']['question_id']}"


if OCR_CACHE_PATH.exists():
    ocr_cache = json.loads(OCR_CACHE_PATH.read_text(encoding="utf-8"))
else:
    ocr_cache = {}


def parse_paddle_result(result):
    payload = result.json
    payload = payload.get("res", payload)
    texts = payload.get("rec_texts") or []
    scores = payload.get("rec_scores") or []
    boxes = payload.get("rec_boxes") or []

    entries = []
    for text, score, box in zip(texts, scores, boxes):
        text = re.sub(r"\s+", " ", str(text)).strip()
        score = float(score)
        if not text or score < OCR_SCORE_THRESHOLD:
            continue
        entries.append({
            "text": text,
            "score": round(score, 6),
            "box": [int(value) for value in box],
        })
    return entries


def run_ocr(record, split, force=False):
    key = cache_key(record, split)
    if key in ocr_cache and not force:
        return ocr_cache[key]

    image = load_original_rgb(image_path_for(record, split))
    results = list(ocr_engine.predict(np.asarray(image)))
    entries = []
    for result in results:
        entries.extend(parse_paddle_result(result))

    ocr_cache[key] = entries
    return entries


print("기존 OCR 캐시:", len(ocr_cache))
"""
    ),
    markdown(
        r"""
### OCR 단독 스모크 테스트

먼저 Qwen을 로드하지 않고 OCR 후보 validation 문항 5개만 처리합니다. 원본 이미지 위의 빨간 box, 인식 텍스트와 신뢰도를 확인하세요.

- 중요한 글자가 box로 탐지되는지
- 한글·숫자·기관명 등이 글자 단위로 맞는지
- 신뢰도 `0.55` 이상 결과에 오인식이 과도하지 않은지

결과가 좋지 않으면 여기에서 멈추고 OCR 설정을 조정합니다. 정상일 때만 다음의 validation 200건 OCR 셀로 넘어가세요.
"""
    ),
    code(
        r"""
from PIL import ImageDraw


def draw_ocr_preview(record, split, entries, max_side=1200):
    image = load_original_rgb(image_path_for(record, split))
    draw = ImageDraw.Draw(image)
    for entry in entries:
        x1, y1, x2, y2 = entry["box"]
        line_width = max(3, int(max(image.size) / 800))
        draw.rectangle((x1, y1, x2, y2), outline="red", width=line_width)
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


ocr_smoke_records = [
    row for row in records["validation"] if is_ocr_candidate(row)
][:OCR_SMOKE_COUNT]

for record in ocr_smoke_records:
    entries = run_ocr(record, "validation")
    print("=" * 80)
    print("ID:", record["metadata"]["question_id"])
    print("유형:", record["metadata"]["question_form"])
    print("질문:", record["model_input"]["question"])
    print("정답:", record["model_output"]["answer"])
    if entries:
        for entry in entries:
            print(f"- [{entry['score']:.3f}] {entry['text']} | box={entry['box']}")
    else:
        print("- 인식된 텍스트 없음")
    display(draw_ocr_preview(record, "validation", entries))

atomic_write_json(ocr_cache, OCR_CACHE_PATH)
print("\nOCR 단독 스모크 테스트 완료. 위 결과를 확인한 뒤 다음 셀로 이동하세요.")
"""
    ),
    code(
        r"""
# 스모크 테스트가 정상일 때 validation 전체 OCR을 계산하고 20건마다 Drive에 저장합니다.
validation_subset = records["validation"][:VALIDATION_MAX_SAMPLES]
new_ocr_count = 0

for record in tqdm(validation_subset, desc="validation PP-OCRv5"):
    key = cache_key(record, "validation")
    if key in ocr_cache:
        continue
    run_ocr(record, "validation")
    new_ocr_count += 1
    if new_ocr_count % OCR_CACHE_SAVE_EVERY == 0:
        atomic_write_json(ocr_cache, OCR_CACHE_PATH)

atomic_write_json(ocr_cache, OCR_CACHE_PATH)
print("validation OCR 완료")
print("OCR 캐시:", OCR_CACHE_PATH)
"""
    ),
    markdown("### OCR 결과 및 글자 crop 확인"),
    code(
        r"""
def normalized_query_tokens(record):
    query = text_for_ocr_routing(record)
    return set(re.findall(r"[가-힣A-Za-z0-9]{2,}", query.lower()))


def crop_rank(entry, record, image_size):
    x1, y1, x2, y2 = entry["box"]
    area_ratio = max(1, x2 - x1) * max(1, y2 - y1) / max(1, image_size[0] * image_size[1])
    text_lower = entry["text"].lower()
    overlap = sum(token in text_lower for token in normalized_query_tokens(record))
    return (
        3.0 * min(overlap, 2)
        + entry["score"]
        + min(len(entry["text"]) / 20, 1.0)
        + min(area_ratio * 50, 1.0)
    )


def make_ocr_crops(record, split, entries, max_crops=MAX_OCR_CROPS):
    original = load_original_rgb(image_path_for(record, split))
    width, height = original.size
    ranked = sorted(
        entries,
        key=lambda entry: crop_rank(entry, record, original.size),
        reverse=True,
    )

    crops = []
    for entry in ranked[:max_crops]:
        x1, y1, x2, y2 = entry["box"]
        padding = max(6, int(0.08 * max(x2 - x1, y2 - y1)))
        box = (
            max(0, x1 - padding),
            max(0, y1 - padding),
            min(width, x2 + padding),
            min(height, y2 + padding),
        )
        crop = original.crop(box)
        if max(crop.size) < OCR_CROP_MAX_SIDE // 2:
            crop = crop.resize(
                (crop.width * 2, crop.height * 2), Image.Resampling.LANCZOS
            )
        if max(crop.size) > OCR_CROP_MAX_SIDE:
            crop.thumbnail(
                (OCR_CROP_MAX_SIDE, OCR_CROP_MAX_SIDE), Image.Resampling.LANCZOS
            )
        crops.append({"image": crop, "entry": entry})
    return crops


def ocr_text_block(entries):
    selected = sorted(
        entries,
        key=lambda entry: (entry["box"][1], entry["box"][0]),
    )[:MAX_OCR_LINES]
    lines = [f"[{entry['score']:.2f}] {entry['text']}" for entry in selected]
    return "\n".join(lines)[:MAX_OCR_TEXT_CHARS]


example_records = [row for row in validation_subset if is_ocr_candidate(row)][:3]
for record in example_records:
    entries = ocr_cache.get(cache_key(record, "validation"), [])
    print("=" * 80)
    print("ID:", record["metadata"]["question_id"])
    print("질문:", record["model_input"]["question"])
    print("정답:", record["model_output"]["answer"])
    print("OCR:\n", ocr_text_block(entries) or "(인식 결과 없음)")
    display(load_qwen_image(image_path_for(record, "validation")))
    for crop_info in make_ocr_crops(record, "validation", entries):
        print("crop:", crop_info["entry"])
        display(crop_info["image"])
"""
    ),
    markdown("### 5. Qwen3-VL 8B와 기존 LoRA adapter 로드"),
    code(
        r"""
from peft import PeftModel
from transformers import AutoProcessor

try:
    from transformers import Qwen3VLForConditionalGeneration as Qwen3VLModelClass
except ImportError:
    from transformers import AutoModelForImageTextToText as Qwen3VLModelClass


compute_dtype = torch.bfloat16 if bf16_supported else torch.float16

processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right"
if processor.tokenizer.pad_token_id is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

base_model = Qwen3VLModelClass.from_pretrained(
    MODEL_ID,
    dtype=compute_dtype,
    device_map={"": 0},
    low_cpu_mem_usage=True,
    attn_implementation="sdpa",
)
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR, is_trainable=False)
model.eval()
model.config.use_cache = True
if hasattr(model.config, "text_config"):
    model.config.text_config.use_cache = True

torch.set_float32_matmul_precision("high")
print("모델:", MODEL_ID)
print("LoRA adapter:", ADAPTER_DIR)
print("dtype:", compute_dtype)
"""
    ),
    markdown("### 6. 공통 프롬프트·후처리·추론 함수"),
    code(
        r"""
SYSTEM_PROMPT = (
    "당신은 이미지와 질문을 함께 이해하는 한국문화 질의응답 모델입니다. "
    "이미지에서 확인한 정보와 한국문화 지식을 바탕으로 답하되, "
    "사용자가 요구한 정답 형식만 출력하고 풀이 과정은 출력하지 마세요."
)


def build_user_text(record, retry=False, ocr_entries=None):
    form = record["metadata"]["question_form"]
    model_input = record["model_input"]
    question = model_input["question"].strip()
    options = model_input.get("options") or []

    parts = [question]
    if options:
        parts.append("선택지:\n" + "\n".join(str(option) for option in options))

    if ocr_entries is not None:
        extracted = ocr_text_block(ocr_entries)
        if extracted:
            parts.append(
                "다음은 원본 이미지에서 PP-OCRv5로 추출한 참고 텍스트입니다. "
                "오인식이 있을 수 있으므로 전체 이미지와 확대 영역을 함께 확인하세요.\n"
                + extracted
            )

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
        raise ValueError(form)

    if retry:
        instruction += " 이전 출력 형식이 잘못되었습니다. 이번에는 반드시 지정 형식만 출력하세요."
    parts.append(instruction)
    return "\n\n".join(parts)


def build_messages(record, full_image, crops=None, ocr_entries=None, retry=False):
    crops = crops or []
    user_content = [{"type": "image", "image": full_image}]

    for index, crop_info in enumerate(crops, start=1):
        user_content.append({
            "type": "text",
            "text": f"확대 글자 영역 {index}: {crop_info['entry']['text']}",
        })
        user_content.append({"type": "image", "image": crop_info["image"]})

    user_content.append({
        "type": "text",
        "text": build_user_text(record, retry=retry, ocr_entries=ocr_entries),
    })
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]
"""
    ),
    code(
        r"""
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


@torch.inference_mode()
def generate_raw(record, split, use_ocr=False, retry=False):
    full_image = load_qwen_image(image_path_for(record, split))
    entries = run_ocr(record, split) if use_ocr else None
    crops = make_ocr_crops(record, split, entries) if entries else []
    messages = build_messages(
        record,
        full_image=full_image,
        crops=crops,
        ocr_entries=entries,
        retry=retry,
    )

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    device = next(model.parameters()).device
    inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    input_length = inputs["input_ids"].shape[1]
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens_for(record["metadata"]["question_form"]),
        do_sample=False,
        num_beams=1,
        use_cache=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    raw_text = processor.batch_decode(
        generated[:, input_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    del full_image, entries, crops, messages, inputs, generated
    return raw_text


def predict_one(record, split, use_ocr=False):
    form = record["metadata"]["question_form"]
    raw_text = generate_raw(record, split, use_ocr=use_ocr, retry=False)
    answer, valid = normalize_prediction(raw_text, form)
    retried = False

    if not valid:
        retried = True
        retry_raw = generate_raw(record, split, use_ocr=use_ocr, retry=True)
        retry_answer, retry_valid = normalize_prediction(retry_raw, form)
        raw_text = raw_text + "\n[RETRY]\n" + retry_raw
        if retry_valid:
            answer, valid = retry_answer, True

    if not valid:
        answer = "1" if form == "MC" else "확인 불가"
    return {"answer": answer, "raw": raw_text, "retried": retried, "valid": valid}
"""
    ),
    markdown("### 모델 입출력 스모크 테스트"),
    code(
        r"""
smoke_record = next(row for row in validation_subset if is_ocr_candidate(row))
baseline_smoke = predict_one(smoke_record, "validation", use_ocr=False)
ocr_smoke = predict_one(smoke_record, "validation", use_ocr=True)

print("ID:", smoke_record["metadata"]["question_id"])
print("질문:", smoke_record["model_input"]["question"])
print("정답:", smoke_record["model_output"]["answer"])
print("Baseline:", baseline_smoke["answer"])
print("OCR 결합:", ocr_smoke["answer"])
"""
    ),
    markdown(
        r"""
### 7. Validation OCR ablation

같은 Qwen LoRA 모델과 생성 설정에서 이미지·질문만 사용한 baseline과 PP-OCRv5 정보를 추가한 결과를 비교합니다. 캐시는 Drive에 저장되므로 중간에 끊겨도 완료한 문항은 건너뜁니다.
"""
    ),
    code(
        r"""
VALIDATION_CACHE_PATH = RUN_OUTPUT_DIR / "validation_ablation_cache.json"

if RESET_VALIDATION_PREDICTION_CACHE and VALIDATION_CACHE_PATH.exists():
    VALIDATION_CACHE_PATH.unlink()

if VALIDATION_CACHE_PATH.exists():
    validation_cache = json.loads(VALIDATION_CACHE_PATH.read_text(encoding="utf-8"))
else:
    validation_cache = {}


def validation_prediction_key(record, mode):
    return f"{mode}:{record['metadata']['question_id']}"


new_prediction_count = 0
for record in tqdm(validation_subset, desc="validation baseline/OCR 추론"):
    for mode, use_ocr in [("baseline", False), ("ocr_augmented", True)]:
        key = validation_prediction_key(record, mode)
        if key in validation_cache:
            continue
        result = predict_one(record, "validation", use_ocr=use_ocr)
        validation_cache[key] = result
        new_prediction_count += 1
        if new_prediction_count % SAVE_EVERY_N_PREDICTIONS == 0:
            atomic_write_json(validation_cache, VALIDATION_CACHE_PATH)

atomic_write_json(validation_cache, VALIDATION_CACHE_PATH)
atomic_write_json(ocr_cache, OCR_CACHE_PATH)
print("Validation 추론 캐시:", VALIDATION_CACHE_PATH)
"""
    ),
    code(
        r"""
def strict_correct(prediction, target, form):
    if form == "MC":
        pred_norm, pred_valid = normalize_mc(prediction)
        target_norm, target_valid = normalize_mc(target)
        return pred_valid and target_valid and pred_norm == target_norm
    if form == "SA":
        return str(prediction).strip() == str(target).strip()
    return None


analysis_rows = []
for record in validation_subset:
    question_id = record["metadata"]["question_id"]
    form = record["metadata"]["question_form"]
    target = record["model_output"]["answer"]
    candidate = is_ocr_candidate(record)
    baseline = validation_cache[validation_prediction_key(record, "baseline")]
    ocr_result = validation_cache[validation_prediction_key(record, "ocr_augmented")]
    routed = ocr_result if candidate else baseline

    for mode, result in [
        ("baseline", baseline),
        ("ocr_augmented", ocr_result),
        ("ocr_routed", routed),
    ]:
        analysis_rows.append({
            "question_id": question_id,
            "question_form": form,
            "ocr_candidate": candidate,
            "mode": mode,
            "prediction": result["answer"],
            "target": target,
            "strict_correct": strict_correct(result["answer"], target, form),
            "la_similarity": (
                SequenceMatcher(None, result["answer"], target).ratio()
                if form == "LA" else np.nan
            ),
            "retried": result["retried"],
            "raw": result["raw"],
        })

validation_ablation_df = pd.DataFrame(analysis_rows)
validation_ablation_path = RUN_OUTPUT_DIR / "validation_ocr_ablation.csv"
validation_ablation_df.to_csv(
    validation_ablation_path, index=False, encoding="utf-8-sig"
)

mc_sa_summary = (
    validation_ablation_df[validation_ablation_df["question_form"].isin(["MC", "SA"])]
    .groupby(["mode", "question_form"])["strict_correct"]
    .agg(["count", "mean"])
)
display(mc_sa_summary)

la_summary = (
    validation_ablation_df[validation_ablation_df["question_form"] == "LA"]
    .groupby("mode")["la_similarity"]
    .agg(["count", "mean"])
)
display(la_summary)
print("분석 CSV:", validation_ablation_path)
"""
    ),
    code(
        r"""
# OCR로 맞게 바뀐 문항과 오히려 틀리게 바뀐 문항을 확인합니다.
pivot = validation_ablation_df.pivot(
    index="question_id",
    columns="mode",
    values=["prediction", "strict_correct"],
)

record_lookup = {
    row["metadata"]["question_id"]: row for row in validation_subset
}
change_rows = []
for question_id in pivot.index:
    record = record_lookup[question_id]
    form = record["metadata"]["question_form"]
    if form not in ["MC", "SA"]:
        continue
    baseline_correct = bool(pivot.loc[question_id, ("strict_correct", "baseline")])
    ocr_correct = bool(pivot.loc[question_id, ("strict_correct", "ocr_augmented")])
    if baseline_correct == ocr_correct:
        continue
    change_rows.append({
        "question_id": question_id,
        "question_form": form,
        "change": "OCR_GAIN" if ocr_correct else "OCR_HARM",
        "ocr_candidate": is_ocr_candidate(record),
        "question": record["model_input"]["question"],
        "target": record["model_output"]["answer"],
        "baseline": pivot.loc[question_id, ("prediction", "baseline")],
        "ocr_augmented": pivot.loc[question_id, ("prediction", "ocr_augmented")],
        "ocr_text": ocr_text_block(ocr_cache.get(cache_key(record, "validation"), [])),
    })

change_df = pd.DataFrame(change_rows)
if not change_df.empty:
    display(change_df.groupby(["question_form", "change"]).size().rename("count"))
    display(change_df)
else:
    print("MC·SA에서 정오답이 바뀐 문항이 없습니다.")

change_path = RUN_OUTPUT_DIR / "validation_ocr_changes.csv"
change_df.to_csv(change_path, index=False, encoding="utf-8-sig")
print("변화 분석:", change_path)
"""
    ),
    markdown(
        r"""
### 8. Test OCR 및 최종 추론

`FINAL_INFERENCE_MODE`의 의미는 다음과 같습니다.

- `baseline`: OCR을 사용하지 않음
- `ocr_all`: 모든 문항에 OCR 텍스트와 crop을 추가
- `ocr_routed`: 질문·선택지 휴리스틱으로 OCR 후보인 문항에만 추가

기본값은 OCR 노이즈를 줄이기 위한 `ocr_routed`입니다. 위 validation 표에서 `ocr_augmented`가 모든 유형에서 더 좋다면 `ocr_all`로 변경할 수 있습니다.
"""
    ),
    code(
        r"""
if FINAL_INFERENCE_MODE not in {"baseline", "ocr_all", "ocr_routed"}:
    raise ValueError(FINAL_INFERENCE_MODE)


def test_uses_ocr(record):
    if FINAL_INFERENCE_MODE == "ocr_all":
        return True
    if FINAL_INFERENCE_MODE == "ocr_routed":
        return is_ocr_candidate(record)
    return False


test_ocr_records = [record for record in records["test"] if test_uses_ocr(record)]
print("OCR 적용 test 문항:", len(test_ocr_records), "/", len(records["test"]))

new_test_ocr_count = 0
for record in tqdm(test_ocr_records, desc="test PP-OCRv5"):
    key = cache_key(record, "test")
    if key in ocr_cache:
        continue
    run_ocr(record, "test")
    new_test_ocr_count += 1
    if new_test_ocr_count % OCR_CACHE_SAVE_EVERY == 0:
        atomic_write_json(ocr_cache, OCR_CACHE_PATH)

atomic_write_json(ocr_cache, OCR_CACHE_PATH)
print("test OCR 캐시 완료:", OCR_CACHE_PATH)
"""
    ),
    code(
        r"""
TEST_CACHE_PATH = RUN_OUTPUT_DIR / f"test_predictions_{FINAL_INFERENCE_MODE}.json"

if RESET_TEST_PREDICTION_CACHE and TEST_CACHE_PATH.exists():
    TEST_CACHE_PATH.unlink()

if TEST_CACHE_PATH.exists():
    test_prediction_cache = json.loads(TEST_CACHE_PATH.read_text(encoding="utf-8"))
else:
    test_prediction_cache = {}

if REUSE_BASELINE_TEST_CACHE and BASELINE_TEST_CACHE_PATH.exists():
    baseline_test_cache = json.loads(
        BASELINE_TEST_CACHE_PATH.read_text(encoding="utf-8")
    )
    print("기존 baseline test 캐시:", len(baseline_test_cache))
else:
    baseline_test_cache = {}

test_ids = {row["metadata"]["question_id"] for row in records["test"]}
test_prediction_cache = {
    question_id: value for question_id, value in test_prediction_cache.items()
    if question_id in test_ids
}

completed_since_save = 0
for record in tqdm(records["test"], desc=f"test {FINAL_INFERENCE_MODE} 추론"):
    question_id = record["metadata"]["question_id"]
    if question_id in test_prediction_cache:
        continue

    use_ocr = test_uses_ocr(record)
    if not use_ocr and question_id in baseline_test_cache:
        cached = baseline_test_cache[question_id]
        result = {
            "answer": cached["answer"],
            "raw": cached.get("raw", ""),
            "retried": cached.get("retried", False),
            "valid": cached.get("valid", True),
        }
        source = "reused_baseline"
    else:
        result = predict_one(record, "test", use_ocr=use_ocr)
        source = "ocr_augmented" if use_ocr else "new_baseline"

    test_prediction_cache[question_id] = {
        **result,
        "question_form": record["metadata"]["question_form"],
        "source": source,
        "ocr_candidate": is_ocr_candidate(record),
    }
    completed_since_save += 1
    if completed_since_save >= SAVE_EVERY_N_PREDICTIONS:
        atomic_write_json(test_prediction_cache, TEST_CACHE_PATH)
        completed_since_save = 0

atomic_write_json(test_prediction_cache, TEST_CACHE_PATH)
print("test 예측 완료:", len(test_prediction_cache))
print("test 캐시:", TEST_CACHE_PATH)
"""
    ),
    markdown("### 9. 제출 JSON·ZIP 생성 및 최종 검증"),
    code(
        r"""
if len(test_prediction_cache) != len(records["test"]):
    missing_ids = [
        row["metadata"]["question_id"] for row in records["test"]
        if row["metadata"]["question_id"] not in test_prediction_cache
    ]
    raise RuntimeError(f"예측이 완료되지 않은 test 문항: {missing_ids[:10]}")

submission = copy.deepcopy(records["test"])
preview_rows = []

for record in submission:
    question_id = record["metadata"]["question_id"]
    form = record["metadata"]["question_form"]
    cached = test_prediction_cache[question_id]
    answer = str(cached["answer"]).strip()
    record["model_output"] = {"answer": answer}
    preview_rows.append({
        "question_id": question_id,
        "question_form": form,
        "image_name": record["model_input"]["image_name"],
        "answer": answer,
        "answer_length": len(answer),
        "source": cached["source"],
        "ocr_candidate": cached["ocr_candidate"],
    })

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

file_stem = f"submission_qwen3vl_8b_lora_ppocrv5_{FINAL_INFERENCE_MODE}"
submission_json_path = RUN_OUTPUT_DIR / f"{file_stem}.json"
submission_zip_path = RUN_OUTPUT_DIR / f"{file_stem}.zip"
submission_preview_path = RUN_OUTPUT_DIR / f"{file_stem}_preview.csv"

atomic_write_json(submission, submission_json_path)
pd.DataFrame(preview_rows).to_csv(
    submission_preview_path, index=False, encoding="utf-8-sig"
)
with zipfile.ZipFile(submission_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write(submission_json_path, arcname=submission_json_path.name)

reloaded_submission = json.loads(submission_json_path.read_text(encoding="utf-8"))
assert len(reloaded_submission) == 800
assert all("model_output" in row for row in reloaded_submission)

print("제출 파일 생성 완료")
print("JSON:", submission_json_path)
print("ZIP :", submission_zip_path)
print("미리보기:", submission_preview_path)
display(pd.DataFrame(preview_rows).groupby(["question_form", "source"]).size())
display(pd.DataFrame(preview_rows).head(10))
"""
    ),
    markdown(
        r"""
### 결과 해석 순서

1. `validation_ocr_ablation.csv`에서 MC·SA의 `baseline`, `ocr_augmented`, `ocr_routed` 정확도 비교
2. `validation_ocr_changes.csv`에서 `OCR_GAIN`과 `OCR_HARM` 문항의 OCR 텍스트 확인
3. OCR이 전체 문항에서 안정적으로 좋아지면 `ocr_all`, 특정 문항에서만 좋아지면 `ocr_routed` 선택
4. PP-OCRv5의 한국어 인식 자체가 자주 틀리면 다음 실험에서 동일한 탐지 box에 KLOCR recognizer 비교

리더보드 제출 전에는 반드시 파일명에 표시된 최종 모드와 validation에서 선택한 모드가 일치하는지 확인하세요.
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": OUTPUT_PATH.name, "provenance": []},
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


# 설치 magic 셀을 제외한 Python 코드 셀의 문법을 빌드 단계에서 검증합니다.
for index, cell in enumerate(cells, start=1):
    if cell["cell_type"] != "code" or index == 3:
        continue
    ast.parse("".join(cell["source"]), filename=f"cell_{index}")

OUTPUT_PATH.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8"
)
print(f"Wrote {OUTPUT_PATH}")
print(f"Cells: {len(cells)}")
