import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "notebooks" / "paddleocr_korean_rag_export.ipynb"


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
## PP-OCRv5 Korean OCR 검증 및 RAG팀 전달 데이터 생성

이 노트북은 **OCR팀 전용**입니다. Qwen 학습, 멀티모달 프롬프트 결합, 리더보드 제출은 수행하지 않습니다.

진행 순서는 다음과 같습니다.

1. 한국어 인식 모델을 이름까지 명시하여 PP-OCRv5 초기화
2. 기존에 실패했던 대표 이미지 3개에서 한글 OCR 결과를 먼저 육안 검수
3. 검수 통과 후 train·validation·test의 고유 이미지를 한 번씩 OCR
4. RAG팀 전달용 이미지 단위 corpus와 질문 연결용 context를 JSONL·CSV·ZIP으로 저장

팀 실험에서 OCR 텍스트를 Qwen 프롬프트에 직접 추가했을 때 점수가 낮아졌으므로, 이 노트북의 결과는 **RAG용 외부 문서 데이터**로만 전달합니다.
"""
    ),
    markdown(
        r"""
### 0. Colab 런타임 준비

OCR만 실행하므로 GPU는 필수가 아닙니다. 설치 충돌을 피하기 위해 새 Colab 런타임에서 시작하세요.

첫 실행에서는 패키지 설치 후 런타임이 한 번 자동 재시작됩니다. 재연결되면 첫 셀부터 다시 실행하세요.
"""
    ),
    code(
        r"""
import os
import subprocess
import sys
from pathlib import Path


ENV_MARKER = Path("/content/.paddleocr_korean_rag_env_v1")

if not ENV_MARKER.exists():
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "paddlepaddle-gpu"],
        check=False,
    )
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "paddlepaddle==3.2.0",
        "-i", "https://www.paddlepaddle.org.cn/packages/stable/cpu/",
    ])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "paddleocr==3.5.0",
        "langchain-text-splitters",
        "pandas>=2.2",
        "tqdm>=4.66",
    ])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "--no-cache-dir", "--force-reinstall", "Pillow==12.3.0",
    ])
    ENV_MARKER.write_text("ready", encoding="utf-8")
    print("설치 완료. 런타임을 재시작합니다. 재연결 후 첫 셀부터 실행하세요.")
    os.kill(os.getpid(), 9)

print("OCR 실행 환경 준비 완료")
"""
    ),
    code(
        r"""
import hashlib
import json
import random
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import PIL
import paddle
import paddleocr
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm.auto import tqdm


Image.MAX_IMAGE_PIXELS = None

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
# Google Drive 경로
DRIVE_PROJECT_DIR = Path("/content/drive/MyDrive/CV_korea")
DRIVE_ZIP_DIR = DRIVE_PROJECT_DIR / "data"
RUN_NAME = "paddleocr_korean_rag_v1"
RUN_OUTPUT_DIR = DRIVE_PROJECT_DIR / "outputs" / RUN_NAME

# Colab 로컬 작업 경로
COLAB_WORK_DIR = Path("/content/kculture_paddleocr_rag")
EXTRACT_DIR = COLAB_WORK_DIR / "extracted"

# OCR 모델: 정확도 우선 detector + 명시적인 한국어 recognizer
DETECTION_MODEL_NAME = "PP-OCRv5_server_det"
RECOGNITION_MODEL_NAME = "korean_PP-OCRv5_mobile_rec"
OCR_VERSION = "PP-OCRv5"
OCR_DEVICE = "cpu"
OCR_MAX_IMAGE_SIDE = 4000

# 낮은 점수도 검수 자료에는 보존하되 RAG 본문에는 더 엄격한 기준을 적용합니다.
MIN_KEEP_SCORE = 0.30
RAG_TEXT_SCORE = 0.50

# 먼저 False 상태로 대표 이미지의 한글 인식을 확인하세요.
# 스모크 테스트가 통과한 뒤 True로 바꾸고 전체 추출 셀부터 다시 실행합니다.
RUN_FULL_EXTRACTION = False
SPLITS_TO_EXPORT = ["train", "validation", "test"]
SAVE_EVERY_N_IMAGES = 20
RESET_OCR_CACHE = False
SEED = 42

for path in [COLAB_WORK_DIR, EXTRACT_DIR, RUN_OUTPUT_DIR]:
    path.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)

print("결과 저장:", RUN_OUTPUT_DIR)
print("검출 모델:", DETECTION_MODEL_NAME)
print("인식 모델:", RECOGNITION_MODEL_NAME)
print("전체 추출 실행:", RUN_FULL_EXTRACTION)
"""
    ),
    markdown("### 2. ZIP 압축 해제 및 데이터 검증"),
    code(
        r"""
EXPECTED_ARCHIVES = [
    "한국문화 멀티모달 질의응답.zip",
    "train.zip",
    "validation.zip",
    "test.zip",
]


def normalized_name(value):
    return unicodedata.normalize("NFC", str(value))


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
    candidates = [
        path for path in EXTRACT_DIR.rglob("*.json")
        if split.lower() in path.stem.lower()
    ]
    valid = []
    for path in candidates:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(value, list) and value and "model_input" in value[0]:
            valid.append(path)
    if len(valid) != 1:
        raise FileNotFoundError(f"{split} JSON을 하나로 결정하지 못했습니다: {valid}")
    return valid[0]


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


json_paths = {split: find_split_json(split) for split in SPLITS_TO_EXPORT}
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
        "questions": len(rows),
        "unique_images": len({row["model_input"]["image_name"] for row in rows}),
        "MC": forms["MC"],
        "SA": forms["SA"],
        "LA": forms["LA"],
    })

display(pd.DataFrame(summary_rows))
print("JSON:", json_paths)
print("이미지 폴더:", image_dirs)
"""
    ),
    markdown("### 3. PP-OCRv5 한국어 모델 초기화"),
    code(
        r"""
from paddleocr import PaddleOCR


ocr_engine = PaddleOCR(
    lang="korean",
    ocr_version=OCR_VERSION,
    text_detection_model_name=DETECTION_MODEL_NAME,
    text_recognition_model_name=RECOGNITION_MODEL_NAME,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True,
    text_det_limit_side_len=OCR_MAX_IMAGE_SIDE,
    text_det_limit_type="max",
    text_rec_score_thresh=MIN_KEEP_SCORE,
    device=OCR_DEVICE,
)

print("PP-OCRv5 Korean 초기화 완료")
print("검출:", DETECTION_MODEL_NAME)
print("인식:", RECOGNITION_MODEL_NAME)
"""
    ),
    markdown("### 4. OCR 실행·캐시·품질 측정 함수"),
    code(
        r"""
OCR_CUE_RE = re.compile(
    r"적혀|쓰여|써\s*있|문구|글자|텍스트|안내|표지|간판|현수막|포스터|메뉴|게시판|"
    r"명판|상호|로고|화면|앱|책|연도|날짜|가격|요금|번호|숫자|기관|부처|장소의\s*이름|"
    r"무엇이라\s*하|몇\s*(?:개|명|시|분|원|년|월|일)|읽(?:고|어|으)"
)
HANGUL_RE = re.compile(r"[가-힣]")

ENGINE_SIGNATURE = {
    "ocr_version": OCR_VERSION,
    "detection_model": DETECTION_MODEL_NAME,
    "recognition_model": RECOGNITION_MODEL_NAME,
    "max_image_side": OCR_MAX_IMAGE_SIDE,
    "min_keep_score": MIN_KEEP_SCORE,
    "rag_text_score": RAG_TEXT_SCORE,
}
OCR_CACHE_PATH = RUN_OUTPUT_DIR / "paddleocr_image_cache.json"


def atomic_write_json(data, path):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_path.replace(path)


if OCR_CACHE_PATH.exists() and not RESET_OCR_CACHE:
    loaded_cache = json.loads(OCR_CACHE_PATH.read_text(encoding="utf-8"))
    if loaded_cache.get("engine_signature") == ENGINE_SIGNATURE:
        ocr_cache = loaded_cache
    else:
        print("OCR 모델 설정이 달라 기존 캐시는 사용하지 않습니다.")
        ocr_cache = {"engine_signature": ENGINE_SIGNATURE, "items": {}}
else:
    ocr_cache = {"engine_signature": ENGINE_SIGNATURE, "items": {}}


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_rgb(path):
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB").copy()


def as_list(value):
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def clean_text(value):
    return re.sub(r"\s+", " ", str(value)).strip()


def parse_paddle_result(result):
    payload = result.json
    payload = payload.get("res", payload)
    texts = as_list(payload.get("rec_texts"))
    scores = as_list(payload.get("rec_scores"))
    boxes = as_list(payload.get("rec_boxes"))
    polygons = as_list(payload.get("dt_polys"))

    lines = []
    for text, score, box in zip(texts, scores, boxes):
        text = clean_text(text)
        score = float(score)
        if not text or score < MIN_KEEP_SCORE:
            continue
        lines.append({
            "text": text,
            "score": round(score, 6),
            "box": [int(value) for value in box],
            "accepted_for_rag": score >= RAG_TEXT_SCORE,
        })

    lines.sort(key=lambda row: (row["box"][1], row["box"][0]))
    return lines, polygons


def build_ocr_item(path, force=False):
    image_hash = sha256_file(path)
    if image_hash in ocr_cache["items"] and not force:
        return ocr_cache["items"][image_hash]

    image = load_rgb(path)
    results = list(ocr_engine.predict(np.asarray(image)))
    lines = []
    detection_polygons = []
    for result in results:
        parsed_lines, polygons = parse_paddle_result(result)
        lines.extend(parsed_lines)
        detection_polygons.extend(polygons)

    accepted = [row for row in lines if row["accepted_for_rag"]]
    scores = [row["score"] for row in accepted]
    ocr_text = "\n".join(row["text"] for row in accepted)
    compact_text = re.sub(r"\s+", "", ocr_text)
    hangul_count = len(HANGUL_RE.findall(ocr_text))
    visible_char_count = len(re.findall(r"[^\s]", ocr_text))

    item = {
        "sha256": image_hash,
        "width": image.width,
        "height": image.height,
        "ocr_text": ocr_text,
        "ocr_text_compact": compact_text,
        "ocr_lines": lines,
        "detection_polygons": detection_polygons,
        "detected_region_count": len(detection_polygons),
        "accepted_line_count": len(accepted),
        "kept_line_count": len(lines),
        "mean_confidence": round(float(np.mean(scores)), 6) if scores else 0.0,
        "min_confidence": round(float(np.min(scores)), 6) if scores else 0.0,
        "hangul_count": hangul_count,
        "hangul_ratio": round(hangul_count / max(1, visible_char_count), 6),
    }
    ocr_cache["items"][image_hash] = item
    return item


def draw_ocr_preview(path, item):
    image = load_rgb(path)
    draw = ImageDraw.Draw(image)

    # 빨간색: detector가 찾은 모든 영역
    for polygon in item["detection_polygons"]:
        points = [(int(point[0]), int(point[1])) for point in polygon]
        if len(points) >= 2:
            draw.line(points + [points[0]], fill="red", width=max(2, image.width // 800))

    # 초록색: RAG 본문에 채택된 인식 결과
    for line in item["ocr_lines"]:
        if not line["accepted_for_rag"]:
            continue
        x1, y1, x2, y2 = line["box"]
        draw.rectangle((x1, y1, x2, y2), outline="lime", width=max(2, image.width // 600))
    return image


def question_id(record):
    return str(record["metadata"]["question_id"])


def image_path_for(record, split):
    return image_dirs[split] / record["model_input"]["image_name"]


def is_ocr_candidate(record):
    model_input = record["model_input"]
    parts = [model_input.get("question", "")]
    parts.extend(str(option) for option in (model_input.get("options") or []))
    return bool(OCR_CUE_RE.search(" ".join(parts)))


print("현재 OCR 캐시 이미지:", len(ocr_cache["items"]))
print("엔진 설정:", ENGINE_SIGNATURE)
"""
    ),
    markdown("### 5. 한글 OCR 우선 검수: 실패했던 대표 이미지 3개"),
    code(
        r"""
SMOKE_EXPECTED = {
    "1904": ["화포천습지생태박물관"],
    "0691": ["첫3번무료"],
    "0774": ["공용냉장실", "개인음식물보관", "유통기한"],
}


def compact_for_match(text):
    return re.sub(r"[^0-9A-Za-z가-힣]", "", str(text)).lower()


record_lookup = {}
for split, split_records in records.items():
    for record in split_records:
        record_lookup[question_id(record)] = (split, record)

smoke_rows = []
for expected_id, expected_phrases in SMOKE_EXPECTED.items():
    if expected_id not in record_lookup:
        print("대표 ID를 찾지 못했습니다:", expected_id)
        continue

    split, record = record_lookup[expected_id]
    path = image_path_for(record, split)
    item = build_ocr_item(path, force=True)
    normalized_ocr = compact_for_match(item["ocr_text"])
    phrase_hits = {
        phrase: compact_for_match(phrase) in normalized_ocr
        for phrase in expected_phrases
    }

    print("=" * 80)
    print("ID:", expected_id, "| split:", split)
    print("질문:", record["model_input"].get("question", ""))
    print("확인 문구:", phrase_hits)
    print("검출 영역:", item["detected_region_count"])
    print("RAG 채택 문장:", item["accepted_line_count"])
    print("한글 글자 수:", item["hangul_count"])
    print("OCR 결과:")
    if item["ocr_lines"]:
        for line in item["ocr_lines"]:
            use_mark = "RAG" if line["accepted_for_rag"] else "검수"
            print(f"  [{line['score']:.3f}] [{use_mark}] {line['text']}")
    else:
        print("  (인식 결과 없음)")
    display(draw_ocr_preview(path, item))

    smoke_rows.append({
        "question_id": expected_id,
        "split": split,
        "expected_phrases": expected_phrases,
        "phrase_hits": phrase_hits,
        "all_expected_found": all(phrase_hits.values()),
        "detected_region_count": item["detected_region_count"],
        "accepted_line_count": item["accepted_line_count"],
        "hangul_count": item["hangul_count"],
        "ocr_text": item["ocr_text"],
    })

atomic_write_json(ocr_cache, OCR_CACHE_PATH)
smoke_df = pd.DataFrame(smoke_rows)
display(smoke_df[[
    "question_id", "all_expected_found", "detected_region_count",
    "accepted_line_count", "hangul_count"
]])
print("스모크 OCR 캐시 저장:", OCR_CACHE_PATH)
"""
    ),
    markdown(
        r"""
### 스모크 테스트 판정 기준

- 빨간 박스가 한글 영역을 감싸지 못하면 **검출 문제**입니다.
- 빨간 박스는 있지만 한글 결과가 틀리면 **인식 문제**입니다.
- 초록 박스와 OCR 문구가 실제 이미지의 핵심 한글과 일치해야 전체 추출로 넘어갑니다.
- 세 문항의 `all_expected_found`가 모두 `True`가 아니더라도 띄어쓰기 정도의 차이는 육안으로 판단할 수 있습니다. 그러나 한글 문구 자체를 놓친다면 전체 추출을 시작하지 마세요.

통과했다고 판단한 경우 위 설정 셀에서 `RUN_FULL_EXTRACTION = True`로 변경하고 다음 셀부터 실행하세요.
"""
    ),
    markdown("### 6. 전체 고유 이미지 OCR 추출"),
    code(
        r"""
image_jobs = {}
for split in SPLITS_TO_EXPORT:
    for record in records[split]:
        image_name = record["model_input"]["image_name"]
        key = f"{split}:{image_name}"
        if key not in image_jobs:
            image_jobs[key] = {
                "split": split,
                "image_name": image_name,
                "path": image_dirs[split] / image_name,
                "question_ids": [],
                "ocr_candidate": False,
            }
        image_jobs[key]["question_ids"].append(question_id(record))
        image_jobs[key]["ocr_candidate"] |= is_ocr_candidate(record)

print("전체 질문:", sum(len(rows) for rows in records.values()))
print("OCR 처리할 고유 이미지:", len(image_jobs))

if not RUN_FULL_EXTRACTION:
    print("전체 추출을 실행하지 않았습니다.")
    print("스모크 결과 확인 후 설정 셀의 RUN_FULL_EXTRACTION을 True로 변경하세요.")
else:
    completed_since_save = 0
    for job in tqdm(image_jobs.values(), desc="PP-OCRv5 Korean 전체 이미지"):
        image_hash = sha256_file(job["path"])
        if image_hash in ocr_cache["items"]:
            continue
        build_ocr_item(job["path"], force=False)
        completed_since_save += 1
        if completed_since_save >= SAVE_EVERY_N_IMAGES:
            atomic_write_json(ocr_cache, OCR_CACHE_PATH)
            completed_since_save = 0

    atomic_write_json(ocr_cache, OCR_CACHE_PATH)
    print("OCR 처리 완료. 캐시 이미지:", len(ocr_cache["items"]))
    print("캐시:", OCR_CACHE_PATH)
"""
    ),
    markdown("### 7. OCR 품질 요약 및 우선 검수 목록"),
    code(
        r"""
if not RUN_FULL_EXTRACTION:
    print("RUN_FULL_EXTRACTION=True로 전체 OCR을 완료한 뒤 실행하세요.")
else:
    image_rows = []
    for job in image_jobs.values():
        image_hash = sha256_file(job["path"])
        item = ocr_cache["items"][image_hash]
        image_rows.append({
            "image_id": image_hash,
            "split": job["split"],
            "image_name": job["image_name"],
            "question_ids": job["question_ids"],
            "ocr_candidate": job["ocr_candidate"],
            "width": item["width"],
            "height": item["height"],
            "ocr_text": item["ocr_text"],
            "ocr_lines": item["ocr_lines"],
            "detected_region_count": item["detected_region_count"],
            "accepted_line_count": item["accepted_line_count"],
            "mean_confidence": item["mean_confidence"],
            "min_confidence": item["min_confidence"],
            "hangul_count": item["hangul_count"],
            "hangul_ratio": item["hangul_ratio"],
        })

    image_df = pd.DataFrame(image_rows)
    quality_summary = (
        image_df.groupby("split")
        .agg(
            images=("image_id", "count"),
            no_text=("accepted_line_count", lambda values: int((values == 0).sum())),
            mean_lines=("accepted_line_count", "mean"),
            mean_confidence=("mean_confidence", "mean"),
            images_with_hangul=("hangul_count", lambda values: int((values > 0).sum())),
        )
        .reset_index()
    )
    quality_summary["no_text_rate"] = quality_summary["no_text"] / quality_summary["images"]
    quality_summary["hangul_image_rate"] = (
        quality_summary["images_with_hangul"] / quality_summary["images"]
    )
    display(quality_summary)

    # OCR 관련 질문인데 결과가 없거나 한글이 전혀 없는 이미지를 최우선 검수 대상으로 둡니다.
    image_df["review_priority"] = (
        4 * (image_df["ocr_candidate"] & (image_df["accepted_line_count"] == 0)).astype(int)
        + 3 * (image_df["ocr_candidate"] & (image_df["hangul_count"] == 0)).astype(int)
        + 2 * (image_df["accepted_line_count"] == 0).astype(int)
        + (image_df["mean_confidence"] < 0.65).astype(int)
    )
    priority_review_df = image_df.sort_values(
        ["review_priority", "mean_confidence"], ascending=[False, True]
    ).head(100).copy()
    display(priority_review_df[[
        "split", "image_name", "ocr_candidate", "review_priority",
        "accepted_line_count", "mean_confidence", "hangul_count", "ocr_text"
    ]].head(20))
"""
    ),
    markdown("### 8. RAG팀 전달용 JSONL·CSV·ZIP 생성"),
    code(
        r"""
def write_jsonl(rows, path):
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


if not RUN_FULL_EXTRACTION:
    print("RUN_FULL_EXTRACTION=True로 전체 OCR을 완료한 뒤 실행하세요.")
else:
    # 동일 파일이 split을 넘어 중복된 경우 SHA-256 기준으로 하나의 이미지 문서로 합칩니다.
    image_documents = {}
    for row in image_rows:
        image_id = row["image_id"]
        if image_id not in image_documents:
            image_documents[image_id] = {
                "schema_version": "paddleocr_korean_rag_v1",
                "document_id": f"image:{image_id}",
                "image_sha256": image_id,
                "splits": [],
                "image_names": [],
                "question_ids": [],
                "ocr_candidate": False,
                "width": row["width"],
                "height": row["height"],
                "ocr_text": row["ocr_text"],
                "ocr_lines": row["ocr_lines"],
                "detected_region_count": row["detected_region_count"],
                "accepted_line_count": row["accepted_line_count"],
                "mean_confidence": row["mean_confidence"],
                "hangul_count": row["hangul_count"],
                "hangul_ratio": row["hangul_ratio"],
                "ocr_model": RECOGNITION_MODEL_NAME,
                "detection_model": DETECTION_MODEL_NAME,
            }
        document = image_documents[image_id]
        document["splits"].append(row["split"])
        document["image_names"].append(row["image_name"])
        document["question_ids"].extend(row["question_ids"])
        document["ocr_candidate"] |= bool(row["ocr_candidate"])

    for document in image_documents.values():
        document["splits"] = sorted(set(document["splits"]))
        document["image_names"] = sorted(set(document["image_names"]))
        document["question_ids"] = sorted(set(document["question_ids"]))

    # 질문 연결용 파일에도 정답과 모델 예측은 넣지 않습니다.
    path_hash_cache = {}
    question_contexts = []
    for split in SPLITS_TO_EXPORT:
        for record in records[split]:
            path = image_path_for(record, split)
            path_key = str(path)
            if path_key not in path_hash_cache:
                path_hash_cache[path_key] = sha256_file(path)
            image_id = path_hash_cache[path_key]
            item = ocr_cache["items"][image_id]
            question_contexts.append({
                "schema_version": "paddleocr_korean_rag_v1",
                "question_id": question_id(record),
                "split": split,
                "question_form": record["metadata"]["question_form"],
                "image_name": record["model_input"]["image_name"],
                "image_sha256": image_id,
                "question": record["model_input"].get("question", ""),
                "options": record["model_input"].get("options") or [],
                "ocr_candidate": is_ocr_candidate(record),
                "ocr_text": item["ocr_text"],
                "ocr_lines": item["ocr_lines"],
                "mean_confidence": item["mean_confidence"],
                "hangul_count": item["hangul_count"],
            })

    image_documents = list(image_documents.values())

    image_jsonl_path = RUN_OUTPUT_DIR / "paddleocr_image_corpus.jsonl"
    image_csv_path = RUN_OUTPUT_DIR / "paddleocr_image_corpus.csv"
    context_jsonl_path = RUN_OUTPUT_DIR / "paddleocr_question_context.jsonl"
    context_csv_path = RUN_OUTPUT_DIR / "paddleocr_question_context.csv"
    review_csv_path = RUN_OUTPUT_DIR / "paddleocr_manual_review_priority.csv"
    summary_json_path = RUN_OUTPUT_DIR / "paddleocr_quality_summary.json"
    delivery_zip_path = RUN_OUTPUT_DIR / "paddleocr_korean_rag_delivery.zip"

    write_jsonl(image_documents, image_jsonl_path)
    write_jsonl(question_contexts, context_jsonl_path)

    image_csv_rows = []
    for document in image_documents:
        row = dict(document)
        row["splits"] = json.dumps(row["splits"], ensure_ascii=False)
        row["image_names"] = json.dumps(row["image_names"], ensure_ascii=False)
        row["question_ids"] = json.dumps(row["question_ids"], ensure_ascii=False)
        row["ocr_lines"] = json.dumps(row["ocr_lines"], ensure_ascii=False)
        image_csv_rows.append(row)
    pd.DataFrame(image_csv_rows).to_csv(image_csv_path, index=False, encoding="utf-8-sig")

    context_csv_rows = []
    for context in question_contexts:
        row = dict(context)
        row["options"] = json.dumps(row["options"], ensure_ascii=False)
        row["ocr_lines"] = json.dumps(row["ocr_lines"], ensure_ascii=False)
        context_csv_rows.append(row)
    pd.DataFrame(context_csv_rows).to_csv(
        context_csv_path, index=False, encoding="utf-8-sig"
    )

    review_columns = [
        "image_id", "split", "image_name", "question_ids", "ocr_candidate",
        "review_priority", "accepted_line_count", "mean_confidence", "hangul_count",
        "ocr_text",
    ]
    review_export = priority_review_df[review_columns].copy()
    review_export["question_ids"] = review_export["question_ids"].apply(
        lambda value: json.dumps(value, ensure_ascii=False)
    )
    review_export["review_status"] = ""
    review_export["corrected_text"] = ""
    review_export["reviewer_note"] = ""
    review_export.to_csv(review_csv_path, index=False, encoding="utf-8-sig")

    quality_payload = {
        "schema_version": "paddleocr_korean_rag_v1",
        "engine_signature": ENGINE_SIGNATURE,
        "image_documents": len(image_documents),
        "question_contexts": len(question_contexts),
        "split_summary": quality_summary.to_dict(orient="records"),
        "smoke_test": smoke_rows,
        "notes": [
            "Qwen 또는 다른 VLM 프롬프트에 OCR 텍스트를 주입하지 않은 OCR 전용 결과입니다.",
            "정답과 모델 예측은 RAG 전달 파일에 포함하지 않았습니다.",
            "RAG 본문은 OCR 신뢰도 기준을 통과한 문장만 포함합니다.",
        ],
    }
    atomic_write_json(quality_payload, summary_json_path)

    delivery_files = [
        image_jsonl_path,
        image_csv_path,
        context_jsonl_path,
        context_csv_path,
        review_csv_path,
        summary_json_path,
    ]
    with zipfile.ZipFile(delivery_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in delivery_files:
            zf.write(path, arcname=path.name)

    print("RAG팀 전달 파일 생성 완료")
    for path in delivery_files:
        print("-", path)
    print("전달용 ZIP:", delivery_zip_path)
    print("이미지 문서:", len(image_documents))
    print("질문 연결 문서:", len(question_contexts))
"""
    ),
    markdown(
        r"""
### RAG팀에 전달할 파일

가장 간단하게는 `paddleocr_korean_rag_delivery.zip` 하나를 전달하면 됩니다.

- `paddleocr_image_corpus.jsonl`: 이미지 한 장당 하나의 RAG 문서. 실제 검색·인덱싱의 기본 파일
- `paddleocr_question_context.jsonl`: 대회 질문과 이미지 OCR을 연결한 분석용 파일
- 같은 이름의 CSV: 사람이 열어 확인하거나 간단히 분석할 때 사용
- `paddleocr_manual_review_priority.csv`: OCR 관련 질문인데 텍스트가 없거나 한글이 없는 우선 검수 대상
- `paddleocr_quality_summary.json`: 사용 모델, 임계값, split별 품질 지표와 스모크 테스트 결과

이 결과는 OCR 텍스트를 Qwen 프롬프트에 넣기 위한 파일이 아니라, RAG팀이 별도의 검색 문서로 인덱싱하기 위한 데이터입니다.
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "CPU",
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


OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUTPUT_PATH)
