from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "notebooks" / "tourapi_drive_image_download.ipynb"


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "colab": {"name": OUTPUT_PATH.name, "provenance": []},
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.x"},
}


def markdown(source: str) -> None:
    nb.cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source: str) -> None:
    nb.cells.append(nbf.v4.new_code_cell(source.strip()))


markdown(
    """
## TourAPI 이미지 Google Drive 증분 다운로드

이 노트북은 `tourapi_collected_records.jsonl`의 이미지 URL을 Google Drive에 저장합니다.

- API 키가 필요하지 않으며 TourAPI 일일 호출 한도에 포함되지 않습니다.
- 파일명은 `doc_id.jpg`로 통일합니다.
- 정상적으로 저장된 기존 이미지는 자동으로 건너뜁니다.
- 다운로드 직후 Pillow로 이미지를 검증합니다.
- 런타임이 끊겨도 다시 실행하면 기존 파일 다음부터 이어집니다.
- 완료 후 이미지 상대 경로가 반영된 JSONL, 실패 목록, 다운로드 요약을 생성합니다.

처음 실행할 때 로컬 컴퓨터의 `tourapi_collected_records.jsonl`을 업로드합니다. 다음 날 새로운 설명을 수집했다면 설정 셀의 `UPLOAD_NEW_JSONL=True`로 변경하고 최신 누적 JSONL을 다시 업로드하세요.

GPU는 필요하지 않습니다. 코랩 기본 CPU 런타임으로 실행하면 됩니다.
"""
)


markdown("### 1. 패키지 설치 및 Google Drive 연결")


code(
    r"""
%pip install -q -U "requests>=2.31" "Pillow>=10.4" "tqdm>=4.66"

from google.colab import drive

drive.mount("/content/drive")
print("Google Drive 연결 완료")
"""
)


markdown("### 2. 저장 경로 및 다운로드 설정")


code(
    r"""
from pathlib import Path

# 저장 폴더는 필요에 따라 변경할 수 있습니다.
DRIVE_DB_DIR = Path("/content/drive/MyDrive/CV_korea/tourapi_image_db")
INPUT_JSONL = DRIVE_DB_DIR / "tourapi_collected_records.jsonl"
IMAGE_DIR = DRIVE_DB_DIR / "images"
MANIFEST_PATH = DRIVE_DB_DIR / "download_manifest.jsonl"
OUTPUT_JSONL = DRIVE_DB_DIR / "tourapi_with_images.jsonl"
FAILURES_PATH = DRIVE_DB_DIR / "download_failures.jsonl"
SUMMARY_PATH = DRIVE_DB_DIR / "download_summary.json"

# 최초 실행에서는 INPUT_JSONL이 없으므로 자동으로 업로드 창이 열립니다.
# 다음 날 최신 누적 JSONL로 교체할 때만 True로 변경하세요.
UPLOAD_NEW_JSONL = False

# 이미지 RAG와 유사도 비교에 충분한 해상도를 유지하면서 용량을 줄입니다.
# 원본 크기를 유지하려면 None으로 바꾸세요.
MAX_LONG_EDGE = 1600
JPEG_QUALITY = 92

# Drive 쓰기 안정성을 고려한 보수적 동시 다운로드 수입니다.
WORKERS = 8
CONNECT_TIMEOUT = 15
READ_TIMEOUT = 90
MAX_RETRIES = 3
MAX_DOWNLOAD_MIB = 60
VERIFY_EXISTING = False

DRIVE_DB_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

print("DB 폴더   :", DRIVE_DB_DIR)
print("이미지 폴더:", IMAGE_DIR)
print("입력 JSONL:", INPUT_JSONL)
"""
)


markdown("### 3. 최신 누적 JSONL 업로드")


code(
    r"""
from google.colab import files


def upload_jsonl_to_drive(destination: Path) -> None:
    print("로컬 컴퓨터에서 tourapi_collected_records.jsonl을 선택하세요.")
    uploaded = files.upload()
    jsonl_files = [
        (name, content)
        for name, content in uploaded.items()
        if name.lower().endswith(".jsonl")
    ]
    if len(jsonl_files) != 1:
        raise RuntimeError("JSONL 파일을 정확히 1개 업로드해야 합니다.")
    _, content = jsonl_files[0]
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(destination)
    print(f"Drive에 입력 JSONL 저장 완료: {destination}")


if UPLOAD_NEW_JSONL or not INPUT_JSONL.exists():
    upload_jsonl_to_drive(INPUT_JSONL)
else:
    print("기존 Drive 입력 JSONL을 사용합니다:", INPUT_JSONL)

assert INPUT_JSONL.exists(), INPUT_JSONL
"""
)


markdown("### 4. 입력 데이터 확인")


code(
    r"""
import json


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"JSONL {line_no}번째 줄 오류: {path}") from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"JSONL {line_no}번째 줄이 객체가 아닙니다.")
            rows.append(value)
    return rows


raw_records = load_jsonl(INPUT_JSONL)
records_by_id = {}
for record in raw_records:
    doc_id = str(record.get("doc_id", "")).strip()
    image_url = str(record.get("image_url", "")).strip()
    if not doc_id or not image_url:
        continue
    records_by_id.setdefault(doc_id, record)

records = list(records_by_id.values())

print(f"입력 행 수            : {len(raw_records):,}")
print(f"고유 다운로드 대상 수 : {len(records):,}")
print(f"중복·필수값 누락 제외 : {len(raw_records) - len(records):,}")
print("첫 번째 대상:", {k: records[0].get(k) for k in ("doc_id", "title", "image_url")})
"""
)


markdown("### 5. 이미지 증분 다운로드")


code(
    r"""
import io
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests
from PIL import Image, ImageOps
from tqdm.auto import tqdm


thread_local = threading.local()
TEMP_DIR = Path("/content/tourapi_image_download_tmp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def get_session():
    if not hasattr(thread_local, "session"):
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": "KCultureRAG/1.0 (TourAPI image corpus)",
                "Referer": "https://korean.visitkorea.or.kr/",
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            }
        )
        thread_local.session = session
    return thread_local.session


def safe_doc_id(value):
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value).strip())
    if not safe:
        raise ValueError("빈 doc_id")
    return safe


def validate_image(path):
    try:
        if not path.exists() or path.stat().st_size == 0:
            return False
        if not VERIFY_EXISTING:
            return True
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def to_rgb(image):
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba).convert("RGB")
    else:
        image = image.convert("RGB")
    return image


def normalize_and_save(download_path, destination):
    with Image.open(download_path) as source:
        source.load()
        original_width, original_height = source.size
        image = to_rgb(source)
        if MAX_LONG_EDGE is not None and max(image.size) > MAX_LONG_EDGE:
            image.thumbnail(
                (MAX_LONG_EDGE, MAX_LONG_EDGE),
                Image.Resampling.LANCZOS,
            )
        width, height = image.size
        drive_temporary = destination.with_suffix(".jpg.tmp")
        with drive_temporary.open("wb") as handle:
            image.save(
                handle,
                format="JPEG",
                quality=JPEG_QUALITY,
                optimize=True,
                progressive=True,
            )
        drive_temporary.replace(destination)
    return original_width, original_height, width, height


def download_one(record):
    doc_id = str(record.get("doc_id", "")).strip()
    title = str(record.get("title", "")).strip()
    image_url = str(record.get("image_url", "")).strip()
    safe_id = safe_doc_id(doc_id)
    destination = IMAGE_DIR / f"{safe_id}.jpg"

    if validate_image(destination):
        return {
            "doc_id": doc_id,
            "title": title,
            "status": "skipped_existing",
            "image_path": f"images/{safe_id}.jpg",
            "bytes": destination.stat().st_size,
            "error": "",
        }

    if destination.exists():
        destination.unlink()

    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        download_path = TEMP_DIR / f"{safe_id}.download"
        try:
            total_bytes = 0
            with get_session().get(
                image_url,
                stream=True,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                allow_redirects=True,
            ) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                if content_type and not content_type.startswith("image/"):
                    raise RuntimeError(f"이미지 응답이 아님: {content_type}")
                with download_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        total_bytes += len(chunk)
                        if total_bytes > MAX_DOWNLOAD_MIB * 1024 * 1024:
                            raise RuntimeError(
                                f"다운로드 크기가 {MAX_DOWNLOAD_MIB} MiB를 초과"
                            )
                        handle.write(chunk)

            original_width, original_height, width, height = normalize_and_save(
                download_path, destination
            )
            download_path.unlink(missing_ok=True)
            return {
                "doc_id": doc_id,
                "title": title,
                "status": "downloaded",
                "image_path": f"images/{safe_id}.jpg",
                "source_bytes": total_bytes,
                "bytes": destination.stat().st_size,
                "original_width": original_width,
                "original_height": original_height,
                "width": width,
                "height": height,
                "error": "",
            }
        except Exception as exc:
            download_path.unlink(missing_ok=True)
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < MAX_RETRIES:
                time.sleep(2 ** (attempt - 1))

    return {
        "doc_id": doc_id,
        "title": title,
        "status": "failed",
        "image_path": "",
        "bytes": 0,
        "error": last_error[:500],
    }


results = []
with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    futures = [executor.submit(download_one, record) for record in records]
    for future in tqdm(
        as_completed(futures),
        total=len(futures),
        desc="TourAPI 이미지",
    ):
        results.append(future.result())

results.sort(key=lambda row: row["doc_id"])
print("다운로드 작업 완료:", len(results))
"""
)


markdown("### 6. Manifest와 이미지 경로 포함 JSONL 생성")


code(
    r"""
from collections import Counter


def atomic_write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_write_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


result_by_id = {row["doc_id"]: row for row in results}
success_statuses = {"downloaded", "skipped_existing"}

output_records = []
failures = []
for record in records:
    doc_id = str(record["doc_id"])
    result = result_by_id[doc_id]
    if result["status"] in success_statuses:
        updated = dict(record)
        updated["image_path"] = result["image_path"]
        output_records.append(updated)
    else:
        failures.append(
            {
                "doc_id": doc_id,
                "title": record.get("title", ""),
                "image_url": record.get("image_url", ""),
                "error": result.get("error", ""),
            }
        )

atomic_write_jsonl(MANIFEST_PATH, results)
atomic_write_jsonl(OUTPUT_JSONL, output_records)
atomic_write_jsonl(FAILURES_PATH, failures)

status_counts = Counter(row["status"] for row in results)
total_image_bytes = sum(
    path.stat().st_size for path in IMAGE_DIR.glob("*.jpg") if path.is_file()
)
summary = {
    "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "input_record_count": len(records),
    "output_record_count": len(output_records),
    "failure_count": len(failures),
    "status_counts": dict(status_counts),
    "drive_image_count": sum(1 for _ in IMAGE_DIR.glob("*.jpg")),
    "drive_image_gib": round(total_image_bytes / (1024 ** 3), 4),
    "max_long_edge": MAX_LONG_EDGE,
    "jpeg_quality": JPEG_QUALITY,
    "image_dir": str(IMAGE_DIR),
    "output_jsonl": str(OUTPUT_JSONL),
}
atomic_write_json(SUMMARY_PATH, summary)

print(json.dumps(summary, ensure_ascii=False, indent=2))
if failures:
    print(f"실패 {len(failures)}건은 노트북을 다시 실행하면 재시도됩니다.")
else:
    print("모든 이미지를 정상적으로 확보했습니다.")
"""
)


markdown(
    """
### 다음 날 증분 다운로드 방법

1. 로컬에서 설명 API를 추가 수집하고 최신 `tourapi_collected_records.jsonl`을 생성합니다.
2. 이 노트북의 설정 셀에서 `UPLOAD_NEW_JSONL=True`로 변경합니다.
3. 모든 셀을 실행하고 최신 누적 JSONL을 업로드합니다.
4. 기존 `images/{doc_id}.jpg`는 건너뛰고 새로 추가된 이미지만 다운로드됩니다.
5. 완료 후 다시 `UPLOAD_NEW_JSONL=False`로 돌려놓아도 됩니다.

임베딩과 Qdrant 적재에는 Drive의 `tourapi_with_images.jsonl`을 사용합니다.
"""
)


OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUTPUT_PATH)
print(f"작성 완료: {OUTPUT_PATH}")
