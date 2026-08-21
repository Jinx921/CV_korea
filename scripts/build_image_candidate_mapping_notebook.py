from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "notebooks" / "image_candidate_mapping_dino_clip.ipynb"


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "colab": {
        "name": OUTPUT_PATH.name,
        "provenance": [],
        "gpuType": "L4",
    },
    "accelerator": "GPU",
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
## 1단계 — 대회 이미지와 RAG 이미지 후보 매핑

이 노트북은 **Kanana 학습을 실행하지 않습니다.** 먼저 대회 train·validation·test 이미지와
TourAPI RAG DB 이미지를 같은 방식으로 비교해, 이미지별 Top-K 문서 후보를 만듭니다.

- 1차 검색: `facebook/dinov2-base` CLS + `openai/clip-vit-base-patch32` 이미지 임베딩
- 2차 재정렬: DINOv2 patch-token MaxSim
- 대상: train 1,000장 + validation 200장 + test 800장 전체
- 저장 키: `(split, image_name)` — 동일 파일명 충돌 방지
- 출력: 후보 JSONL, 펼친 후보 JSONL, 요약 JSON, 재사용 가능한 이미지 인덱스

이 단계에서는 임계값으로 후보를 제거하지 않습니다. 다른 DB에서 정한 점수 기준을 그대로
적용하면 정상 후보를 잃을 수 있으므로, Top-5와 원점수를 저장하고 이후 텍스트 검색·수동 검수와
결합해 최종 신뢰도를 정합니다.

Colab에서 **런타임 → 런타임 유형 변경 → GPU(L4 권장)** 후 위에서부터 실행하세요.
"""
)


markdown("### 1. 패키지 설치와 Google Drive 연결")


code(
    r'''
import subprocess
import sys


subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "Pillow==11.3.0",
    "transformers>=4.52,<5",
    "tqdm>=4.66",
])

import PIL
import torch
from google.colab import drive


drive.mount("/content/drive")
if not torch.cuda.is_available():
    raise RuntimeError("GPU 런타임이 아닙니다. Colab 런타임 유형을 GPU로 변경하세요.")

print("Pillow:", PIL.__version__)
print("PyTorch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0))
'''
)


markdown("### 2. 경로와 실험 설정")


code(
    r'''
from pathlib import Path


DRIVE_PROJECT_DIR = Path("/content/drive/MyDrive/CV_korea")
DRIVE_ZIP_DIR = DRIVE_PROJECT_DIR / "data"
RAG_DB_DIR = DRIVE_PROJECT_DIR / "tourapi_image_db"
RAG_JSONL = RAG_DB_DIR / "tourapi_with_images.jsonl"

MAPPING_ROOT = RAG_DB_DIR / "mapping"
# Google Drive에서 1GB 이상 memmap을 직접 갱신하면 Errno 5가 발생할 수 있습니다.
# Drive에는 재개 가능한 batch chunk만 두고, 큰 배열은 Colab 로컬 디스크에서 조립합니다.
INDEX_DIR = MAPPING_ROOT / "image_dino_clip_index_chunked"
LOCAL_INDEX_DIR = Path("/content/image_dino_clip_index_work")
OUTPUT_DIR = MAPPING_ROOT / "image_candidates"
EXTRACT_DIR = Path("/content/competition_image_mapping_data")

DINO_MODEL_NAME = "facebook/dinov2-base"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

STAGE1_TOPK_PER_MODEL = 15
FINAL_TOPK = 5
RAG_BATCH_SIZE = 16
QUERY_BATCH_SIZE = 8
MAX_IMAGE_SIDE = 1600
DRIVE_IO_RETRIES = 5

# 인덱스 모델·RAG 입력이 바뀌었을 때만 True로 변경합니다.
REBUILD_RAG_INDEX = False
# 후보 검색 설정을 바꿔 처음부터 다시 검색할 때만 True로 변경합니다.
RESET_IMAGE_CANDIDATES = False
# 빠른 코드 확인용입니다. 실제 실행에서는 둘 다 None을 유지합니다.
RAG_LIMIT = None
QUERY_LIMIT = None

INDEX_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("대회 ZIP 폴더:", DRIVE_ZIP_DIR)
print("RAG 입력:", RAG_JSONL)
print("인덱스 저장:", INDEX_DIR)
print("후보 저장:", OUTPUT_DIR)
'''
)


markdown("### 3. 대회 train·validation·test 이미지 압축 해제 및 검증")


code(
    r'''
import json
import shutil
import zipfile


def atomic_write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def resolve_archive(filename: str) -> Path:
    exact = DRIVE_ZIP_DIR / filename
    if exact.is_file():
        return exact
    matches = [
        path for path in DRIVE_ZIP_DIR.glob("*.zip")
        if path.name.lower() == filename.lower()
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"{filename}을 하나로 찾지 못했습니다: {matches}")
    return matches[0]


expected_counts = {"train": 1000, "validation": 200, "test": 800}
extract_state_path = EXTRACT_DIR / "extract_state.json"
archive_state = {}
archives = {}
for split in expected_counts:
    path = resolve_archive(f"{split}.zip")
    archives[split] = path
    archive_state[split] = {"name": path.name, "size": path.stat().st_size}

previous_state = None
if extract_state_path.exists():
    previous_state = json.loads(extract_state_path.read_text(encoding="utf-8"))

if previous_state != archive_state:
    if EXTRACT_DIR.exists():
        shutil.rmtree(EXTRACT_DIR)
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    for split, archive_path in archives.items():
        target = EXTRACT_DIR / split
        target.mkdir(parents=True, exist_ok=True)
        print("압축 해제:", archive_path.name)
        with zipfile.ZipFile(archive_path) as handle:
            handle.extractall(target)
    atomic_write_json(extract_state_path, archive_state)
else:
    print("동일한 대회 이미지 ZIP이 이미 해제되어 있어 건너뜁니다.")

image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
query_records = []
for split, expected_count in expected_counts.items():
    paths = sorted(
        path for path in (EXTRACT_DIR / split).rglob("*")
        if path.is_file() and path.suffix.lower() in image_extensions
    )
    if len(paths) != expected_count:
        raise RuntimeError(
            f"{split} 이미지 수가 예상과 다릅니다: {len(paths):,} != {expected_count:,}"
        )
    for path in paths:
        query_records.append({
            "split": split,
            "image_name": path.name,
            "image_key": f"{split}:{path.name}",
            "absolute_path": str(path),
        })

if QUERY_LIMIT is not None:
    query_records = query_records[: int(QUERY_LIMIT)]

print("대회 이미지 수:", len(query_records))
for split in expected_counts:
    print(split, sum(row["split"] == split for row in query_records))
'''
)


markdown("### 4. TourAPI RAG 레코드와 이미지 경로 검증")


code(
    r'''
import hashlib


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"{path}:{line_no}가 JSON 객체가 아닙니다.")
            rows.append(value)
    return rows


def resolve_rag_image_path(record):
    raw = str(record.get("image_path", "")).strip()
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else RAG_DB_DIR / path


if not RAG_JSONL.is_file():
    raise FileNotFoundError(
        f"{RAG_JSONL}이 없습니다. TourAPI 이미지 다운로드 노트북의 결과를 확인하세요."
    )

rag_records = []
seen_doc_ids = set()
invalid_counts = {"doc_id": 0, "description": 0, "image": 0, "duplicate": 0}
for source_record in load_jsonl(RAG_JSONL):
    doc_id = str(source_record.get("doc_id", "")).strip()
    description = str(source_record.get("description", "")).strip()
    image_path = resolve_rag_image_path(source_record)
    if not doc_id:
        invalid_counts["doc_id"] += 1
        continue
    if not description:
        invalid_counts["description"] += 1
        continue
    if image_path is None or not image_path.is_file() or image_path.stat().st_size == 0:
        invalid_counts["image"] += 1
        continue
    if doc_id in seen_doc_ids:
        invalid_counts["duplicate"] += 1
        continue
    seen_doc_ids.add(doc_id)
    record = dict(source_record)
    record["doc_id"] = doc_id
    record["source"] = str(record.get("source") or "tourapi")
    record["absolute_image_path"] = str(image_path)
    rag_records.append(record)

if RAG_LIMIT is not None:
    rag_records = rag_records[: int(RAG_LIMIT)]
if not rag_records:
    raise RuntimeError("정상적인 RAG 이미지 레코드가 없습니다.")

fingerprint_source = "\n".join(
    "|".join([
        row["doc_id"],
        str(row.get("title", "")),
        str(row.get("image_path", "")),
        str(Path(row["absolute_image_path"]).stat().st_size),
    ])
    for row in rag_records
)
rag_fingerprint = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()

print("정상 RAG 이미지 레코드:", len(rag_records))
print("제외 통계:", invalid_counts)
print("RAG fingerprint:", rag_fingerprint[:16])
print("예시:", {
    key: rag_records[0].get(key)
    for key in ("doc_id", "source", "title", "image_path")
})
'''
)


markdown("### 5. DINOv2·CLIP 모델 로드")


code(
    r'''
import gc
import io
import numpy as np
import time
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor


ImageFile.LOAD_TRUNCATED_IMAGES = True
device = torch.device("cuda")
model_dtype = torch.float16

dino_processor = AutoImageProcessor.from_pretrained(DINO_MODEL_NAME)
dino_model = AutoModel.from_pretrained(
    DINO_MODEL_NAME,
    torch_dtype=model_dtype,
).to(device).eval()
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
clip_model = CLIPModel.from_pretrained(
    CLIP_MODEL_NAME,
    torch_dtype=model_dtype,
).to(device).eval()


def safe_open_rgb(path, max_side=MAX_IMAGE_SIDE):
    path = Path(path)
    last_error = None
    for attempt in range(1, DRIVE_IO_RETRIES + 1):
        try:
            # PIL이 Drive 파일을 지연 읽기하지 않도록 먼저 bytes로 완전히 읽습니다.
            payload = path.read_bytes()
            image = Image.open(io.BytesIO(payload))
            try:
                image.draft("RGB", (max_side, max_side))
            except Exception:
                pass
            width, height = image.size
            if width * height > 80_000_000:
                raise ValueError(f"지나치게 큰 이미지입니다: {width}x{height}")
            image = image.convert("RGB")
            if max(image.size) > max_side:
                image.thumbnail((max_side, max_side))
            image.load()
            return image
        except OSError as error:
            last_error = error
            if attempt == DRIVE_IO_RETRIES:
                break
            wait_seconds = min(2 ** (attempt - 1), 8)
            print(
                f"Drive 이미지 읽기 재시도 {attempt}/{DRIVE_IO_RETRIES}: "
                f"{path.name} ({error})"
            )
            time.sleep(wait_seconds)
    raise OSError(
        f"{DRIVE_IO_RETRIES}회 재시도 후에도 이미지를 읽지 못했습니다: {path}"
    ) from last_error


def clip_image_features(inputs):
    features = clip_model.get_image_features(**inputs)
    if not torch.is_tensor(features):
        features = features.pooler_output
    return features


@torch.inference_mode()
def embed_images(paths):
    images = [safe_open_rgb(path) for path in paths]
    dino_inputs = dino_processor(images=images, return_tensors="pt").to(
        device, dtype=model_dtype
    )
    dino_hidden = dino_model(**dino_inputs).last_hidden_state
    dino_cls = F.normalize(dino_hidden[:, 0, :], dim=-1)
    dino_patches = F.normalize(dino_hidden[:, 1:, :], dim=-1)

    clip_inputs = clip_processor(images=images, return_tensors="pt").to(
        device, dtype=model_dtype
    )
    clip_features = F.normalize(clip_image_features(clip_inputs), dim=-1)
    return dino_cls, dino_patches, clip_features


dummy_cls, dummy_patches, dummy_clip = embed_images(
    [rag_records[0]["absolute_image_path"]]
)
DINO_CLS_DIM = int(dummy_cls.shape[-1])
DINO_PATCH_COUNT = int(dummy_patches.shape[1])
CLIP_IMAGE_DIM = int(dummy_clip.shape[-1])
print("DINO CLS:", tuple(dummy_cls.shape))
print("DINO patch:", tuple(dummy_patches.shape))
print("CLIP:", tuple(dummy_clip.shape))
del dummy_cls, dummy_patches, dummy_clip
gc.collect()
torch.cuda.empty_cache()
'''
)


markdown("### 6. RAG 이미지 인덱스 생성 또는 이어서 실행")


code(
    r'''
import shutil
from numpy.lib.format import open_memmap


CHUNK_DIR = INDEX_DIR / "chunks"
RAG_META_PATH = INDEX_DIR / "rag_meta.jsonl"
INDEX_STATE_PATH = INDEX_DIR / "index_state.json"
LOCAL_RAG_DINO_CLS_PATH = LOCAL_INDEX_DIR / "rag_dino_cls.npy"
LOCAL_RAG_DINO_PATCH_PATH = LOCAL_INDEX_DIR / "rag_dino_patch.npy"
LOCAL_RAG_CLIP_PATH = LOCAL_INDEX_DIR / "rag_clip.npy"

index_signature = {
    "storage_format": "drive_batch_chunks_v2",
    "rag_fingerprint": rag_fingerprint,
    "record_count": len(rag_records),
    "dino_model": DINO_MODEL_NAME,
    "clip_model": CLIP_MODEL_NAME,
    # config.image_size가 아니라 실제 processor 출력 shape를 사용해야 합니다.
    # DINOv2-base processor는 224x224 crop, patch 14이므로 실제 patch 수는 256입니다.
    "dino_dim": DINO_CLS_DIM,
    "clip_dim": CLIP_IMAGE_DIM,
    "patch_count": DINO_PATCH_COUNT,
}

if REBUILD_RAG_INDEX:
    if CHUNK_DIR.exists():
        shutil.rmtree(CHUNK_DIR)
    for path in [RAG_META_PATH, INDEX_STATE_PATH]:
        if path.exists():
            path.unlink()

CHUNK_DIR.mkdir(parents=True, exist_ok=True)
state = None
if INDEX_STATE_PATH.exists():
    state = json.loads(INDEX_STATE_PATH.read_text(encoding="utf-8"))
    identity_keys = (
        "storage_format",
        "rag_fingerprint",
        "record_count",
        "dino_model",
        "clip_model",
    )
    previous_identity = {key: state.get(key) for key in identity_keys}
    current_identity = {key: index_signature[key] for key in identity_keys}
    if previous_identity != current_identity:
        raise RuntimeError(
            "기존 인덱스와 현재 RAG 입력/모델이 다릅니다. "
            "내용을 확인한 뒤 REBUILD_RAG_INDEX=True로 다시 실행하세요."
        )
    previous_dimensions = {
        key: state.get(key) for key in ("dino_dim", "clip_dim", "patch_count")
    }
    current_dimensions = {
        key: index_signature[key] for key in ("dino_dim", "clip_dim", "patch_count")
    }
    if previous_dimensions != current_dimensions:
        # v1 노트북은 config.image_size로 patch_count=1369를 기록했지만 실제 chunk는
        # processor 출력인 256 patch로 정상 저장했습니다. chunk를 삭제하지 않고
        # state의 shape 정보만 실제 tensor shape로 교정합니다.
        print("인덱스 shape 상태값을 실제 모델 출력에 맞게 교정합니다.")
        print("이전:", previous_dimensions)
        print("현재:", current_dimensions)
        state.update(index_signature)
        atomic_write_json(INDEX_STATE_PATH, state)

if state is None:
    with RAG_META_PATH.open("w", encoding="utf-8") as handle:
        for row in rag_records:
            metadata = {
                key: value for key, value in row.items()
                if key != "absolute_image_path"
            }
            handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
    state = {**index_signature, "completed_chunk_count": 0, "complete": False}
    atomic_write_json(INDEX_STATE_PATH, state)

chunk_ranges = [
    (start, min(start + RAG_BATCH_SIZE, len(rag_records)))
    for start in range(0, len(rag_records), RAG_BATCH_SIZE)
]


def chunk_path_for(start, end):
    return CHUNK_DIR / f"chunk_{start:06d}_{end:06d}.npz"


def copy_to_drive_with_retry(source, destination):
    source = Path(source)
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    last_error = None
    for attempt in range(1, DRIVE_IO_RETRIES + 1):
        try:
            if temporary.exists():
                temporary.unlink()
            shutil.copyfile(source, temporary)
            temporary.replace(destination)
            return
        except OSError as error:
            last_error = error
            if attempt < DRIVE_IO_RETRIES:
                print(
                    f"Drive chunk 저장 재시도 {attempt}/{DRIVE_IO_RETRIES}: "
                    f"{destination.name} ({error})"
                )
                time.sleep(min(2 ** (attempt - 1), 8))
    raise OSError(f"Drive chunk 저장 실패: {destination}") from last_error


for start, end in tqdm(chunk_ranges, desc="RAG 이미지 인덱스"):
    chunk_path = chunk_path_for(start, end)
    if chunk_path.is_file() and chunk_path.stat().st_size > 0:
        continue
    end = min(start + RAG_BATCH_SIZE, len(rag_records))
    paths = [row["absolute_image_path"] for row in rag_records[start:end]]
    dino_cls, dino_patches, clip_features = embed_images(paths)
    local_chunk_path = Path(f"/content/{chunk_path.name}")
    np.savez(
        local_chunk_path,
        dino_cls=dino_cls.float().cpu().numpy().astype(np.float16),
        dino_patch=dino_patches.float().cpu().numpy().astype(np.float16),
        clip=clip_features.float().cpu().numpy().astype(np.float16),
    )
    copy_to_drive_with_retry(local_chunk_path, chunk_path)
    local_chunk_path.unlink(missing_ok=True)
    completed_chunk_count = sum(
        chunk_path_for(chunk_start, chunk_end).is_file()
        for chunk_start, chunk_end in chunk_ranges
    )
    state = {
        **index_signature,
        "completed_chunk_count": completed_chunk_count,
        "complete": completed_chunk_count == len(chunk_ranges),
    }
    atomic_write_json(INDEX_STATE_PATH, state)
    del dino_cls, dino_patches, clip_features
    torch.cuda.empty_cache()

missing_chunks = [
    chunk_path_for(start, end)
    for start, end in chunk_ranges
    if not chunk_path_for(start, end).is_file()
]
if missing_chunks:
    raise RuntimeError("RAG 이미지 인덱스가 아직 완료되지 않았습니다.")

print("Drive chunk 완료. Colab 로컬 검색 배열을 조립합니다.")
if LOCAL_INDEX_DIR.exists():
    shutil.rmtree(LOCAL_INDEX_DIR)
LOCAL_INDEX_DIR.mkdir(parents=True, exist_ok=True)
n = len(rag_records)
rag_dino_cls_memmap = open_memmap(
    LOCAL_RAG_DINO_CLS_PATH,
    mode="w+",
    dtype=np.float16,
    shape=(n, index_signature["dino_dim"]),
)
rag_dino_patch_memmap = open_memmap(
    LOCAL_RAG_DINO_PATCH_PATH,
    mode="w+",
    dtype=np.float16,
    shape=(n, index_signature["patch_count"], index_signature["dino_dim"]),
)
rag_clip_memmap = open_memmap(
    LOCAL_RAG_CLIP_PATH,
    mode="w+",
    dtype=np.float16,
    shape=(n, index_signature["clip_dim"]),
)

for start, end in tqdm(chunk_ranges, desc="로컬 인덱스 조립"):
    drive_chunk_path = chunk_path_for(start, end)
    local_chunk_path = LOCAL_INDEX_DIR / drive_chunk_path.name
    last_error = None
    for attempt in range(1, DRIVE_IO_RETRIES + 1):
        try:
            shutil.copyfile(drive_chunk_path, local_chunk_path)
            break
        except OSError as error:
            last_error = error
            if attempt < DRIVE_IO_RETRIES:
                print(
                    f"Drive chunk 읽기 재시도 {attempt}/{DRIVE_IO_RETRIES}: "
                    f"{drive_chunk_path.name} ({error})"
                )
                time.sleep(min(2 ** (attempt - 1), 8))
    else:
        raise OSError(f"Drive chunk 읽기 실패: {drive_chunk_path}") from last_error
    with np.load(local_chunk_path) as chunk:
        if chunk["dino_cls"].shape[0] != end - start:
            raise RuntimeError(f"손상된 chunk입니다: {chunk_path_for(start, end)}")
        rag_dino_cls_memmap[start:end] = chunk["dino_cls"]
        rag_dino_patch_memmap[start:end] = chunk["dino_patch"]
        rag_clip_memmap[start:end] = chunk["clip"]
    local_chunk_path.unlink(missing_ok=True)

rag_dino_cls_memmap.flush()
rag_dino_patch_memmap.flush()
rag_clip_memmap.flush()
state = {
    **index_signature,
    "completed_chunk_count": len(chunk_ranges),
    "complete": True,
}
atomic_write_json(INDEX_STATE_PATH, state)

print("RAG 이미지 인덱스 완료")
print("DINO CLS:", rag_dino_cls_memmap.shape)
print("DINO patch:", rag_dino_patch_memmap.shape)
print("CLIP:", rag_clip_memmap.shape)
'''
)


markdown("### 7. 대회 이미지별 Top-K 후보 검색 및 중단·재개 저장")


code(
    r'''
RAW_CANDIDATES_PATH = OUTPUT_DIR / "image_candidates_raw.jsonl"
CANDIDATE_STATE_PATH = OUTPUT_DIR / "candidate_state.json"

candidate_signature = {
    "rag_fingerprint": rag_fingerprint,
    "query_count": len(query_records),
    "dino_model": DINO_MODEL_NAME,
    "clip_model": CLIP_MODEL_NAME,
    "stage1_topk_per_model": STAGE1_TOPK_PER_MODEL,
    "final_topk": FINAL_TOPK,
}

if RESET_IMAGE_CANDIDATES:
    for path in [RAW_CANDIDATES_PATH, CANDIDATE_STATE_PATH]:
        if path.exists():
            path.unlink()

if CANDIDATE_STATE_PATH.exists():
    candidate_state = json.loads(CANDIDATE_STATE_PATH.read_text(encoding="utf-8"))
    previous_signature = {key: candidate_state.get(key) for key in candidate_signature}
    if previous_signature != candidate_signature:
        raise RuntimeError(
            "기존 후보 결과와 현재 설정이 다릅니다. 확인 후 "
            "RESET_IMAGE_CANDIDATES=True로 다시 실행하세요."
        )
else:
    candidate_state = {**candidate_signature, "complete": False}
    atomic_write_json(CANDIDATE_STATE_PATH, candidate_state)

done_keys = set()
if RAW_CANDIDATES_PATH.exists():
    for row in load_jsonl(RAW_CANDIDATES_PATH):
        done_keys.add(str(row["image_key"]))

rag_meta = load_jsonl(RAG_META_PATH)
rag_dino_cls = np.asarray(rag_dino_cls_memmap, dtype=np.float32)
rag_clip = np.asarray(rag_clip_memmap, dtype=np.float32)
pending = [row for row in query_records if row["image_key"] not in done_keys]


def top_indices(scores, count):
    count = min(int(count), len(scores))
    if count == len(scores):
        return np.argsort(-scores)
    selected = np.argpartition(-scores, count - 1)[:count]
    return selected[np.argsort(-scores[selected])]


def serializable_candidate(index, dino_scores, clip_scores, dino_rank, clip_rank, late_score):
    metadata = rag_meta[index]
    return {
        "doc_id": str(metadata.get("doc_id", "")),
        "source": str(metadata.get("source", "tourapi")),
        "title": str(metadata.get("title", "")),
        "description": str(metadata.get("description", "")),
        "image_path": str(metadata.get("image_path", "")),
        "image_url": str(metadata.get("image_url", "")),
        "dino_cls_score": round(float(dino_scores[index]), 6),
        "clip_score": round(float(clip_scores[index]), 6),
        "dino_rank": dino_rank.get(index),
        "clip_rank": clip_rank.get(index),
        "late_interaction_score": round(float(late_score), 6),
    }


print(f"전체 {len(query_records):,}장 / 완료 {len(done_keys):,}장 / 남음 {len(pending):,}장")
with RAW_CANDIDATES_PATH.open("a", encoding="utf-8") as output_handle:
    for batch_start in tqdm(
        range(0, len(pending), QUERY_BATCH_SIZE),
        desc="대회 이미지 후보 검색",
    ):
        batch = pending[batch_start : batch_start + QUERY_BATCH_SIZE]
        q_cls, q_patches, q_clip = embed_images([row["absolute_path"] for row in batch])
        q_cls_np = q_cls.float().cpu().numpy()
        q_clip_np = q_clip.float().cpu().numpy()

        for batch_index, query in enumerate(batch):
            dino_scores = rag_dino_cls @ q_cls_np[batch_index]
            clip_scores = rag_clip @ q_clip_np[batch_index]
            dino_top = top_indices(dino_scores, STAGE1_TOPK_PER_MODEL).tolist()
            clip_top = top_indices(clip_scores, STAGE1_TOPK_PER_MODEL).tolist()
            candidate_indices = sorted(set(dino_top) | set(clip_top))
            dino_rank = {index: rank + 1 for rank, index in enumerate(dino_top)}
            clip_rank = {index: rank + 1 for rank, index in enumerate(clip_top)}

            candidate_patches = torch.from_numpy(
                np.asarray(rag_dino_patch_memmap[candidate_indices], dtype=np.float16)
            ).to(device)
            similarities = torch.einsum(
                "qd,ckd->cqk",
                q_patches[batch_index],
                candidate_patches,
            )
            late_scores = similarities.max(dim=-1).values.mean(dim=-1).float().cpu().numpy()
            order = np.argsort(-late_scores)

            candidates = []
            for position in order[:FINAL_TOPK]:
                rag_index = candidate_indices[int(position)]
                candidates.append(serializable_candidate(
                    rag_index,
                    dino_scores,
                    clip_scores,
                    dino_rank,
                    clip_rank,
                    late_scores[int(position)],
                ))

            top_score = candidates[0]["late_interaction_score"] if candidates else None
            second_score = candidates[1]["late_interaction_score"] if len(candidates) > 1 else None
            result = {
                "split": query["split"],
                "image_name": query["image_name"],
                "image_key": query["image_key"],
                "method": "dinov2_cls_clip_union_then_dinov2_patch_maxsim",
                "top_score": top_score,
                "top1_top2_margin": (
                    round(top_score - second_score, 6)
                    if top_score is not None and second_score is not None else None
                ),
                "candidates": candidates,
            }
            output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_handle.flush()
            done_keys.add(query["image_key"])
            del candidate_patches, similarities

        candidate_state = {
            **candidate_signature,
            "completed_query_count": len(done_keys),
            "complete": len(done_keys) == len(query_records),
        }
        atomic_write_json(CANDIDATE_STATE_PATH, candidate_state)
        del q_cls, q_patches, q_clip
        torch.cuda.empty_cache()

print("후보 검색 완료:", RAW_CANDIDATES_PATH)
'''
)


markdown("### 8. 최종 후보 파일 정리 및 품질 통계")


code(
    r'''
FINAL_CANDIDATES_PATH = OUTPUT_DIR / "image_candidates.jsonl"
FLAT_CANDIDATES_PATH = OUTPUT_DIR / "image_candidate_pairs.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "image_candidate_summary.json"

raw_rows = load_jsonl(RAW_CANDIDATES_PATH)
latest_by_key = {row["image_key"]: row for row in raw_rows}
ordered_rows = [
    latest_by_key[row["image_key"]]
    for row in query_records
    if row["image_key"] in latest_by_key
]

if len(ordered_rows) != len(query_records):
    missing = [
        row["image_key"] for row in query_records
        if row["image_key"] not in latest_by_key
    ]
    raise RuntimeError(f"후보 검색이 완료되지 않았습니다. 누락 예시: {missing[:5]}")

with FINAL_CANDIDATES_PATH.with_suffix(".jsonl.tmp").open("w", encoding="utf-8") as handle:
    for row in ordered_rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
FINAL_CANDIDATES_PATH.with_suffix(".jsonl.tmp").replace(FINAL_CANDIDATES_PATH)

flat_rows = []
for row in ordered_rows:
    for rank, candidate in enumerate(row["candidates"], start=1):
        flat_rows.append({
            "split": row["split"],
            "image_name": row["image_name"],
            "image_key": row["image_key"],
            "image_rank": rank,
            **candidate,
        })

with FLAT_CANDIDATES_PATH.with_suffix(".jsonl.tmp").open("w", encoding="utf-8") as handle:
    for row in flat_rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
FLAT_CANDIDATES_PATH.with_suffix(".jsonl.tmp").replace(FLAT_CANDIDATES_PATH)

top_scores = np.array([
    row["top_score"] for row in ordered_rows if row.get("top_score") is not None
], dtype=np.float32)
margins = np.array([
    row["top1_top2_margin"]
    for row in ordered_rows if row.get("top1_top2_margin") is not None
], dtype=np.float32)

summary = {
    "method": "dinov2_cls_clip_union_then_dinov2_patch_maxsim",
    "dino_model": DINO_MODEL_NAME,
    "clip_model": CLIP_MODEL_NAME,
    "rag_record_count": len(rag_records),
    "query_image_count": len(ordered_rows),
    "candidate_pair_count": len(flat_rows),
    "split_counts": {
        split: sum(row["split"] == split for row in ordered_rows)
        for split in expected_counts
    },
    "top_score_quantiles": {
        str(q): round(float(np.quantile(top_scores, q)), 6)
        for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    },
    "top1_top2_margin_quantiles": {
        str(q): round(float(np.quantile(margins, q)), 6)
        for q in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
    },
    "note": "이 단계에서는 임계값과 최종 신뢰도를 확정하지 않았습니다.",
}
atomic_write_json(SUMMARY_PATH, summary)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print("\n생성 파일")
print("1.", FINAL_CANDIDATES_PATH)
print("2.", FLAT_CANDIDATES_PATH)
print("3.", SUMMARY_PATH)
'''
)


markdown("### 9. 상위 후보 이미지 육안 확인")


code(
    r'''
import matplotlib.pyplot as plt


query_by_key = {row["image_key"]: row for row in query_records}
rag_by_doc_id = {str(row["doc_id"]): row for row in rag_records}

# 최고 점수만 몰아보지 않도록 점수 분위별로 예시를 선택합니다.
sorted_rows = sorted(ordered_rows, key=lambda row: row["top_score"])
sample_positions = np.linspace(0, len(sorted_rows) - 1, 6, dtype=int)
sample_rows = [sorted_rows[position] for position in sample_positions]

figure, axes = plt.subplots(len(sample_rows), 2, figsize=(12, 4 * len(sample_rows)))
for row_index, result in enumerate(sample_rows):
    query = query_by_key[result["image_key"]]
    top_candidate = result["candidates"][0]
    rag_record = rag_by_doc_id[top_candidate["doc_id"]]

    query_image = safe_open_rgb(query["absolute_path"])
    rag_image = safe_open_rgb(rag_record["absolute_image_path"])
    axes[row_index, 0].imshow(query_image)
    axes[row_index, 0].set_title(f'QUERY {result["image_key"]}')
    axes[row_index, 0].axis("off")
    axes[row_index, 1].imshow(rag_image)
    axes[row_index, 1].set_title(
        f'RAG {top_candidate["title"]}\n'
        f'late={top_candidate["late_interaction_score"]:.3f}, '
        f'margin={result["top1_top2_margin"]:.3f}'
    )
    axes[row_index, 1].axis("off")

plt.tight_layout()
plt.show()
'''
)


markdown(
    """
### 10. 이 단계에서 확인하고 공유할 파일

우선 다음 세 파일만 내려받아 확인하면 됩니다.

1. `image_candidates.jsonl` — 대회 이미지별 Top-5 후보
2. `image_candidate_pairs.jsonl` — 이미지–문서 후보를 한 줄씩 펼친 파일
3. `image_candidate_summary.json` — 건수와 점수 분포

`image_dino_clip_index` 폴더는 다음 실행에서 임베딩을 재사용하기 위한 중간 인덱스입니다.
크기가 크므로 팀원에게 바로 전달할 필요는 없지만, 본인의 Drive에서는 삭제하지 마세요.

다음 단계에서는 질문·선택지·OCR 텍스트로 BM25 후보를 만들고, 여기서 생성한 이미지 후보와
`doc_id` 기준으로 합칩니다.
"""
)


OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUTPUT_PATH)
print(OUTPUT_PATH)
