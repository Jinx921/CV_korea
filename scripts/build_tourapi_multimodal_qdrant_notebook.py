from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "notebooks" / "tourapi_multimodal_qdrant_build.ipynb"


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
## TourAPI 텍스트·이미지 멀티모달 RAG 구축

이 노트북은 Google Drive에 모아 둔 `tourapi_with_images.jsonl`과 이미지를 사용해 검색 DB를 구축합니다.

- 텍스트 임베딩: `BAAI/bge-m3`
- 이미지 임베딩: `google/siglip2-base-patch16-224`
- 벡터 DB: Qdrant Local Mode의 named vectors (`text_vector`, `image_vector`)
- 검색: 한국어 의미 검색, 이미지 유사도 검색, 텍스트→이미지 검색, RRF 기반 결합 검색
- 증분 처리: Qdrant에 이미 들어간 `doc_id`는 건너뛰고 새 데이터만 임베딩·적재
- 저장: Qdrant DB를 압축해 Google Drive에 보관하고 다음 실행 때 자동 복원

실행 전 Colab에서 **런타임 → 런타임 유형 변경 → GPU**를 선택하세요. L4가 권장되며 T4에서도 실행할 수 있습니다.
"""
)


markdown("### 1. 패키지 설치, 런타임 1회 재시작 및 Google Drive 연결")


code(
    r"""
import os
import signal
import subprocess
import sys
from pathlib import Path


# Colab 기본 Pillow와 pip로 갱신된 Pillow가 섞이면
# `cannot import name '_Ink' from PIL._typing` 오류가 발생할 수 있습니다.
# 최초 1회만 Pillow를 완전히 재설치한 뒤 Python 프로세스를 자동 재시작합니다.
SETUP_MARKER = Path("/content/.tourapi_multimodal_rag_dependencies_v2")

if not SETUP_MARKER.exists():
    print("RAG 패키지를 설치합니다. 완료 후 런타임이 한 번 자동 재시작됩니다.")
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "Pillow"],
        check=False,
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-cache-dir",
            "--force-reinstall",
            "Pillow==11.3.0",
        ]
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-U",
            "sentence-transformers>=3.4,<6",
            "transformers>=4.53,<6",
            "qdrant-client>=1.14,<2",
            "tqdm>=4.66",
        ]
    )
    SETUP_MARKER.write_text("ready\n", encoding="utf-8")
    print("설치 완료. 런타임을 재시작합니다. 재연결 후 '모두 실행'을 다시 누르세요.")
    os.kill(os.getpid(), signal.SIGKILL)

import PIL
from PIL import Image, ImageOps
from google.colab import drive

print("Pillow 정상 로드:", PIL.__version__)
drive.mount("/content/drive")
print("Google Drive 연결 완료")
"""
)


markdown("### 2. 경로와 임베딩 설정")


code(
    r"""
from pathlib import Path


DRIVE_DB_DIR = Path("/content/drive/MyDrive/CV_korea/tourapi_image_db")
INPUT_JSONL = DRIVE_DB_DIR / "tourapi_with_images.jsonl"

# Qdrant Local DB는 Colab 로컬 디스크에서 사용한 뒤 Drive에 ZIP으로 저장합니다.
# Drive 폴더에서 DB를 직접 열면 파일 잠금과 작은 파일 I/O 때문에 느려질 수 있습니다.
QDRANT_ARCHIVE = DRIVE_DB_DIR / "tourapi_multimodal_qdrant.zip"
INDEX_STATE_PATH = DRIVE_DB_DIR / "tourapi_multimodal_qdrant_state.json"
BUILD_SUMMARY_PATH = DRIVE_DB_DIR / "tourapi_multimodal_qdrant_summary.json"
WORK_DIR = Path("/content/tourapi_multimodal_qdrant")

COLLECTION_NAME = "tourapi_multimodal"
TEXT_VECTOR_NAME = "text_vector"
IMAGE_VECTOR_NAME = "image_vector"
TEXT_MODEL_NAME = "BAAI/bge-m3"
IMAGE_MODEL_NAME = "google/siglip2-base-patch16-224"

TEXT_BATCH_SIZE = 32
IMAGE_BATCH_SIZE = 32
UPSERT_BATCH_SIZE = 64
TEXT_MAX_LENGTH = 1024

# 처음에는 False로 둡니다. 임베딩 모델을 바꾸거나 DB를 완전히 다시 만들 때만 True로 변경하세요.
REBUILD_INDEX = False

# 일부 기존 문서만 다시 임베딩해야 할 때 doc_id를 문자열로 추가할 수 있습니다.
FORCE_REINDEX_DOC_IDS = set()

# 빠른 동작 확인용입니다. 실제 DB 구축 시에는 None을 유지하세요.
INDEX_LIMIT = None

DRIVE_DB_DIR.mkdir(parents=True, exist_ok=True)

print("입력 JSONL   :", INPUT_JSONL)
print("Qdrant ZIP   :", QDRANT_ARCHIVE)
print("텍스트 모델  :", TEXT_MODEL_NAME)
print("이미지 모델  :", IMAGE_MODEL_NAME)
"""
)


markdown("### 3. 입력 JSONL과 이미지 경로 검증")


code(
    r"""
import json
from collections import Counter


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


def resolve_image_path(record):
    value = str(record.get("image_path", "")).strip()
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else DRIVE_DB_DIR / path


assert INPUT_JSONL.exists(), (
    f"입력 파일이 없습니다: {INPUT_JSONL}\n"
    "이미지 다운로드 노트북을 완료했는지 확인하세요."
)

raw_records = load_jsonl(INPUT_JSONL)
records_by_id = {}
rejection_reasons = Counter()

for record in raw_records:
    doc_id = str(record.get("doc_id", "")).strip()
    description = str(record.get("description", "")).strip()
    image_path = resolve_image_path(record)

    if not doc_id:
        rejection_reasons["doc_id 누락"] += 1
        continue
    if not description:
        rejection_reasons["description 누락"] += 1
        continue
    if image_path is None or not image_path.is_file() or image_path.stat().st_size == 0:
        rejection_reasons["이미지 누락"] += 1
        continue
    if doc_id in records_by_id:
        rejection_reasons["중복 doc_id"] += 1
        continue

    normalized = dict(record)
    normalized["doc_id"] = doc_id
    normalized["_absolute_image_path"] = str(image_path)
    records_by_id[doc_id] = normalized

records = list(records_by_id.values())
if INDEX_LIMIT is not None:
    records = records[: int(INDEX_LIMIT)]

if not records:
    raise RuntimeError("임베딩할 수 있는 정상 레코드가 없습니다.")

print(f"입력 행 수          : {len(raw_records):,}")
print(f"정상 고유 레코드 수 : {len(records):,}")
print("제외 사유           :", dict(rejection_reasons))
print(
    "첫 레코드          :",
    {key: records[0].get(key) for key in ("doc_id", "title", "image_path")},
)
"""
)


markdown("### 4. 기존 Qdrant DB 복원 및 증분 대상 결정")


code(
    r"""
import shutil
import zipfile

from qdrant_client import QdrantClient


def read_json_if_exists(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def restore_qdrant_archive(archive_path: Path, destination: Path):
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(destination)
        print("기존 Qdrant DB 복원 완료:", archive_path)
    else:
        print("기존 Qdrant DB가 없어 새로 구축합니다.")


stored_state = read_json_if_exists(INDEX_STATE_PATH)
if REBUILD_INDEX:
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    stored_state = None
    print("REBUILD_INDEX=True: 새 Qdrant DB를 만듭니다.")
else:
    if QDRANT_ARCHIVE.exists() != (stored_state is not None):
        raise RuntimeError(
            "Qdrant ZIP과 상태 JSON 중 하나만 존재합니다. "
            "두 파일을 함께 복구하거나 REBUILD_INDEX=True로 다시 구축하세요."
        )
    restore_qdrant_archive(QDRANT_ARCHIVE, WORK_DIR)

if stored_state is not None:
    expected_models = {
        "text_model": TEXT_MODEL_NAME,
        "image_model": IMAGE_MODEL_NAME,
        "collection_name": COLLECTION_NAME,
    }
    mismatches = {
        key: (stored_state.get(key), expected)
        for key, expected in expected_models.items()
        if stored_state.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            "기존 DB와 현재 모델 설정이 다릅니다. "
            f"{mismatches}. 모델을 바꾸려면 REBUILD_INDEX=True로 실행하세요."
        )

client = QdrantClient(path=str(WORK_DIR))
collection_exists = client.collection_exists(COLLECTION_NAME)


def get_existing_doc_ids(qdrant_client, collection_name):
    if not qdrant_client.collection_exists(collection_name):
        return set()

    doc_ids = set()
    offset = None
    while True:
        points, offset = qdrant_client.scroll(
            collection_name=collection_name,
            limit=1000,
            offset=offset,
            with_payload=["doc_id"],
            with_vectors=False,
        )
        for point in points:
            if point.payload and point.payload.get("doc_id") is not None:
                doc_ids.add(str(point.payload["doc_id"]))
        if offset is None:
            break
    return doc_ids


existing_doc_ids = get_existing_doc_ids(client, COLLECTION_NAME)
records_to_index = [
    record
    for record in records
    if record["doc_id"] not in existing_doc_ids
    or record["doc_id"] in FORCE_REINDEX_DOC_IDS
]

print(f"Qdrant 기존 문서 수 : {len(existing_doc_ids):,}")
print(f"이번 임베딩 대상 수 : {len(records_to_index):,}")
print(f"건너뛴 기존 문서 수 : {len(records) - len(records_to_index):,}")
"""
)


markdown("### 5. BGE-M3와 SigLIP2 로드")


code(
    r"""
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoProcessor


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE != "cuda":
    raise RuntimeError(
        "GPU 런타임이 아닙니다. Colab 런타임 유형을 GPU로 변경한 뒤 다시 실행하세요."
    )

print("GPU:", torch.cuda.get_device_name(0))

text_encoder = SentenceTransformer(TEXT_MODEL_NAME, device=DEVICE)
text_encoder.max_seq_length = TEXT_MAX_LENGTH

image_processor = AutoProcessor.from_pretrained(IMAGE_MODEL_NAME)
image_encoder = AutoModel.from_pretrained(
    IMAGE_MODEL_NAME,
    torch_dtype=torch.float16,
).to(DEVICE)
image_encoder.eval()

print("BGE-M3 차원:", text_encoder.get_sentence_embedding_dimension())
print("모델 로드 완료")
"""
)


markdown("### 6. 새 레코드의 텍스트·이미지 임베딩 생성")


code(
    r"""
import numpy as np
from PIL import Image, ImageOps
from tqdm.auto import tqdm


def normalize_search_terms(value):
    if isinstance(value, list):
        terms = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(terms)
    return str(value or "").strip()


def make_document_text(record):
    parts = []
    title = str(record.get("title", "")).strip()
    search_terms = normalize_search_terms(record.get("search_terms", []))
    description = str(record.get("description", "")).strip()
    if title:
        parts.append(f"제목: {title}")
    if search_terms:
        parts.append(f"검색어: {search_terms}")
    parts.append(f"설명: {description}")
    return "\n".join(parts)


def ensure_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "pooler_output") and value.pooler_output is not None:
        return value.pooler_output
    if isinstance(value, (tuple, list)) and value:
        return value[0]
    raise TypeError(f"지원하지 않는 임베딩 반환 형식: {type(value)}")


def open_rgb_image(path):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        return image.copy()


def encode_images(paths, batch_size):
    chunks = []
    for start in tqdm(
        range(0, len(paths), batch_size),
        desc="SigLIP2 이미지 임베딩",
    ):
        batch_paths = paths[start : start + batch_size]
        try:
            images = [open_rgb_image(path) for path in batch_paths]
        except Exception as exc:
            raise RuntimeError(
                f"이미지 로드 실패: {batch_paths}. 이미지 다운로드 노트북에서 재검증하세요."
            ) from exc

        inputs = image_processor(images=images, return_tensors="pt")
        inputs = {
            key: value.to(DEVICE)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            features = ensure_tensor(image_encoder.get_image_features(**inputs))
            features = torch.nn.functional.normalize(features.float(), p=2, dim=-1)
        chunks.append(features.cpu().numpy())

    if not chunks:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


if records_to_index:
    document_texts = [make_document_text(record) for record in records_to_index]
    text_vectors = text_encoder.encode(
        document_texts,
        batch_size=TEXT_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)

    image_paths = [record["_absolute_image_path"] for record in records_to_index]
    image_vectors = encode_images(image_paths, IMAGE_BATCH_SIZE)

    if len(text_vectors) != len(records_to_index):
        raise RuntimeError("텍스트 임베딩 수와 레코드 수가 다릅니다.")
    if len(image_vectors) != len(records_to_index):
        raise RuntimeError("이미지 임베딩 수와 레코드 수가 다릅니다.")

    print("텍스트 임베딩:", text_vectors.shape)
    print("이미지 임베딩:", image_vectors.shape)
else:
    text_vectors = np.empty((0, 0), dtype=np.float32)
    image_vectors = np.empty((0, 0), dtype=np.float32)
    print("새 레코드가 없어 임베딩 생성을 건너뜁니다.")
"""
)


markdown("### 7. Qdrant named vectors 생성 및 증분 적재")


code(
    r"""
import uuid

from qdrant_client.models import Distance, PointStruct, VectorParams


def point_id_for_doc(doc_id):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"tourapi:{doc_id}"))


def payload_for_record(record):
    return {
        "doc_id": record["doc_id"],
        "source": str(record.get("source", "tourapi")),
        "title": str(record.get("title", "")),
        "search_terms": record.get("search_terms", []),
        "description": str(record.get("description", "")),
        "image_path": str(record.get("image_path", "")),
        "image_url": str(record.get("image_url", "")),
        "thumbnail_url": str(record.get("thumbnail_url", "")),
        "content_type_id": str(record.get("content_type_id", "")),
        "category_codes": record.get("category_codes", []),
        "copyright_type": str(record.get("copyright_type", "")),
    }


if not collection_exists:
    if not records_to_index:
        raise RuntimeError("새 컬렉션을 만들 임베딩이 없습니다.")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            TEXT_VECTOR_NAME: VectorParams(
                size=int(text_vectors.shape[1]),
                distance=Distance.COSINE,
            ),
            IMAGE_VECTOR_NAME: VectorParams(
                size=int(image_vectors.shape[1]),
                distance=Distance.COSINE,
            ),
        },
    )
    collection_exists = True
    print("Qdrant 컬렉션 생성 완료:", COLLECTION_NAME)

if records_to_index:
    for start in tqdm(
        range(0, len(records_to_index), UPSERT_BATCH_SIZE),
        desc="Qdrant 적재",
    ):
        end = min(start + UPSERT_BATCH_SIZE, len(records_to_index))
        points = []
        for index in range(start, end):
            record = records_to_index[index]
            points.append(
                PointStruct(
                    id=point_id_for_doc(record["doc_id"]),
                    vector={
                        TEXT_VECTOR_NAME: text_vectors[index].tolist(),
                        IMAGE_VECTOR_NAME: image_vectors[index].tolist(),
                    },
                    payload=payload_for_record(record),
                )
            )
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

collection_info = client.get_collection(COLLECTION_NAME)
indexed_doc_ids = get_existing_doc_ids(client, COLLECTION_NAME)

print(f"Qdrant 최종 points_count : {collection_info.points_count:,}")
print(f"Qdrant 고유 doc_id 수   : {len(indexed_doc_ids):,}")
if len(indexed_doc_ids) < len(records):
    print(f"주의: 입력 중 {len(records) - len(indexed_doc_ids):,}건이 아직 미적재 상태입니다.")
"""
)


markdown("### 8. 텍스트·이미지·결합 검색 함수")


code(
    r"""
import pandas as pd


def points_to_rows(points, retrieval_channel):
    rows = []
    for rank, point in enumerate(points, start=1):
        payload = dict(point.payload or {})
        rows.append(
            {
                "rank": rank,
                "score": float(point.score),
                "retrieval_channel": retrieval_channel,
                **payload,
            }
        )
    return rows


def query_qdrant(vector, vector_name, top_k):
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=np.asarray(vector, dtype=np.float32).tolist(),
        using=vector_name,
        limit=int(top_k),
        with_payload=True,
        with_vectors=False,
    )
    return response.points


def encode_text_query(query):
    return text_encoder.encode(
        [str(query)],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]


def encode_siglip_text_query(query):
    inputs = image_processor(
        text=[str(query)],
        padding="max_length",
        max_length=64,
        truncation=True,
        return_tensors="pt",
    )
    inputs = {
        key: value.to(DEVICE)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in inputs.items()
    }
    with torch.inference_mode():
        features = ensure_tensor(image_encoder.get_text_features(**inputs))
        features = torch.nn.functional.normalize(features.float(), p=2, dim=-1)
    return features[0].cpu().numpy()


def encode_image_query(image_path):
    return encode_images([str(image_path)], batch_size=1)[0]


def search_text(query, top_k=5):
    points = query_qdrant(
        encode_text_query(query),
        TEXT_VECTOR_NAME,
        top_k,
    )
    return pd.DataFrame(points_to_rows(points, "bge_text"))


def search_image(image_path, top_k=5):
    points = query_qdrant(
        encode_image_query(image_path),
        IMAGE_VECTOR_NAME,
        top_k,
    )
    return pd.DataFrame(points_to_rows(points, "siglip_image"))


def search_image_by_text(query, top_k=5):
    points = query_qdrant(
        encode_siglip_text_query(query),
        IMAGE_VECTOR_NAME,
        top_k,
    )
    return pd.DataFrame(points_to_rows(points, "siglip_text_to_image"))


def hybrid_search(query_text=None, query_image_path=None, top_k=5, candidate_k=30):
    # BGE 텍스트, SigLIP 텍스트→이미지, 이미지→이미지 순위를 RRF로 결합합니다.
    ranked_lists = []

    if query_text and str(query_text).strip():
        ranked_lists.append(
            points_to_rows(
                query_qdrant(
                    encode_text_query(query_text),
                    TEXT_VECTOR_NAME,
                    candidate_k,
                ),
                "bge_text",
            )
        )
        ranked_lists.append(
            points_to_rows(
                query_qdrant(
                    encode_siglip_text_query(query_text),
                    IMAGE_VECTOR_NAME,
                    candidate_k,
                ),
                "siglip_text_to_image",
            )
        )

    if query_image_path is not None:
        ranked_lists.append(
            points_to_rows(
                query_qdrant(
                    encode_image_query(query_image_path),
                    IMAGE_VECTOR_NAME,
                    candidate_k,
                ),
                "siglip_image",
            )
        )

    if not ranked_lists:
        raise ValueError("query_text 또는 query_image_path 중 하나는 입력해야 합니다.")

    merged = {}
    rrf_k = 60
    for ranked_rows in ranked_lists:
        for rank, row in enumerate(ranked_rows, start=1):
            doc_id = str(row["doc_id"])
            if doc_id not in merged:
                merged[doc_id] = {
                    key: value
                    for key, value in row.items()
                    if key not in {"rank", "score", "retrieval_channel"}
                }
                merged[doc_id]["rrf_score"] = 0.0
                merged[doc_id]["matched_channels"] = []
            merged[doc_id]["rrf_score"] += 1.0 / (rrf_k + rank)
            merged[doc_id]["matched_channels"].append(row["retrieval_channel"])

    result = sorted(
        merged.values(),
        key=lambda row: row["rrf_score"],
        reverse=True,
    )[: int(top_k)]
    for rank, row in enumerate(result, start=1):
        row["rank"] = rank

    columns = [
        "rank",
        "rrf_score",
        "matched_channels",
        "doc_id",
        "title",
        "description",
        "image_path",
        "image_url",
        "source",
    ]
    return pd.DataFrame(result).reindex(columns=columns)


def build_rag_context(search_results, max_docs=5):
    # 검색 DataFrame을 Kanana 프롬프트에 넣을 근거 텍스트로 변환합니다.
    if not isinstance(search_results, pd.DataFrame):
        search_results = pd.DataFrame(search_results)
    blocks = []
    for rank, (_, row) in enumerate(search_results.head(max_docs).iterrows(), start=1):
        blocks.append(
            "\n".join(
                [
                    f"[근거 {rank}]",
                    f"제목: {row.get('title', '')}",
                    f"설명: {row.get('description', '')}",
                    f"출처: {row.get('source', '')}",
                    f"이미지: {row.get('image_path', '')}",
                ]
            )
        )
    return "\n\n".join(blocks)


print(
    "검색 함수 준비 완료: search_text / search_image / "
    "search_image_by_text / hybrid_search / build_rag_context"
)
"""
)


markdown("### 9. 검색 스모크 테스트")


code(
    r"""
from IPython.display import display


sample = records[0]
sample_query = str(sample.get("title", "한국 관광"))
sample_image_path = sample["_absolute_image_path"]

print("텍스트 검색 질의:", sample_query)
display(
    search_text(sample_query, top_k=5)[
        ["rank", "score", "doc_id", "title", "description"]
    ]
)

print("\n동일 이미지 유사도 검색:", sample_image_path)
display(
    search_image(sample_image_path, top_k=5)[
        ["rank", "score", "doc_id", "title", "image_path"]
    ]
)

print("\n텍스트+이미지 결합 검색:")
sample_hybrid_results = hybrid_search(
    query_text=sample_query,
    query_image_path=sample_image_path,
    top_k=5,
)
display(sample_hybrid_results)

print("\nKanana에 전달할 RAG 근거 예시:\n")
print(build_rag_context(sample_hybrid_results, max_docs=3))
"""
)


markdown("### 10. Qdrant DB를 Google Drive에 저장")


code(
    r"""
from datetime import datetime


def atomic_write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def save_qdrant_archive(source_dir: Path, destination: Path):
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=str(path.relative_to(source_dir)))
    temporary.replace(destination)


state = {
    "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "collection_name": COLLECTION_NAME,
    "text_model": TEXT_MODEL_NAME,
    "image_model": IMAGE_MODEL_NAME,
    "text_vector_name": TEXT_VECTOR_NAME,
    "image_vector_name": IMAGE_VECTOR_NAME,
    "text_vector_size": int(text_encoder.get_sentence_embedding_dimension()),
    "image_vector_size": int(
        image_vectors.shape[1]
        if image_vectors.size
        else stored_state["image_vector_size"]
    ),
    "input_record_count": len(records),
    "indexed_doc_id_count": len(indexed_doc_ids),
    "new_or_updated_count": len(records_to_index),
}

client.close()
save_qdrant_archive(WORK_DIR, QDRANT_ARCHIVE)
atomic_write_json(INDEX_STATE_PATH, state)

summary = {
    **state,
    "qdrant_archive": str(QDRANT_ARCHIVE),
    "qdrant_archive_mib": round(QDRANT_ARCHIVE.stat().st_size / (1024 ** 2), 2),
    "input_jsonl": str(INPUT_JSONL),
}
atomic_write_json(BUILD_SUMMARY_PATH, summary)

# 저장 후에도 아래에서 검색 함수를 계속 사용할 수 있도록 DB를 다시 엽니다.
client = QdrantClient(path=str(WORK_DIR))

print(json.dumps(summary, ensure_ascii=False, indent=2))
print("Qdrant DB 저장 완료. 다음 실행에서는 새 doc_id만 추가됩니다.")
"""
)


markdown(
    """
### 다음 수집분을 추가하는 방법

1. 이미지 다운로드 노트북에서 최신 누적 JSONL을 업로드하고 새 이미지만 다운로드합니다.
2. 이 노트북을 처음 셀부터 다시 실행합니다.
3. Drive의 기존 Qdrant ZIP이 자동 복원됩니다.
4. 기존 `doc_id`는 건너뛰고 새 데이터만 BGE-M3·SigLIP2 임베딩 후 upsert합니다.
5. 마지막 저장 셀까지 실행해 갱신된 Qdrant ZIP과 상태 JSON을 Drive에 보관합니다.

현재 단계의 결과물은 **검색 DB**입니다. 이후 Kanana 질의응답 파이프라인에서는 `hybrid_search()`의 상위 문서 설명과 이미지 경로를 프롬프트 컨텍스트로 전달하면 됩니다.
"""
)


OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUTPUT_PATH)
print(f"작성 완료: {OUTPUT_PATH}")
