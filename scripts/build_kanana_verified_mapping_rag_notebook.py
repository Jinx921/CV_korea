from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
BASE_NOTEBOOK = ROOT / "notebooks" / "kanana15v_3b_lora_tourapi_rag_typed.ipynb"
OUTPUT_PATH = ROOT / "notebooks" / "kanana15v_3b_lora_verified_mapping_rag_typed.ipynb"


nb = nbf.read(BASE_NOTEBOOK, as_version=4)
nb.metadata["colab"]["name"] = OUTPUT_PATH.name

nb.cells[0].source = """
## Kanana-1.5-V 3B 유형별 LoRA + 검증 매핑 RAG

44점대 Kanana Shared→MC·SA·LA 유형별 LoRA 조건을 유지하고, 사람이 직접 승인한
대회 이미지↔TourAPI 문서 매핑에만 RAG 설명을 적용하는 비교 실험입니다.

- 최종 매핑 DB: `rag_mapping.jsonl` 2,000건
- RAG 적용: 승인된 train 3건, validation 2건, test 7건만
- 미매칭 문항: 기존 이미지+질문 프롬프트 그대로 처리
- 검색 재실행 없음: BGE·DINO·CLIP·Qdrant 모델을 다시 로드하지 않음
- OCR: 이번 RAG 단독 효과 비교에서는 적용하지 않음

첫 실험은 **RAG만 변경한 A/B 비교**입니다. 결과 확인 후 별도 실험에서 선택적
PaddleOCR crop을 추가해야 RAG와 OCR 각각의 영향을 해석할 수 있습니다.

Colab에서 `런타임 → 런타임 유형 변경 → GPU`를 선택한 뒤 위에서부터 실행하세요.
L4(24GB) 이상을 권장합니다.
""".strip()

nb.cells[1].source = "### 0. Colab 런타임 준비"
nb.cells[2].source = r'''
import os
import signal
import subprocess
import sys
from pathlib import Path


restart_marker = Path("/content/.kanana15v_verified_mapping_rag_env_v1")

if not restart_marker.exists():
    def run_pip(*args):
        subprocess.check_call([sys.executable, "-m", "pip", *args])

    run_pip("uninstall", "-q", "-y", "gradio", "gradio_client", "torchao", "torchaudio", "Pillow")
    run_pip(
        "install", "-q", "-U",
        "transformers>=4.57.0,<5",
        "huggingface-hub==0.36.2",
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
'''.strip()

nb.cells[5].source = r'''
import copy
import gc
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


# Google Drive 경로
DRIVE_PROJECT_DIR = Path("/content/drive/MyDrive/CV_korea")
DRIVE_ZIP_DIR = DRIVE_PROJECT_DIR / "data"
RAG_DB_DIR = DRIVE_PROJECT_DIR / "tourapi_image_db"
MAPPING_DB_PATH = (
    RAG_DB_DIR / "mapping" / "combined" / "final" / "rag_mapping.jsonl"
)

MODEL_ID = "kakaocorp/kanana-1.5-v-3b-instruct"
# 현재 Transformers의 Qwen2-VL position_embeddings 호출 방식과 호환되는 revision
KANANA_REVISION = "2e00ef13ccec2e99459a8eada18a1bfd05bff44b"

RUN_NAME = "kanana15v_3b_lora_verified_mapping_rag_typed"
TRAINING_METHOD = "shared_then_typed_lora_with_manual_verified_mapping_rag"

# True: Shared + 유형별 Adapter 학습 / False: 같은 RUN_NAME에 저장된 Adapter로 추론만 수행
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

# 승인된 문서 하나만 사용하며 너무 긴 설명은 잘라냅니다.
RAG_TOP_K = 1
RAG_DOC_CHAR_LIMIT = 900
RESET_PREDICTION_CACHE = False
SAVE_EVERY_N_PREDICTIONS = 20
SEED = 42
REFERENCE_BASELINE_SCORE = 44.8395439
OCR_MODE = "disabled_for_rag_ablation"

COLAB_WORK_DIR = Path("/content/kculture_kanana15v_verified_mapping_rag")
EXTRACT_DIR = COLAB_WORK_DIR / "extracted"
CHECKPOINT_ROOT = DRIVE_PROJECT_DIR / "checkpoints" / RUN_NAME
RUN_OUTPUT_DIR = DRIVE_PROJECT_DIR / "outputs" / RUN_NAME
SHARED_ADAPTER_DIR = RUN_OUTPUT_DIR / "adapter_shared"
FORM_ADAPTER_DIRS = {
    form: RUN_OUTPUT_DIR / f"adapter_{form.lower()}" for form in ["MC", "SA", "LA"]
}

for path in [COLAB_WORK_DIR, EXTRACT_DIR, CHECKPOINT_ROOT, RUN_OUTPUT_DIR]:
    path.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("대회 ZIP 폴더:", DRIVE_ZIP_DIR)
print("최종 매핑 DB:", MAPPING_DB_PATH)
print("결과 폴더:", RUN_OUTPUT_DIR)
print("모델:", MODEL_ID)
print("RAG: 사람이 승인한 동일 대상 문서 1개만 선택 적용")
print("OCR:", OCR_MODE)
print("학습: Shared", SHARED_EPOCHS, "epoch + 유형별", TYPED_EPOCHS, "epoch")
'''.strip()

nb.cells[11].source = "### 4. 최종 이미지↔TourAPI 매핑 DB 검증"
nb.cells[12].source = r'''
def atomic_write_json(data, path):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"{path}:{line_number}가 JSON 객체가 아닙니다.")
            rows.append(value)
    return rows


if not MAPPING_DB_PATH.is_file():
    raise FileNotFoundError(
        "최종 매핑 DB가 없습니다. 다음 위치에 rag_mapping.jsonl을 업로드하세요:\n"
        + str(MAPPING_DB_PATH)
    )

mapping_rows = read_jsonl(MAPPING_DB_PATH)
if len(mapping_rows) != 2000:
    raise RuntimeError(f"매핑 DB가 2,000건이 아닙니다: {len(mapping_rows)}")

mapping_by_key = {}
for row in mapping_rows:
    key = str(row.get("image_key", ""))
    if not key or key in mapping_by_key:
        raise RuntimeError(f"매핑 DB image_key가 비어 있거나 중복됩니다: {key}")
    mapping_by_key[key] = row

dataset_keys = {
    f"{split}:{record['model_input']['image_name']}"
    for split in ["train", "validation", "test"]
    for record in records[split]
}
if dataset_keys != set(mapping_by_key):
    missing = sorted(dataset_keys - set(mapping_by_key))
    extra = sorted(set(mapping_by_key) - dataset_keys)
    raise RuntimeError(f"대회 데이터와 매핑 키가 다릅니다. missing={missing[:5]}, extra={extra[:5]}")

matched_rows = [row for row in mapping_rows if row.get("mapping_status") == "matched"]
unmatched_rows = [row for row in mapping_rows if row.get("mapping_status") == "unmatched"]
if len(matched_rows) + len(unmatched_rows) != len(mapping_rows):
    raise RuntimeError("matched/unmatched 이외의 mapping_status가 있습니다.")
if any(not row.get("doc_id") or not row.get("description") for row in matched_rows):
    raise RuntimeError("matched 행의 doc_id 또는 description이 비어 있습니다.")
if any(row.get("doc_id") is not None for row in unmatched_rows):
    raise RuntimeError("unmatched 행에 doc_id가 남아 있습니다.")

mapping_status_rows = []
for split in ["train", "validation", "test"]:
    split_rows = [row for row in mapping_rows if row["split"] == split]
    mapping_status_rows.append({
        "split": split,
        "rows": len(split_rows),
        "rag_matched": sum(row["mapping_status"] == "matched" for row in split_rows),
        "unmatched": sum(row["mapping_status"] == "unmatched" for row in split_rows),
    })

display(pd.DataFrame(mapping_status_rows))
print("최종 매핑 DB 검증 완료")
print("matched:", len(matched_rows), "/", len(mapping_rows))
'''.strip()

nb.cells[13].source = "### 5. Train·Validation·Test 직접 매핑 RAG 구성"
nb.cells[14].source = r'''
def mapping_key_for(record, split):
    return f"{split}:{record['model_input']['image_name']}"


def rag_item_for(record, split):
    key = mapping_key_for(record, split)
    mapping = mapping_by_key[key]
    if mapping["mapping_status"] != "matched":
        return {
            "rag_applied": False,
            "mapping_status": "unmatched",
            "documents": [],
        }
    document = {
        "doc_id": str(mapping["doc_id"]),
        "title": str(mapping.get("title", "")),
        "description": str(mapping.get("description", "")),
        "image_path": str(mapping.get("rag_image_path", "")),
        "image_url": str(mapping.get("rag_image_url", "")),
        "source": str(mapping.get("source", "tourapi")),
        "confidence": str(mapping.get("confidence", "verified")),
    }
    return {
        "rag_applied": True,
        "mapping_status": "matched",
        "documents": [document],
    }


def rag_context_for(record, split):
    item = rag_item_for(record, split)
    if not item["rag_applied"]:
        return ""
    document = item["documents"][0]
    description = document["description"].strip()[:RAG_DOC_CHAR_LIMIT]
    return "\n".join([
        "[검증된 참고자료]",
        f"제목: {document['title']}",
        f"설명: {description}",
        f"출처: {document['source']}",
    ])


rag_summary_rows = []
for split in ["train", "validation", "test"]:
    split_items = [rag_item_for(record, split) for record in records[split]]
    rag_summary_rows.append({
        "split": split,
        "rows": len(split_items),
        "rag_applied": sum(item["rag_applied"] for item in split_items),
        "rag_not_applied": sum(not item["rag_applied"] for item in split_items),
    })

display(pd.DataFrame(rag_summary_rows))
print("임베딩·검색 없이 최종 매핑 DB를 직접 사용합니다.")
'''.strip()

nb.cells[15].source = "### 6. 검증된 RAG 근거를 포함한 유형별 프롬프트"
nb.cells[16].source = r'''
SYSTEM_PROMPT = (
    "당신은 이미지와 검증된 참고자료를 함께 이해하는 한국문화 질의응답 모델입니다. "
    "이미지와 질문을 먼저 확인하고, 제공된 참고자료가 답에 필요한 경우 활용하세요. "
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
            "다음 자료는 사진과 같은 대상으로 사람이 확인한 참고자료입니다. "
            "질문에 필요한 정보만 사용하세요.\n\n" + rag_context
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
    form_records = [
        row for row in records["train"]
        if row["metadata"]["question_form"] == form
    ]
    example = next(
        (row for row in form_records if rag_item_for(row, "train")["rag_applied"]),
        form_records[0],
    )
    print("=" * 80)
    print(form, example["metadata"]["question_id"])
    print("RAG 적용:", rag_item_for(example, "train")["rag_applied"])
    print(build_user_text(example, "train"))
    print("정답:", example["model_output"]["answer"])
'''.strip()

nb.cells[21].source = "### 모델 입출력 스모크 테스트"
nb.cells[22].source = r'''
smoke_record = next(
    record for record in records["validation"]
    if rag_item_for(record, "validation")["rag_applied"]
)
smoke_rag = rag_item_for(smoke_record, "validation")

print("ID:", smoke_record["metadata"]["question_id"])
print("질문:", smoke_record["model_input"]["question"])
print("정답:", smoke_record["model_output"]["answer"])
print("RAG 적용:", smoke_rag["rag_applied"])
print("RAG 문서:", [doc["title"] for doc in smoke_rag["documents"]])

smoke_result = predict_one(smoke_record, "validation")
print("예측:", smoke_result["answer"])
print("원본 출력:", smoke_result["raw"])
'''.strip()

nb.cells[29].source = """
### 11. Validation 직접 매핑 RAG 진단

유형과 RAG 적용 여부별 exact match를 확인합니다. 매칭 validation은 2건뿐이므로
개별 예측도 함께 확인하고, 최종 판단은 리더보드 점수로 진행합니다.
""".strip()
nb.cells[30].source = r'''
def comparable_answer(text, form):
    if form == "MC":
        normalized, valid = normalize_mc(text)
        return normalized if valid else str(text).strip()
    return re.sub(r"\s+", " ", str(text)).strip()


validation_subset = records["validation"][:VALIDATION_MAX_SAMPLES]
validation_rows = []

for record in tqdm(validation_subset, desc="validation 검증 매핑 RAG 추론"):
    result = predict_one(record, "validation")
    form = record["metadata"]["question_form"]
    target = record["model_output"]["answer"]
    rag_item = rag_item_for(record, "validation")
    validation_rows.append({
        "question_id": record["metadata"]["question_id"],
        "question_form": form,
        "rag_applied": rag_item["rag_applied"],
        "rag_doc_ids": "/".join(doc["doc_id"] for doc in rag_item["documents"]),
        "rag_titles": " / ".join(doc["title"] for doc in rag_item["documents"]),
        "prediction": result["answer"],
        "target": target,
        "exact_match": comparable_answer(result["answer"], form)
        == comparable_answer(target, form),
        "prediction_length": len(result["answer"]),
        "retried": result["retried"],
        "raw": result["raw"],
    })

validation_df = pd.DataFrame(validation_rows)
display(validation_df.groupby(["question_form", "rag_applied"])["exact_match"].agg(["count", "mean"]))
display(validation_df[validation_df["rag_applied"]])

validation_path = RUN_OUTPUT_DIR / "validation_predictions.csv"
validation_df.to_csv(validation_path, index=False, encoding="utf-8-sig")
print("validation 결과 저장:", validation_path)
'''.strip()

nb.cells[31].source = """
### 12. Test 800건 검증 매핑 RAG 추론

test 7건에만 승인된 TourAPI 설명이 들어가며 나머지 793건은 기존 프롬프트를 사용합니다.
모델, Adapter, 매핑 DB 또는 프롬프트를 변경하면 `RUN_NAME`을 바꾸거나
`RESET_PREDICTION_CACHE=True`로 한 번 실행하세요.
""".strip()
nb.cells[32].source = nb.cells[32].source.replace(
    "test TourAPI RAG 추론", "test 검증 매핑 RAG 추론"
)

nb.cells[33].source = "### 13. 제출 JSON·ZIP 생성 및 최종 검증"
nb.cells[34].source = (
    nb.cells[34].source
    .replace(
        "submission_kanana15v_3b_lora_tourapi_rag_typed.json",
        "submission_kanana15v_3b_lora_verified_mapping_rag_typed.json",
    )
    .replace(
        "submission_kanana15v_3b_lora_tourapi_rag_typed.zip",
        "submission_kanana15v_3b_lora_verified_mapping_rag_typed.zip",
    )
)

nb.cells[35].source = "### 14. 검증 매핑 RAG 실험 요약 저장"
nb.cells[36].source = r'''
rag_experiment_summary = {
    "run_name": RUN_NAME,
    "model_id": MODEL_ID,
    "reference_baseline_score": REFERENCE_BASELINE_SCORE,
    "mapping_db": str(MAPPING_DB_PATH),
    "mapping_record_count": len(mapping_rows),
    "mapping_matched_count": len(matched_rows),
    "retrieval_summary": rag_summary_rows,
    "train_method": TRAINING_METHOD,
    "shared_epochs": SHARED_EPOCHS,
    "typed_epochs": TYPED_EPOCHS,
    "ocr_mode": OCR_MODE,
    "submission_json": str(submission_json_path),
    "submission_zip": str(submission_zip_path),
    "validation_csv": str(validation_path),
}

rag_experiment_summary_path = RUN_OUTPUT_DIR / "rag_experiment_summary.json"
atomic_write_json(rag_experiment_summary, rag_experiment_summary_path)
print(json.dumps(rag_experiment_summary, ensure_ascii=False, indent=2))
print("RAG 실험 요약:", rag_experiment_summary_path)
'''.strip()

nb.cells[37].source = """
### 실행 결과 확인 순서

1. 매핑 DB가 전체 2,000건, matched 12건인지 확인
2. 적용 수가 train=3, validation=2, test=7인지 확인
3. 학습 전 loss와 LoRA gradient 스모크 테스트 통과 확인
4. Shared 1 epoch + MC·SA·LA 유형별 1 epoch 완료 확인
5. validation에서 RAG 적용 2건의 개별 예측 확인
6. `submission_kanana15v_3b_lora_verified_mapping_rag_typed.json` 제출
7. 44.8395 기준 점수와 비교

이번 노트북에는 OCR이 적용되지 않습니다. RAG 단독 결과가 확인된 다음 동일 조건에서
선택적 PaddleOCR crop만 추가해야 두 변화의 효과를 분리해서 비교할 수 있습니다.
""".strip()

for cell in nb.cells:
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUTPUT_PATH)
print(OUTPUT_PATH)
