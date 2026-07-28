from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "notebooks" / "internvl35_8b_baseline.ipynb"


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "colab": {
        "name": OUTPUT_PATH.name,
        "provenance": [],
        "gpuType": "A100",
    },
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.x"},
    "accelerator": "GPU",
}


def markdown(source: str) -> None:
    nb.cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source: str) -> None:
    nb.cells.append(nbf.v4.new_code_cell(source.strip()))


markdown(
    """
# InternVL3.5-8B 한국문화 멀티모달 QA 베이스라인

이 노트북은 Google Colab에서 다음 작업을 한 번에 수행합니다.

1. Google Drive의 **CV_korea/data** 안에 있는 ZIP 4개를 Colab 로컬 디스크로 해제
2. train / validation / test JSON과 이미지 경로 검증
3. **OpenGVLab/InternVL3_5-8B-HF**를 4-bit로 로드
4. 선택적으로 train 1,000건에 QLoRA 미세조정
5. validation 진단 및 test 800건 추론
6. 원본 test 구조를 유지한 제출 JSON과 ZIP을 Drive에 저장

권장 런타임은 **A100 40GB** 또는 **L4 24GB**입니다. T4 16GB에서는 이미지 크기나 최대 길이를 더 줄여야 할 수 있습니다.

팀 모델 간 순수 제로샷 성능만 비교할 때는 설정 셀의 DO_TRAIN을 False로 바꾸세요. 현재 기본값은 요청에 맞춰 **Q-LoRA 학습을 수행하도록 True**입니다.
"""
)

markdown(
    """
## 0. Colab 런타임 준비

Colab 메뉴에서 **런타임 → 런타임 유형 변경 → GPU**를 선택한 뒤 아래 셀부터 순서대로 실행합니다.

모델은 공개 체크포인트이므로 Hugging Face 토큰은 필요하지 않습니다. 첫 패키지 설치가 끝나면 Pillow 버전을 일치시키기 위해 런타임이 자동으로 한 번 재시작됩니다. 다시 연결된 뒤 이 노트북을 첫 셀부터 실행하세요.
"""
)

code(
    r"""
# Colab 기본 Gradio와 구버전 torchao는 현재 Transformers/PEFT 조합과
# 의존성 충돌을 일으킬 수 있으며 이 노트북에서는 둘 다 사용하지 않습니다.
%pip uninstall -q -y gradio gradio_client torchao

%pip install -q -U \
  "transformers>=4.57.0,<5" \
  "huggingface-hub==0.36.2" \
  "accelerate>=1.2" \
  "bitsandbytes>=0.45" \
  "peft>=0.14" \
  "datasets>=3.2" \
  "sentencepiece" \
  "safetensors" \
  "pandas>=2.2" \
  "tqdm>=4.66"

# Colab 기본 Pillow와 업그레이드된 Pillow 파일이 섞이는 현상을 방지합니다.
%pip install -q --no-cache-dir --force-reinstall "Pillow==12.3.0"

import os
from pathlib import Path

restart_marker = Path("/content/.internvl35_pillow_12_3_ready")
if not restart_marker.exists():
    restart_marker.write_text("ready", encoding="utf-8")
    print("패키지 설치 완료. Pillow를 깨끗하게 다시 불러오기 위해 런타임을 재시작합니다.")
    print("재연결된 뒤 이 노트북을 첫 셀부터 다시 실행하세요.")
    os.kill(os.getpid(), 9)

from PIL import Image, ImageOps
import PIL

assert PIL.__version__ == "12.3.0", PIL.__version__
print("Pillow 정상 로드:", PIL.__version__)
"""
)

markdown("## 1. Drive 연결 및 실행 설정")

code(
    r"""
from google.colab import drive
drive.mount("/content/drive")
"""
)

code(
    r"""
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


# ------------------------- 사용자가 주로 바꿀 설정 -------------------------
DRIVE_PROJECT_DIR = Path("/content/drive/MyDrive/CV_korea")
DRIVE_ZIP_DIR = DRIVE_PROJECT_DIR / "data"

MODEL_ID = "OpenGVLab/InternVL3_5-8B-HF"
RUN_NAME = "internvl35_8b_qlora_v1"

# True: train 1,000건 QLoRA 학습 / False: 제로샷 베이스라인
DO_TRAIN = True

# 이전에 저장한 adapter를 새 Colab 세션에서 불러올 때 True
USE_SAVED_ADAPTER = False

# 빠른 코드 점검 시 32 같은 정수로 바꾸고, 최종 학습은 None
TRAIN_MAX_SAMPLES = None
VALIDATION_MAX_SAMPLES = 30  # 최종 확인은 200 권장

NUM_EPOCHS = 1
LEARNING_RATE = 1e-4
GRADIENT_ACCUMULATION_STEPS = 16
TRAIN_MAX_IMAGE_SIDE = 896
INFERENCE_MAX_IMAGE_SIDE = 1344
# InternVL 기본값은 최대 12개 패치(+thumbnail)입니다. 학습 시에는 메모리와
# 2,048 토큰 길이를 고려해 3개로 제한하고, 추론은 세부 인식을 위해 8개를 사용합니다.
TRAIN_MAX_PATCHES = 3
INFERENCE_MAX_PATCHES = 8
MAX_SEQUENCE_LENGTH = 2048

# 이전 test 예측 캐시를 지우고 처음부터 다시 추론할 때만 True
RESET_PREDICTION_CACHE = False
SAVE_EVERY_N_PREDICTIONS = 20
SEED = 42
# -------------------------------------------------------------------------

COLAB_WORK_DIR = Path("/content/kculture_internvl35")
EXTRACT_DIR = COLAB_WORK_DIR / "extracted"
CHECKPOINT_DIR = DRIVE_PROJECT_DIR / "checkpoints" / RUN_NAME
RUN_OUTPUT_DIR = DRIVE_PROJECT_DIR / "outputs" / RUN_NAME
ADAPTER_DIR = RUN_OUTPUT_DIR / "adapter"

for path in [COLAB_WORK_DIR, EXTRACT_DIR, CHECKPOINT_DIR, RUN_OUTPUT_DIR]:
    path.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("ZIP 폴더:", DRIVE_ZIP_DIR)
print("결과 저장 폴더:", RUN_OUTPUT_DIR)
"""
)

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

if DO_TRAIN and gpu_mem_gb < 20:
    print(
        "주의: 8B 이미지 모델 QLoRA는 16GB GPU에서 OOM이 날 수 있습니다. "
        "발생 시 TRAIN_MAX_PATCHES=2, TRAIN_MAX_IMAGE_SIDE=672로 낮추거나 "
        "DO_TRAIN=False로 제로샷 제출을 먼저 만드세요."
    )
"""
)

markdown("## 2. ZIP 자동 탐색 및 압축 해제")

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
        path
        for path in directory.glob("*.zip")
        if normalized_name(path.name) == expected_nfc
    ]
    if len(matches) != 1:
        found = [path.name for path in directory.glob("*.zip")]
        raise FileNotFoundError(
            f"{expected_name!r}을 찾지 못했습니다. 현재 ZIP 목록: {found}"
        )
    return matches[0]


archive_paths = [resolve_archive(DRIVE_ZIP_DIR, name) for name in EXPECTED_ARCHIVES]
archive_state = {
    path.name: {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
    for path in archive_paths
}
state_path = EXTRACT_DIR / ".archive_state.json"

previous_state = None
if state_path.exists():
    previous_state = json.loads(state_path.read_text(encoding="utf-8"))

if previous_state == archive_state:
    print("동일한 ZIP이 이미 해제되어 있어 압축 해제를 건너뜁니다.")
else:
    print("ZIP 4개를 Colab 로컬 디스크에 해제합니다.")
    for archive_path in archive_paths:
        print("-", archive_path.name)
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(EXTRACT_DIR)
    state_path.write_text(
        json.dumps(archive_state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("압축 해제 완료:", EXTRACT_DIR)
"""
)

markdown("## 3. JSON·이미지 경로 탐색 및 무결성 검사")

code(
    r"""
def find_split_json(split):
    matches = [
        path
        for path in EXTRACT_DIR.rglob("*.json")
        if normalized_name(path.stem).endswith(f"_{split}")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"{split} JSON 후보가 1개가 아닙니다: {matches}")
    return matches[0]


def read_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


json_paths = {split: find_split_json(split) for split in ["train", "validation", "test"]}
records = {split: read_json(path) for split, path in json_paths.items()}


def find_split_image_dir(split, split_records):
    sample_names = [row["model_input"]["image_name"] for row in split_records[:10]]
    scores = Counter()
    for image_name in sample_names:
        for candidate in EXTRACT_DIR.rglob(image_name):
            if candidate.is_file():
                scores[candidate.parent] += 1
    if not scores:
        raise FileNotFoundError(f"{split} 이미지 디렉터리를 찾지 못했습니다.")
    best_dir, score = scores.most_common(1)[0]
    if score < min(3, len(sample_names)):
        raise RuntimeError(f"{split} 이미지 경로 탐색 결과가 불안정합니다: {scores}")
    return best_dir


image_dirs = {
    split: find_split_image_dir(split, split_records)
    for split, split_records in records.items()
}

expected_counts = {"train": 1000, "validation": 200, "test": 800}
summary_rows = []
for split, split_records in records.items():
    image_dir = image_dirs[split]
    missing = [
        row["model_input"]["image_name"]
        for row in split_records
        if not (image_dir / row["model_input"]["image_name"]).is_file()
    ]
    forms = Counter(row["metadata"]["question_form"] for row in split_records)
    assert len(split_records) == expected_counts[split], (split, len(split_records))
    assert not missing, f"{split} 누락 이미지 {len(missing)}개: {missing[:5]}"
    if split == "test":
        assert all("model_output" not in row for row in split_records)
    else:
        assert all(isinstance(row["model_output"]["answer"], str) for row in split_records)
    summary_rows.append(
        {
            "split": split,
            "rows": len(split_records),
            "MC": forms["MC"],
            "SA": forms["SA"],
            "LA": forms["LA"],
            "image_dir": str(image_dir),
            "missing_images": len(missing),
        }
    )

display(pd.DataFrame(summary_rows))
print("JSON 경로")
for split, path in json_paths.items():
    print(f"- {split}: {path}")
"""
)

markdown("## 4. 문항 유형별 프롬프트와 출력 규칙")

code(
    r"""
SYSTEM_PROMPT = (
    "당신은 이미지와 질문을 함께 이해하는 한국문화 질의응답 모델입니다. "
    "이미지에서 확인한 정보와 한국문화 지식을 바탕으로 답하되, "
    "사용자가 요구한 정답 형식만 출력하고 풀이 과정은 출력하지 마세요."
)


def build_user_text(record, retry=False):
    form = record["metadata"]["question_form"]
    model_input = record["model_input"]
    question = model_input["question"].strip()
    options = model_input.get("options") or []

    parts = [question]
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


def build_messages(record, answer=None, add_answer=False, retry=False):
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": build_user_text(record, retry=retry)},
            ],
        },
    ]
    if add_answer:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": str(answer)}],
            }
        )
    return messages


for form in ["MC", "SA", "LA"]:
    example = next(row for row in records["train"] if row["metadata"]["question_form"] == form)
    print("=" * 80)
    print(form, example["metadata"]["question_id"])
    print(build_user_text(example))
    print("정답:", example["model_output"]["answer"])
"""
)

markdown("## 5. InternVL3.5-8B 4-bit 로드")

code(
    r"""
from transformers import AutoProcessor, BitsAndBytesConfig

try:
    from transformers import AutoModelForMultimodalLM as InternVLModelClass
except ImportError:
    from transformers import AutoModelForImageTextToText as InternVLModelClass


compute_dtype = torch.bfloat16 if bf16_supported else torch.float16
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
if getattr(processor, "tokenizer", None) is not None:
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

model = InternVLModelClass.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    quantization_config=quantization_config,
    torch_dtype=compute_dtype,
    device_map={"": 0},
    low_cpu_mem_usage=True,
)

model.config.use_cache = not DO_TRAIN
torch.set_float32_matmul_precision("high")

print("모델 로드 완료:", MODEL_ID)
print("연산 dtype:", compute_dtype)
print("모델 입력 장치:", next(model.parameters()).device)
"""
)

markdown("## 6. 이미지 로더·응답 정규화·단일 추론 함수")

code(
    r"""
Image.MAX_IMAGE_PIXELS = None


def image_path_for(record, split):
    return image_dirs[split] / record["model_input"]["image_name"]


def load_rgb_image(path, max_side):
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        if max(image.size) > max_side:
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return image.copy()


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
def generate_raw(record, split, retry=False):
    image = load_rgb_image(image_path_for(record, split), INFERENCE_MAX_IMAGE_SIDE)
    messages = build_messages(record, retry=retry)
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[prompt_text],
        images=[image],
        add_special_tokens=False,
        max_patches=INFERENCE_MAX_PATCHES,
        return_tensors="pt",
    )
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
    )
    new_tokens = generated[:, input_length:]
    raw_text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    del image, inputs, generated, new_tokens
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
        # 제출 파일의 빈 값을 막기 위한 최후의 안전장치입니다.
        answer = "1" if form == "MC" else "확인 불가"

    return {"answer": answer, "raw": raw_text, "retried": retried, "valid": valid}
"""
)

markdown(
    """
### 모델 입출력 스모크 테스트

학습 전에 validation 1건을 추론해 이미지 처리와 채팅 템플릿이 정상인지 확인합니다. 이 셀에서 오류가 나면 학습을 시작하지 말고 런타임과 패키지 버전을 먼저 확인하세요.
"""
)

code(
    r"""
smoke_record = records["validation"][0]
smoke_result = predict_one(smoke_record, "validation")

print("문항 유형:", smoke_record["metadata"]["question_form"])
print("질문:", smoke_record["model_input"]["question"])
print("예측:", smoke_result["answer"])
print("원문 출력:", smoke_result["raw"])
print("정답:", smoke_record["model_output"]["answer"])
"""
)

markdown("## 7. QLoRA 학습 준비")

code(
    r"""
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset


class MultimodalQADataset(Dataset):
    def __init__(self, split_records, max_samples=None):
        self.rows = list(split_records)
        if max_samples is not None:
            self.rows = self.rows[:max_samples]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class InternVLCollator:
    def __init__(self, processor, split, image_dir, max_length, max_image_side, max_patches):
        self.processor = processor
        self.split = split
        self.image_dir = image_dir
        self.max_length = max_length
        self.max_image_side = max_image_side
        self.max_patches = max_patches

    def __call__(self, features):
        images = []
        full_texts = []
        prompt_texts = []

        for record in features:
            image_path = self.image_dir / record["model_input"]["image_name"]
            images.append(load_rgb_image(image_path, self.max_image_side))
            answer = record["model_output"]["answer"]

            full_messages = build_messages(record, answer=answer, add_answer=True)
            prompt_messages = build_messages(record, add_answer=False)
            full_texts.append(
                self.processor.apply_chat_template(
                    full_messages, tokenize=False, add_generation_prompt=False
                )
            )
            prompt_texts.append(
                self.processor.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                )
            )

        full_batch = self.processor(
            text=full_texts,
            images=images,
            add_special_tokens=False,
            max_patches=self.max_patches,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        prompt_batch = self.processor(
            text=prompt_texts,
            images=images,
            add_special_tokens=False,
            max_patches=self.max_patches,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )

        sequence_lengths = full_batch["attention_mask"].sum(dim=1)
        longest_sequence = int(sequence_lengths.max().item())
        if longest_sequence > self.max_length:
            raise RuntimeError(
                f"학습 입력이 {longest_sequence}토큰으로 MAX_SEQUENCE_LENGTH="
                f"{self.max_length}를 넘었습니다. TRAIN_MAX_PATCHES를 더 낮추세요."
            )

        labels = full_batch["input_ids"].clone()
        labels[full_batch["attention_mask"] == 0] = -100

        # 이미지·질문·assistant 시작 토큰까지는 loss에서 제외합니다.
        for row_index in range(labels.shape[0]):
            prompt_length = int(prompt_batch["attention_mask"][row_index].sum().item())
            labels[row_index, :prompt_length] = -100
            if torch.all(labels[row_index] == -100):
                raise RuntimeError(
                    "정답 토큰이 MAX_SEQUENCE_LENGTH 밖으로 잘렸습니다. "
                    "TRAIN_MAX_PATCHES를 낮추거나 MAX_SEQUENCE_LENGTH를 늘리세요."
                )

        full_batch["labels"] = labels
        return full_batch


train_dataset = MultimodalQADataset(records["train"], TRAIN_MAX_SAMPLES)
train_collator = InternVLCollator(
    processor=processor,
    split="train",
    image_dir=image_dirs["train"],
    max_length=MAX_SEQUENCE_LENGTH,
    max_image_side=TRAIN_MAX_IMAGE_SIDE,
    max_patches=TRAIN_MAX_PATCHES,
)


def available_lora_targets(model):
    preferred = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    linear_names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and "language_model" in name and "lm_head" not in name
    ]
    targets = [suffix for suffix in preferred if any(name.endswith(suffix) for name in linear_names)]
    if not targets:
        raise RuntimeError("언어 모델 LoRA target module을 찾지 못했습니다.")
    return targets


if USE_SAVED_ADAPTER:
    adapter_config_path = ADAPTER_DIR / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"저장된 adapter가 없습니다: {adapter_config_path}")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR, is_trainable=DO_TRAIN)
    print("저장된 adapter 로드:", ADAPTER_DIR)
elif DO_TRAIN:
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    target_modules = available_lora_targets(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("LoRA targets:", target_modules)
else:
    print("DO_TRAIN=False: 제로샷 모델을 그대로 사용합니다.")
"""
)

markdown("### 학습 전 loss 스모크 테스트")

code(
    r"""
if DO_TRAIN:
    model.train()
    one_batch = train_collator([train_dataset[0]])
    device = next(model.parameters()).device
    one_batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in one_batch.items()
    }
    with torch.no_grad():
        smoke_output = model(**one_batch)
    print("학습 loss 스모크 테스트:", float(smoke_output.loss))
    del one_batch, smoke_output
    gc.collect()
    torch.cuda.empty_cache()
else:
    print("제로샷 모드이므로 학습 loss 테스트를 건너뜁니다.")
"""
)

markdown("## 8. QLoRA 학습 및 adapter 저장")

code(
    r"""
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint


if DO_TRAIN:
    model.config.use_cache = False
    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        bf16=bf16_supported,
        fp16=not bf16_supported,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=train_collator,
    )

    last_checkpoint = get_last_checkpoint(str(CHECKPOINT_DIR))
    if last_checkpoint:
        print("중단 지점부터 학습 재개:", last_checkpoint)
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(str(ADAPTER_DIR))
    processor.save_pretrained(str(ADAPTER_DIR))

    train_metrics_path = RUN_OUTPUT_DIR / "train_metrics.json"
    train_metrics_path.write_text(
        json.dumps(train_result.metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("adapter 저장 완료:", ADAPTER_DIR)
    print("학습 지표 저장:", train_metrics_path)
else:
    print("DO_TRAIN=False: 학습을 건너뜁니다.")

model.eval()
model.config.use_cache = True
gc.collect()
torch.cuda.empty_cache()
"""
)

markdown(
    """
## 9. Validation 진단

MC와 SA는 문자열 exact match를 간단히 확인합니다. LA는 공식 평가기의 토큰화·ROUGE·BLEU 구현과 다를 수 있으므로 여기서는 길이와 예시만 확인합니다. 최종 점수는 대회 평가 결과를 기준으로 판단하세요.
"""
)

code(
    r"""
def comparable_answer(text, form):
    if form == "MC":
        normalized, valid = normalize_mc(text)
        return normalized if valid else str(text).strip()
    return re.sub(r"\s+", " ", str(text)).strip()


validation_subset = records["validation"][:VALIDATION_MAX_SAMPLES]
validation_rows = []

for record in tqdm(validation_subset, desc="validation 추론"):
    result = predict_one(record, "validation")
    form = record["metadata"]["question_form"]
    target = record["model_output"]["answer"]
    validation_rows.append(
        {
            "question_id": record["metadata"]["question_id"],
            "question_form": form,
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
display(validation_df.groupby("question_form")["exact_match"].agg(["count", "mean"]))
display(validation_df.head(10))

validation_path = RUN_OUTPUT_DIR / "validation_predictions.csv"
validation_df.to_csv(validation_path, index=False, encoding="utf-8-sig")
print("validation 결과 저장:", validation_path)
"""
)

markdown(
    """
## 10. Test 800건 추론

20건마다 Google Drive에 예측 캐시를 저장합니다. Colab 연결이 끊겨도 같은 설정으로 다시 실행하면 이미 완료된 question_id는 건너뜁니다.

모델, adapter 또는 프롬프트를 변경했다면 기존 캐시가 섞이지 않도록 RUN_NAME을 바꾸거나 RESET_PREDICTION_CACHE를 True로 한 번 실행하세요.
"""
)

code(
    r"""
prediction_cache_path = RUN_OUTPUT_DIR / "test_prediction_cache.json"


def atomic_write_json(data, path):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_path.replace(path)


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
for record in tqdm(records["test"], desc="test 추론"):
    question_id = record["metadata"]["question_id"]
    if question_id in prediction_cache:
        continue

    result = predict_one(record, "test")
    prediction_cache[question_id] = {
        "answer": result["answer"],
        "raw": result["raw"],
        "question_form": record["metadata"]["question_form"],
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
"""
)

markdown("## 11. 제출 JSON·ZIP 생성 및 최종 검증")

code(
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
    answer = str(prediction_cache[question_id]["answer"]).strip()
    record["model_output"] = {"answer": answer}
    preview_rows.append(
        {
            "question_id": question_id,
            "question_form": form,
            "image_name": record["model_input"]["image_name"],
            "answer": answer,
            "answer_length": len(answer),
        }
    )

# 원본 순서·메타데이터·입력 보존 여부
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

submission_json_path = RUN_OUTPUT_DIR / "submission_internvl35_8b.json"
submission_zip_path = RUN_OUTPUT_DIR / "submission_internvl35_8b.zip"
submission_preview_path = RUN_OUTPUT_DIR / "submission_preview.csv"

atomic_write_json(submission, submission_json_path)
pd.DataFrame(preview_rows).to_csv(
    submission_preview_path, index=False, encoding="utf-8-sig"
)
with zipfile.ZipFile(submission_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write(submission_json_path, arcname=submission_json_path.name)

# 저장 파일을 다시 읽어 마지막으로 검사합니다.
reloaded_submission = json.loads(submission_json_path.read_text(encoding="utf-8"))
assert len(reloaded_submission) == 800
assert all("model_output" in row for row in reloaded_submission)

print("제출 파일 생성 완료")
print("JSON:", submission_json_path)
print("ZIP :", submission_zip_path)
print("미리보기:", submission_preview_path)
print("JSON 크기:", f"{submission_json_path.stat().st_size / 1024:.1f} KiB")
display(pd.DataFrame(preview_rows).head(10))
display(pd.DataFrame(preview_rows).groupby("question_form")["answer_length"].describe())
"""
)

markdown(
    """
## 12. 제출 전 체크리스트

- Drive의 **CV_korea/outputs/internvl35_8b_qlora_v1** 폴더에 JSON과 ZIP이 생성됐는지 확인
- submission_preview.csv에서 MC가 1 또는 1/3 형태인지 확인
- SA가 질문의 음절·어절 지시를 지켰는지 표본 점검
- LA가 250자 이하이고 불필요한 풀이·생각 태그가 없는지 표본 점검
- 대회 제출 시스템이 JSON 직접 업로드인지 ZIP 업로드인지 공지에서 다시 확인
- 팀원에게 모델 ID, DO_TRAIN 값, 학습 epoch, 이미지 최대 크기, RUN_NAME을 함께 공유

중간 캐시와 adapter도 모두 Drive에 저장되므로 Colab 세션이 종료되어도 결과를 복구할 수 있습니다.
"""
)


OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUTPUT_PATH)
print(f"Wrote {OUTPUT_PATH}")
