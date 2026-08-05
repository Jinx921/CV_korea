import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / "notebooks" / "qwen3vl_8b_lora.ipynb"
OUTPUT_PATH = ROOT / "notebooks" / "qwen3vl_8b_lora_paddle_crop.ipynb"


def source_lines(text):
    return text.strip("\n").splitlines(keepends=True)


def set_source(notebook, index, text):
    notebook["cells"][index]["source"] = source_lines(text)
    notebook["cells"][index]["execution_count"] = None
    notebook["cells"][index]["outputs"] = []


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


def find_markdown_index(notebook, prefix):
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "markdown":
            continue
        if "".join(cell["source"]).strip().startswith(prefix):
            return index
    raise ValueError(prefix)


notebook = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))

set_source(
    notebook,
    0,
    r"""
## Qwen3-VL 8B LoRA + PaddleOCR 글자 crop 실험

이 노트북은 OCR팀의 리더보드 비교를 위한 end-to-end 실험입니다.

1. PP-OCRv5 한국어 모델로 글자 위치와 텍스트를 추출
2. OCR 후보 문항에만 원본 이미지와 확대 글자 영역을 Qwen3-VL에 입력
3. **OCR이 인식한 문자열은 Qwen 프롬프트에 넣지 않음**
4. 기존 Qwen LoRA 기준과 동일하게 train 1,000건, 1 epoch 일반 LoRA 학습
5. validation 확인 후 test 800건 추론 및 제출 JSON·ZIP 생성
6. 팀 내 최고 리더보드 모델로 선정되면 RAG팀 전달용 PaddleOCR corpus 생성

기존 Qwen baseline과 바뀌는 핵심 변수는 OCR 후보 문항에 추가되는 **확대 글자 이미지**입니다.
""",
)

set_source(
    notebook,
    1,
    r"""
### 0. Colab 런타임 준비

일반 LoRA 학습에는 A100 40GB급 GPU를 권장합니다. PaddleOCR는 CPU에서 실행하고 Qwen3-VL만 GPU를 사용합니다.

첫 설치 후 런타임이 한 번 자동 재시작됩니다. 재연결되면 첫 셀부터 다시 실행하세요.
""",
)

set_source(
    notebook,
    2,
    r"""
# Qwen과 PaddleOCR가 함께 동작하는 버전을 고정합니다.
%pip uninstall -q -y gradio gradio_client torchao paddlepaddle-gpu

%pip install -q "paddlepaddle==3.2.0" -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
%pip install -q "paddleocr==3.5.0" "langchain-text-splitters"

%pip install -q -U \
  "transformers>=4.57.0,<5" \
  "huggingface-hub==0.36.2" \
  "accelerate>=1.2" \
  "peft>=0.14" \
  "datasets>=3.2" \
  "sentencepiece" \
  "safetensors" \
  "pandas>=2.2" \
  "tqdm>=4.66"

%pip install -q --no-cache-dir --force-reinstall "Pillow==12.3.0"

import os
from pathlib import Path

restart_marker = Path("/content/.qwen3vl_paddle_crop_env_v1")
if not restart_marker.exists():
    restart_marker.write_text("ready", encoding="utf-8")
    print("패키지 설치 완료. 런타임을 재시작합니다.")
    print("재연결된 뒤 이 노트북을 첫 셀부터 다시 실행하세요.")
    os.kill(os.getpid(), 9)

import PIL
import paddle
import paddleocr
from PIL import Image, ImageOps

assert PIL.__version__ == "12.3.0", PIL.__version__
print("Pillow:", PIL.__version__)
print("PaddlePaddle:", paddle.__version__)
print("PaddleOCR:", paddleocr.__version__)
""",
)

set_source(
    notebook,
    5,
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
from PIL import Image, ImageDraw, ImageOps
from tqdm.auto import tqdm


# 사용자가 주로 바꿀 설정
DRIVE_PROJECT_DIR = Path("/content/drive/MyDrive/CV_korea")
DRIVE_ZIP_DIR = DRIVE_PROJECT_DIR / "data"

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
RUN_NAME = "qwen3vl_8b_lora_paddle_crop_v1"
TRAINING_METHOD = "lora_bf16_paddle_crop"

# True: train 1,000건 일반 LoRA 학습 / False: 저장 adapter로 추론만 수행
DO_TRAIN = True
USE_SAVED_ADAPTER = False

TRAIN_MAX_SAMPLES = None
VALIDATION_MAX_SAMPLES = 30  # 최종 진단은 200 권장

# 기존 Qwen LoRA와 동일한 학습 조건
NUM_EPOCHS = 1
LEARNING_RATE = 1e-4
GRADIENT_ACCUMULATION_STEPS = 16
TRAIN_MAX_IMAGE_SIDE = 896
INFERENCE_MAX_IMAGE_SIDE = 1344

# OCR crop 이미지 토큰이 추가되므로 입력 상한만 2048에서 3072로 확장합니다.
MAX_SEQUENCE_LENGTH = 3072
MAX_OCR_CROPS = 2
OCR_CROP_MAX_SIDE = 448

# PaddleOCR 설정
OCR_VERSION = "PP-OCRv5"
OCR_DETECTION_MODEL = "PP-OCRv5_server_det"
OCR_RECOGNITION_MODEL = "korean_PP-OCRv5_mobile_rec"
OCR_MAX_IMAGE_SIDE = 4000
OCR_MIN_SCORE = 0.30
OCR_RAG_SCORE = 0.50
OCR_APPLY_MODE = "routed"  # OCR 후보 질문에만 crop 적용
RESET_OCR_CACHE = False
OCR_CACHE_SAVE_EVERY = 20

# 이 제출이 OCR팀 최고점으로 선정된 뒤에만 True로 바꾸고 마지막 RAG export 셀 실행
EXPORT_RAG_AFTER_LEADERBOARD = False

RESET_PREDICTION_CACHE = False
SAVE_EVERY_N_PREDICTIONS = 20
SEED = 42

COLAB_WORK_DIR = Path("/content/kculture_qwen3vl_paddle_crop")
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
print("OCR 검출:", OCR_DETECTION_MODEL)
print("OCR 인식:", OCR_RECOGNITION_MODEL)
print("Qwen 입력 방식: 원본 이미지 + 확대 crop, OCR 문자열 프롬프트 주입 없음")
""",
)

set_source(
    notebook,
    12,
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


def build_messages(record, images=None, answer=None, add_answer=False, retry=False):
    # OCR 인식 문자열은 content에 추가하지 않습니다.
    image_contents = []
    for image in images or [None]:
        image_content = {"type": "image"}
        if image is not None:
            image_content["image"] = image
        image_contents.append(image_content)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": image_contents
            + [{"type": "text", "text": build_user_text(record, retry=retry)}],
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
""",
)

set_source(
    notebook,
    16,
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
    original = load_rgb_image(image_path_for(record, split), max_image_side)
    images = [original]
    if should_apply_ocr(record):
        crops = make_ocr_crops(record, split, max_crops=MAX_OCR_CROPS)
        images.extend(crop["image"] for crop in crops)
    return images


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
    images = build_input_images(record, split, INFERENCE_MAX_IMAGE_SIDE)
    messages = build_messages(record, images=images, retry=retry)
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
    new_tokens = generated[:, input_length:]
    raw_text = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    del images, inputs, generated, new_tokens
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

set_source(
    notebook,
    18,
    r"""
smoke_record = records["validation"][0]
ensure_ocr_records("validation", [smoke_record], only_routed=True)

print("ID:", smoke_record["metadata"]["question_id"])
print("OCR crop 적용:", should_apply_ocr(smoke_record))
if should_apply_ocr(smoke_record):
    print("OCR crop 수:", len(make_ocr_crops(smoke_record, "validation")))
print("질문:", smoke_record["model_input"]["question"])
print("정답:", smoke_record["model_output"]["answer"])

smoke_result = predict_one(smoke_record, "validation")
print("예측:", smoke_result["answer"])
print("원본 출력:", smoke_result["raw"])
""",
)

set_source(
    notebook,
    20,
    r"""
from peft import LoraConfig, PeftModel, get_peft_model
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


class Qwen3VLCollator:
    def __init__(self, processor, split, max_length, max_image_side):
        self.processor = processor
        self.split = split
        self.max_length = max_length
        self.max_image_side = max_image_side

    def __call__(self, features):
        if len(features) != 1:
            raise RuntimeError("Qwen3-VL collator는 per_device_train_batch_size=1을 전제로 합니다.")

        record = features[0]
        images = build_input_images(record, self.split, self.max_image_side)
        answer = record["model_output"]["answer"]

        full_messages = build_messages(
            record, images=images, answer=answer, add_answer=True
        )
        prompt_messages = build_messages(record, images=images, add_answer=False)

        full_batch = self.processor.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        prompt_batch = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        full_batch.pop("token_type_ids", None)
        prompt_batch.pop("token_type_ids", None)

        sequence_length = int(full_batch["attention_mask"].sum().item())
        if sequence_length > self.max_length:
            raise RuntimeError(
                f"학습 입력이 {sequence_length}토큰으로 MAX_SEQUENCE_LENGTH="
                f"{self.max_length}를 넘었습니다. MAX_OCR_CROPS 또는 이미지 크기를 낮추세요."
            )

        labels = full_batch["input_ids"].clone()
        labels[full_batch["attention_mask"] == 0] = -100
        prompt_length = int(prompt_batch["attention_mask"].sum().item())
        labels[:, :prompt_length] = -100
        if torch.all(labels == -100):
            raise RuntimeError("정답 토큰이 학습 label에 포함되지 않았습니다.")

        full_batch["labels"] = labels
        return full_batch


train_dataset = MultimodalQADataset(records["train"], TRAIN_MAX_SAMPLES)
train_collator = Qwen3VLCollator(
    processor=processor,
    split="train",
    max_length=MAX_SEQUENCE_LENGTH,
    max_image_side=TRAIN_MAX_IMAGE_SIDE,
)


def available_lora_targets(model):
    preferred = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    linear_names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and "language_model" in name
        and "lm_head" not in name
    ]
    targets = [suffix for suffix in preferred if any(name.endswith(suffix) for name in linear_names)]
    if not targets:
        raise RuntimeError("Qwen 언어 모델 LoRA target module을 찾지 못했습니다.")
    return targets


if USE_SAVED_ADAPTER:
    adapter_config_path = ADAPTER_DIR / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"저장된 adapter가 없습니다: {adapter_config_path}")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR, is_trainable=DO_TRAIN)
    print("저장된 일반 LoRA adapter 로드:", ADAPTER_DIR)
elif DO_TRAIN:
    for parameter in model.parameters():
        parameter.requires_grad = False

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
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
    print("학습 방식: 일반 LoRA + PaddleOCR 글자 crop")
    print("LoRA targets:", target_modules)
else:
    print("DO_TRAIN=False: 저장 adapter를 사용한 추론 모드입니다.")
""",
)

set_source(
    notebook,
    26,
    r"""
def comparable_answer(text, form):
    if form == "MC":
        normalized, valid = normalize_mc(text)
        return normalized if valid else str(text).strip()
    return re.sub(r"\s+", " ", str(text)).strip()


validation_subset = records["validation"][:VALIDATION_MAX_SAMPLES]
ensure_ocr_records("validation", validation_subset, only_routed=True)

validation_rows = []
for record in tqdm(validation_subset, desc="validation 추론"):
    result = predict_one(record, "validation")
    form = record["metadata"]["question_form"]
    target = record["model_output"]["answer"]
    validation_rows.append(
        {
            "question_id": record["metadata"]["question_id"],
            "question_form": form,
            "ocr_crop_applied": should_apply_ocr(record),
            "ocr_crop_count": len(make_ocr_crops(record, "validation")) if should_apply_ocr(record) else 0,
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
display(validation_df.groupby(["question_form", "ocr_crop_applied"])["exact_match"].agg(["count", "mean"]))
display(validation_df.head(10))

validation_path = RUN_OUTPUT_DIR / "validation_predictions.csv"
validation_df.to_csv(validation_path, index=False, encoding="utf-8-sig")
print("validation 결과 저장:", validation_path)
""",
)

set_source(
    notebook,
    28,
    r"""
ensure_ocr_records("test", records["test"], only_routed=True)

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
for record in tqdm(records["test"], desc="test Paddle crop 추론"):
    question_id = record["metadata"]["question_id"]
    if question_id in prediction_cache:
        continue

    result = predict_one(record, "test")
    prediction_cache[question_id] = {
        "answer": result["answer"],
        "raw": result["raw"],
        "question_form": record["metadata"]["question_form"],
        "ocr_crop_applied": should_apply_ocr(record),
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

set_source(
    notebook,
    30,
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
            "ocr_crop_applied": cached["ocr_crop_applied"],
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

submission_json_path = RUN_OUTPUT_DIR / "submission_qwen3vl_8b_lora_paddle_crop.json"
submission_zip_path = RUN_OUTPUT_DIR / "submission_qwen3vl_8b_lora_paddle_crop.zip"
submission_preview_path = RUN_OUTPUT_DIR / "submission_preview.csv"

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
display(pd.DataFrame(preview_rows).groupby(["question_form", "ocr_crop_applied"]).size())
display(pd.DataFrame(preview_rows).head(10))
""",
)

# OCR 모델과 캐시 함수는 데이터 검증 직후, 프롬프트 정의 전에 실행합니다.
ocr_cells = [
    markdown("### 4. PP-OCRv5 한국어 초기화 및 대표 이미지 검수"),
    code(
        r"""
from paddleocr import PaddleOCR


OCR_CUE_RE = re.compile(
    r"적혀|쓰여|써\s*있|문구|글자|텍스트|안내|표지|간판|현수막|포스터|메뉴|게시판|"
    r"명판|상호|로고|화면|앱|책|연도|날짜|가격|요금|번호|숫자|기관|부처|장소의\s*이름|"
    r"무엇이라\s*하|몇\s*(?:개|명|시|분|원|년|월|일)|읽(?:고|어|으)"
)
ENGINE_SIGNATURE = {
    "ocr_version": OCR_VERSION,
    "detection_model": OCR_DETECTION_MODEL,
    "recognition_model": OCR_RECOGNITION_MODEL,
    "max_side": OCR_MAX_IMAGE_SIDE,
    "min_score": OCR_MIN_SCORE,
}
OCR_CACHE_PATH = RUN_OUTPUT_DIR / "paddleocr_cache.json"


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
        print("OCR 엔진 설정이 달라 기존 캐시를 사용하지 않습니다.")
        ocr_cache = {"engine_signature": ENGINE_SIGNATURE, "items": {}}
else:
    ocr_cache = {"engine_signature": ENGINE_SIGNATURE, "items": {}}


ocr_engine = PaddleOCR(
    lang="korean",
    ocr_version=OCR_VERSION,
    text_detection_model_name=OCR_DETECTION_MODEL,
    text_recognition_model_name=OCR_RECOGNITION_MODEL,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True,
    text_det_limit_side_len=OCR_MAX_IMAGE_SIDE,
    text_det_limit_type="max",
    text_rec_score_thresh=OCR_MIN_SCORE,
    device="cpu",
)


def as_list(value):
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def clean_ocr_text(value):
    return re.sub(r"\s+", " ", str(value)).strip()


def image_cache_key(record, split):
    return f"{split}:{record['model_input']['image_name']}"


def should_apply_ocr(record):
    if OCR_APPLY_MODE == "all":
        return True
    if OCR_APPLY_MODE != "routed":
        raise ValueError(OCR_APPLY_MODE)
    model_input = record["model_input"]
    parts = [model_input.get("question", "")]
    parts.extend(str(option) for option in (model_input.get("options") or []))
    return bool(OCR_CUE_RE.search(" ".join(parts)))


def parse_paddle_result(result):
    payload = result.json
    payload = payload.get("res", payload)
    texts = as_list(payload.get("rec_texts"))
    scores = as_list(payload.get("rec_scores"))
    boxes = as_list(payload.get("rec_boxes"))
    polygons = as_list(payload.get("dt_polys"))

    entries = []
    for text, score, box in zip(texts, scores, boxes):
        text = clean_ocr_text(text)
        score = float(score)
        if not text or score < OCR_MIN_SCORE:
            continue
        entries.append({
            "text": text,
            "score": round(score, 6),
            "box": [int(value) for value in box],
        })
    entries.sort(key=lambda row: (row["box"][1], row["box"][0]))
    return entries, polygons


def run_ocr_record(record, split, force=False):
    key = image_cache_key(record, split)
    if key in ocr_cache["items"] and not force:
        return ocr_cache["items"][key]

    path = image_dirs[split] / record["model_input"]["image_name"]
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB").copy()
    results = list(ocr_engine.predict(np.asarray(image)))

    entries = []
    polygons = []
    for result in results:
        result_entries, result_polygons = parse_paddle_result(result)
        entries.extend(result_entries)
        polygons.extend(result_polygons)

    item = {
        "width": image.width,
        "height": image.height,
        "entries": entries,
        "detection_polygons": polygons,
    }
    ocr_cache["items"][key] = item
    return item


def ensure_ocr_records(split, split_records, only_routed=True):
    unique_records = {}
    for record in split_records:
        if only_routed and not should_apply_ocr(record):
            continue
        unique_records.setdefault(record["model_input"]["image_name"], record)

    new_count = 0
    for record in tqdm(unique_records.values(), desc=f"{split} PP-OCRv5"):
        key = image_cache_key(record, split)
        if key in ocr_cache["items"]:
            continue
        run_ocr_record(record, split)
        new_count += 1
        if new_count % OCR_CACHE_SAVE_EVERY == 0:
            atomic_write_json(ocr_cache, OCR_CACHE_PATH)
    atomic_write_json(ocr_cache, OCR_CACHE_PATH)
    print(f"{split} OCR 준비 완료: 신규 {new_count}, 전체 캐시 {len(ocr_cache['items'])}")


def crop_rank(entry, image_size):
    x1, y1, x2, y2 = entry["box"]
    area_ratio = max(1, x2 - x1) * max(1, y2 - y1) / max(1, image_size[0] * image_size[1])
    return entry["score"] + min(len(entry["text"]) / 20, 1.0) + min(area_ratio * 50, 1.0)


def make_ocr_crops(record, split, max_crops=MAX_OCR_CROPS):
    key = image_cache_key(record, split)
    if key not in ocr_cache["items"]:
        raise RuntimeError(f"OCR 캐시가 없습니다: {key}. 먼저 ensure_ocr_records를 실행하세요.")

    path = image_dirs[split] / record["model_input"]["image_name"]
    with Image.open(path) as opened:
        original = ImageOps.exif_transpose(opened).convert("RGB").copy()
    entries = ocr_cache["items"][key]["entries"]
    ranked = sorted(entries, key=lambda entry: crop_rank(entry, original.size), reverse=True)

    crops = []
    for entry in ranked[:max_crops]:
        x1, y1, x2, y2 = entry["box"]
        padding = max(6, int(0.10 * max(x2 - x1, y2 - y1)))
        box = (
            max(0, x1 - padding),
            max(0, y1 - padding),
            min(original.width, x2 + padding),
            min(original.height, y2 + padding),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        crop = original.crop(box)
        scale = OCR_CROP_MAX_SIDE / max(crop.size)
        if scale != 1:
            resized = (
                max(1, round(crop.width * scale)),
                max(1, round(crop.height * scale)),
            )
            crop = crop.resize(resized, Image.Resampling.LANCZOS)
        crops.append({"image": crop, "entry": entry, "box": box})
    return crops


def draw_ocr_preview(record, split, item):
    path = image_dirs[split] / record["model_input"]["image_name"]
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    for polygon in item["detection_polygons"]:
        points = [(int(point[0]), int(point[1])) for point in polygon]
        if len(points) >= 2:
            draw.line(points + [points[0]], fill="red", width=max(2, image.width // 800))
    for entry in item["entries"]:
        x1, y1, x2, y2 = entry["box"]
        draw.rectangle((x1, y1, x2, y2), outline="lime", width=max(2, image.width // 600))
    return image


SMOKE_EXPECTED = {
    "1904": ["화포천습지생태박물관"],
    "0691": ["첫3번무료"],
    "0774": ["공용냉장실", "개인음식물보관", "유통기한"],
}


def compact_match(value):
    return re.sub(r"[^0-9A-Za-z가-힣]", "", str(value)).lower()


lookup = {}
for split, split_records in records.items():
    for record in split_records:
        lookup[str(record["metadata"]["question_id"]).zfill(4)] = (split, record)

smoke_rows = []
for expected_id, phrases in SMOKE_EXPECTED.items():
    if expected_id not in lookup:
        print("대표 ID를 찾지 못했습니다:", expected_id)
        continue
    split, record = lookup[expected_id]
    item = run_ocr_record(record, split, force=True)
    combined = compact_match(" ".join(entry["text"] for entry in item["entries"]))
    hits = {phrase: compact_match(phrase) in combined for phrase in phrases}
    print("=" * 80)
    print("ID:", expected_id, "|", hits)
    print("질문:", record["model_input"]["question"])
    for entry in item["entries"]:
        print(f"[{entry['score']:.3f}] {entry['text']}")
    display(draw_ocr_preview(record, split, item))
    smoke_rows.append({"question_id": expected_id, "phrase_hits": hits, "all_found": all(hits.values())})

atomic_write_json(ocr_cache, OCR_CACHE_PATH)
display(pd.DataFrame(smoke_rows))
print("빨간 박스는 검출, 초록 박스는 인식 결과입니다.")
print("한글이 제대로 보일 때만 다음 train OCR 셀로 진행하세요.")
"""
    ),
    markdown("### 5. 학습용 OCR 후보 이미지 사전 처리"),
    code(
        r"""
# PaddleOCR를 DataLoader worker 안에서 실행하지 않고, 학습 전에 Drive 캐시로 고정합니다.
train_rows_for_ocr = records["train"][:TRAIN_MAX_SAMPLES] if TRAIN_MAX_SAMPLES else records["train"]
ensure_ocr_records("train", train_rows_for_ocr, only_routed=True)

routed_train_count = sum(should_apply_ocr(record) for record in train_rows_for_ocr)
crop_train_count = sum(
    bool(ocr_cache["items"][image_cache_key(record, "train")]["entries"])
    for record in train_rows_for_ocr if should_apply_ocr(record)
)
print("OCR 후보 train 문항:", routed_train_count)
print("실제 crop 생성 가능 문항:", crop_train_count)
"""
    ),
]

prompt_index = find_markdown_index(notebook, "### 4. 문항 유형별")
notebook["cells"][prompt_index:prompt_index] = ocr_cells

# 제목 번호를 OCR 셀 삽입 뒤 자연스럽게 맞춥니다.
for cell in notebook["cells"]:
    if cell["cell_type"] != "markdown":
        continue
    text = "".join(cell["source"])
    replacements = {
        "### 4. 문항 유형별": "### 6. 문항 유형별",
        "### 5. Qwen3-VL": "### 7. Qwen3-VL",
        "### 6. 이미지 로더": "### 8. 이미지 로더",
        "### 7. LoRA": "### 9. LoRA",
        "### 8. LoRA": "### 10. LoRA",
        "### 9. Validation": "### 11. Validation",
        "### 10. Test": "### 12. Test",
        "### 11. 제출": "### 13. 제출",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    cell["source"] = source_lines(text)

notebook["cells"].extend([
    markdown("### 14. 팀 최고점 선정 후 RAG팀 전달용 OCR corpus 생성"),
    code(
        r"""
def write_jsonl(rows, path):
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


if not EXPORT_RAG_AFTER_LEADERBOARD:
    print("아직 RAG corpus를 생성하지 않습니다.")
    print("이 제출이 OCR팀 최고 리더보드 점수일 때 설정 셀의")
    print("EXPORT_RAG_AFTER_LEADERBOARD=True로 바꾸고 이 셀을 다시 실행하세요.")
else:
    # 최고점으로 선정된 경우 모든 split의 모든 고유 이미지를 PaddleOCR로 처리합니다.
    for split in ["train", "validation", "test"]:
        ensure_ocr_records(split, records[split], only_routed=False)

    image_documents = []
    question_contexts = []
    seen_images = set()

    for split in ["train", "validation", "test"]:
        for record in records[split]:
            key = image_cache_key(record, split)
            item = ocr_cache["items"][key]
            accepted = [entry for entry in item["entries"] if entry["score"] >= OCR_RAG_SCORE]
            ocr_text = "\n".join(entry["text"] for entry in accepted)

            if key not in seen_images:
                seen_images.add(key)
                image_documents.append({
                    "schema_version": "qwen3vl_paddle_crop_rag_v1",
                    "document_id": f"image:{key}",
                    "split": split,
                    "image_name": record["model_input"]["image_name"],
                    "ocr_text": ocr_text,
                    "ocr_lines": accepted,
                    "ocr_model": OCR_RECOGNITION_MODEL,
                    "detection_model": OCR_DETECTION_MODEL,
                })

            question_contexts.append({
                "schema_version": "qwen3vl_paddle_crop_rag_v1",
                "question_id": record["metadata"]["question_id"],
                "split": split,
                "question_form": record["metadata"]["question_form"],
                "image_name": record["model_input"]["image_name"],
                "question": record["model_input"].get("question", ""),
                "options": record["model_input"].get("options") or [],
                "ocr_text": ocr_text,
            })

    image_jsonl_path = RUN_OUTPUT_DIR / "paddleocr_image_corpus.jsonl"
    context_jsonl_path = RUN_OUTPUT_DIR / "paddleocr_question_context.jsonl"
    image_csv_path = RUN_OUTPUT_DIR / "paddleocr_image_corpus.csv"
    context_csv_path = RUN_OUTPUT_DIR / "paddleocr_question_context.csv"
    rag_zip_path = RUN_OUTPUT_DIR / "paddleocr_rag_delivery.zip"

    write_jsonl(image_documents, image_jsonl_path)
    write_jsonl(question_contexts, context_jsonl_path)

    image_csv_rows = []
    for row in image_documents:
        csv_row = dict(row)
        csv_row["ocr_lines"] = json.dumps(csv_row["ocr_lines"], ensure_ascii=False)
        image_csv_rows.append(csv_row)
    pd.DataFrame(image_csv_rows).to_csv(image_csv_path, index=False, encoding="utf-8-sig")

    context_csv_rows = []
    for row in question_contexts:
        csv_row = dict(row)
        csv_row["options"] = json.dumps(csv_row["options"], ensure_ascii=False)
        context_csv_rows.append(csv_row)
    pd.DataFrame(context_csv_rows).to_csv(context_csv_path, index=False, encoding="utf-8-sig")

    with zipfile.ZipFile(rag_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in [image_jsonl_path, context_jsonl_path, image_csv_path, context_csv_path]:
            zf.write(path, arcname=path.name)

    print("RAG팀 전달 파일 생성 완료:", rag_zip_path)
    print("이미지 OCR 문서:", len(image_documents))
    print("질문 연결 문서:", len(question_contexts))
"""
    ),
    markdown(
        r"""
### 실행 결과 확인 순서

1. 대표 3개 이미지에서 한글 OCR이 제대로 추출되는지 확인
2. 학습 loss 스모크 테스트와 1 epoch LoRA 학습 완료 확인
3. validation에서 `ocr_crop_applied=True` 문항 결과 확인
4. test 추론 후 `submission_qwen3vl_8b_lora_paddle_crop.json` 제출
5. 팀원들의 OCR 실험과 리더보드 점수 비교
6. 이 실험이 최고점이면 `EXPORT_RAG_AFTER_LEADERBOARD=True`로 바꾸고 RAG export 실행

제출에는 JSON 파일을 사용하고, RAG팀에는 `paddleocr_rag_delivery.zip`을 전달합니다.
"""
    ),
])

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
