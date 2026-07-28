import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "notebooks" / "internvl35_8b_lora.ipynb"
OUTPUT_PATH = ROOT / "notebooks" / "qwen3vl_8b_lora.ipynb"


def source_lines(text):
    return text.splitlines(keepends=True)


def set_source(notebook, cell_number, text):
    notebook["cells"][cell_number - 1]["source"] = source_lines(text)


notebook = copy.deepcopy(json.loads(SOURCE_PATH.read_text(encoding="utf-8")))

# 현재 InternVL 노트북의 작은 제목 단계(## / ###), 데이터 처리, 프롬프트,
# 평가 및 제출 형식을 그대로 복사하고 모델 종속 셀만 교체합니다.
for cell in notebook["cells"]:
    text = "".join(cell.get("source", []))
    text = text.replace("InternVL3.5-8B", "Qwen3-VL 8B")
    text = text.replace("OpenGVLab/InternVL3_5-8B-HF", "Qwen/Qwen3-VL-8B-Instruct")
    text = text.replace("internvl35_8b_lora_v1", "qwen3vl_8b_lora_v1")
    text = text.replace("kculture_internvl35", "kculture_qwen3vl")
    text = text.replace(".internvl35_pillow_12_3_ready", ".qwen3vl_pillow_12_3_ready")
    text = text.replace("InternVL 실행", "Qwen3-VL 실행")
    cell["source"] = source_lines(text)
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

set_source(
    notebook,
    1,
    """## Qwen3-VL 8B 한국문화 멀티모달 QA LoRA 베이스라인

이 노트북은 Google Colab에서 다음 작업을 한 번에 수행합니다.

1. Google Drive의 **CV_korea/data** 안에 있는 ZIP 4개를 Colab 로컬 디스크로 해제
2. train / validation / test JSON과 이미지 경로 검증
3. **Qwen/Qwen3-VL-8B-Instruct**를 BF16/FP16로 로드
4. train 1,000건에 일반 LoRA 미세조정
5. validation 진단 및 test 800건 추론
6. 원본 test 구조를 유지한 제출 JSON과 ZIP을 Drive에 저장

InternVL 비교 실험과 동일하게 데이터, MC·SA·LA 프롬프트, 1 epoch, LoRA 설정,
학습률, gradient accumulation, 이미지 최대 변, 생성 길이, 시드와 후처리를 유지합니다.
InternVL 전용 `max_patches`는 Qwen3-VL에 존재하지 않으므로 Qwen의 동적 이미지 토큰화로 대체합니다.

일반 LoRA는 원본 8B 모델을 양자화하지 않으므로 **A100 40GB급 GPU**를 권장합니다.
팀 모델 간 순수 제로샷 성능만 비교할 때는 설정 셀의 `DO_TRAIN=False`로 바꾸세요.
현재 기본값은 LoRA 학습을 수행하도록 `True`입니다.
""",
)

set_source(
    notebook,
    6,
    """import copy
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


# 사용자가 주로 바꿀 설정
DRIVE_PROJECT_DIR = Path("/content/drive/MyDrive/CV_korea")
DRIVE_ZIP_DIR = DRIVE_PROJECT_DIR / "data"

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
RUN_NAME = "qwen3vl_8b_lora_v1"
TRAINING_METHOD = "lora_bf16"

# True: train 1,000건 일반 LoRA 학습 / False: 제로샷 베이스라인
DO_TRAIN = True

# 이전에 저장한 adapter를 새 Colab 세션에서 불러올 때 True
USE_SAVED_ADAPTER = False

# 빠른 코드 점검 시 32 같은 정수로 바꾸고, 최종 학습은 None
TRAIN_MAX_SAMPLES = None
VALIDATION_MAX_SAMPLES = 30  # 최종 확인은 200 권장

# InternVL 비교 노트북과 동일한 학습 조건
NUM_EPOCHS = 1
LEARNING_RATE = 1e-4
GRADIENT_ACCUMULATION_STEPS = 16
TRAIN_MAX_IMAGE_SIDE = 896
INFERENCE_MAX_IMAGE_SIDE = 1344
MAX_SEQUENCE_LENGTH = 2048

# 이전 test 예측 캐시를 지우고 처음부터 다시 추론할 때만 True
RESET_PREDICTION_CACHE = False
SAVE_EVERY_N_PREDICTIONS = 20
SEED = 42

COLAB_WORK_DIR = Path("/content/kculture_qwen3vl")
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
""",
)

set_source(
    notebook,
    7,
    """if not torch.cuda.is_available():
    raise RuntimeError("GPU 런타임이 아닙니다. Colab 런타임 유형을 GPU로 변경하세요.")

gpu_name = torch.cuda.get_device_name(0)
gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
bf16_supported = torch.cuda.is_bf16_supported()

print(f"GPU: {gpu_name}")
print(f"VRAM: {gpu_mem_gb:.1f} GiB")
print(f"BF16 지원: {bf16_supported}")

if DO_TRAIN and gpu_mem_gb < 35:
    raise RuntimeError(
        "Qwen3-VL 8B 일반 LoRA는 원본 모델을 BF16/FP16으로 올리므로 "
        "A100 40GB급 GPU가 필요합니다. 현재 VRAM이 부족합니다. "
        "A100 런타임으로 변경하세요."
    )

print("일반 LoRA 메모리 점검 통과")
""",
)

set_source(
    notebook,
    13,
    r"""SYSTEM_PROMPT = (
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


def build_messages(record, image=None, answer=None, add_answer=False, retry=False):
    image_content = {"type": "image"}
    if image is not None:
        # Qwen3-VL 공식 멀티모달 채팅 템플릿은 PIL 이미지를 content에 직접 받습니다.
        image_content["image"] = image

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                image_content,
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
""",
)

set_source(notebook, 14, "### 5. Qwen3-VL 8B BF16/FP16 로드\n")

set_source(
    notebook,
    15,
    """from transformers import AutoProcessor

try:
    from transformers import Qwen3VLForConditionalGeneration as Qwen3VLModelClass
except ImportError:
    from transformers import AutoModelForImageTextToText as Qwen3VLModelClass


compute_dtype = torch.bfloat16 if bf16_supported else torch.float16

processor = AutoProcessor.from_pretrained(MODEL_ID)
if getattr(processor, "tokenizer", None) is not None:
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

# 일반 LoRA: 원본 모델을 4-bit로 양자화하지 않고 BF16/FP16으로 로드합니다.
# 공식 Qwen3-VL 구현에 포함된 SDPA를 사용해 별도 flash-attn 설치 충돌을 피합니다.
model = Qwen3VLModelClass.from_pretrained(
    MODEL_ID,
    dtype=compute_dtype,
    device_map={"": 0},
    low_cpu_mem_usage=True,
    attn_implementation="sdpa",
)


def set_model_cache(enabled):
    model.config.use_cache = enabled
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = enabled


set_model_cache(not DO_TRAIN)
torch.set_float32_matmul_precision("high")

print("모델 로드 완료:", MODEL_ID)
print("학습 방식: 일반 LoRA (원본 모델 양자화 없음)")
print("원본 모델 dtype:", compute_dtype)
print("모델 입력 장치:", next(model.parameters()).device)
""",
)

set_source(
    notebook,
    17,
    r"""Image.MAX_IMAGE_PIXELS = None


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
    messages = build_messages(record, image=image, retry=retry)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    # Transformers 4.57 일부 조합에서 생성되는 text용 token_type_ids는
    # Qwen3-VL forward 인자가 아니므로 제거합니다.
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
        answer = "1" if form == "MC" else "확인 불가"

    return {"answer": answer, "raw": raw_text, "retried": retried, "valid": valid}
""",
)

set_source(
    notebook,
    21,
    """from peft import LoraConfig, PeftModel, get_peft_model
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
    def __init__(self, processor, image_dir, max_length, max_image_side):
        self.processor = processor
        self.image_dir = image_dir
        self.max_length = max_length
        self.max_image_side = max_image_side

    def __call__(self, features):
        # 비교 실험의 physical batch size가 1이므로 Qwen 공식 멀티모달
        # chat template를 한 문항씩 적용해 이미지/그리드 토큰 정합성을 보장합니다.
        if len(features) != 1:
            raise RuntimeError("Qwen3-VL collator는 per_device_train_batch_size=1을 전제로 합니다.")

        record = features[0]
        image_path = self.image_dir / record["model_input"]["image_name"]
        image = load_rgb_image(image_path, self.max_image_side)
        answer = record["model_output"]["answer"]

        full_messages = build_messages(
            record, image=image, answer=answer, add_answer=True
        )
        prompt_messages = build_messages(record, image=image, add_answer=False)

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
                f"{self.max_length}를 넘었습니다. TRAIN_MAX_IMAGE_SIDE를 더 낮추세요."
            )

        labels = full_batch["input_ids"].clone()
        labels[full_batch["attention_mask"] == 0] = -100

        # 이미지·질문·assistant 시작 토큰까지는 loss에서 제외합니다.
        prompt_length = int(prompt_batch["attention_mask"].sum().item())
        labels[:, :prompt_length] = -100
        if torch.all(labels == -100):
            raise RuntimeError(
                "정답 토큰이 학습 label에 포함되지 않았습니다. "
                "채팅 템플릿과 MAX_SEQUENCE_LENGTH를 확인하세요."
            )

        full_batch["labels"] = labels
        return full_batch


train_dataset = MultimodalQADataset(records["train"], TRAIN_MAX_SAMPLES)
train_collator = Qwen3VLCollator(
    processor=processor,
    image_dir=image_dirs["train"],
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
    print("학습 방식: 일반 LoRA")
    print("LoRA targets:", target_modules)
else:
    print("DO_TRAIN=False: 제로샷 모델을 그대로 사용합니다.")
""",
)

# Qwen의 combined config와 text config 모두에서 cache 상태를 맞춥니다.
cell_25 = "".join(notebook["cells"][24]["source"])
cell_25 = cell_25.replace("model.config.use_cache = False", "set_model_cache(False)")
cell_25 = cell_25.replace("model.config.use_cache = True", "set_model_cache(True)")
notebook["cells"][24]["source"] = source_lines(cell_25)

cell_31 = "".join(notebook["cells"][30]["source"])
cell_31 = cell_31.replace(
    "submission_internvl35_8b_lora.json", "submission_qwen3vl_8b_lora.json"
)
cell_31 = cell_31.replace(
    "submission_internvl35_8b_lora.zip", "submission_qwen3vl_8b_lora.zip"
)
notebook["cells"][30]["source"] = source_lines(cell_31)

notebook.setdefault("metadata", {}).setdefault("kernelspec", {})["display_name"] = "Python 3"
OUTPUT_PATH.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
    encoding="utf-8",
)
print(OUTPUT_PATH)
