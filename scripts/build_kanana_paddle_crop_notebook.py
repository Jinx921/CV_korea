import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / "notebooks" / "qwen3vl_8b_lora_paddle_crop_v2_selective.ipynb"
OUTPUT_PATH = ROOT / "notebooks" / "kanana15v_3b_lora_paddle_crop_typed.ipynb"


def source_lines(text):
    return text.strip("\n").splitlines(keepends=True)


def set_cell(notebook, number, text):
    notebook["cells"][number - 1]["source"] = source_lines(text)


notebook = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))

set_cell(
    notebook,
    1,
    r"""
## Kanana-1.5-V 3B 유형별 LoRA + PaddleOCR 선택적 crop

이 노트북은 Kanana 기준 모델에 기존 PaddleOCR 결과를 선택적으로 결합하는 비교 실험입니다.

1. `kakaocorp/kanana-1.5-v-3b-instruct`를 사용
2. 전체 MC·SA·LA 데이터로 Shared LoRA를 먼저 학습
3. Shared LoRA에서 시작해 MC·SA·LA Adapter를 각각 추가 학습
4. OCR 질문으로 판정된 MC·SA 문항에만 고신뢰도 한글 crop 1개 추가
5. LA는 원본 이미지만 사용
6. OCR 문자열은 프롬프트에 직접 넣지 않음
7. 기존 Qwen PaddleOCR 캐시가 있으면 복사하여 재사용
8. validation 200건 진단 후 test 800건 추론 및 제출 파일 생성

비교 기준은 OCR 없는 Kanana 유형별 Adapter의 리더보드 점수 `44.8395439`입니다.
""",
)

set_cell(
    notebook,
    2,
    r"""
### 0. Colab 런타임 준비

GPU 런타임에서 실행하세요. 첫 실행에서는 패키지를 설치한 뒤 런타임이 한 번 자동 재시작됩니다.
재연결되면 첫 번째 코드 셀부터 다시 실행하세요.
""",
)

set_cell(
    notebook,
    3,
    r"""
# 설치 중간에 재실행해 PaddleX가 중복 초기화되지 않도록 마커를 사용합니다.
import os
import subprocess
import sys
from pathlib import Path


restart_marker = Path("/content/.kanana15v_paddle_crop_env_v1")

if not restart_marker.exists():
    def run_pip(*args):
        subprocess.check_call([sys.executable, "-m", "pip", *args])

    run_pip("uninstall", "-q", "-y", "gradio", "gradio_client", "torchao", "torchaudio", "paddlepaddle-gpu")
    run_pip(
        "install", "-q", "paddlepaddle==3.2.0",
        "-i", "https://www.paddlepaddle.org.cn/packages/stable/cpu/",
    )
    run_pip("install", "-q", "paddleocr==3.5.0", "langchain-text-splitters")
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
    run_pip("install", "-q", "--no-cache-dir", "--force-reinstall", "Pillow==12.3.0")
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

set_cell(
    notebook,
    6,
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

MODEL_ID = "kakaocorp/kanana-1.5-v-3b-instruct"
RUN_NAME = "kanana15v_3b_lora_paddle_crop_typed"
TRAINING_METHOD = "shared_then_typed_lora_paddle_crop"

# True: Shared + 유형별 Adapter 학습 / False: 저장된 유형별 Adapter로 추론만 수행
DO_TRAIN = True
REUSE_COMPLETED_ADAPTERS = True

TRAIN_MAX_SAMPLES = None
VALIDATION_MAX_SAMPLES = 200

# 유형별 Adapter 실험 기본값: 각 문항은 Shared 1회 + 자기 유형 1회 학습됩니다.
SHARED_EPOCHS = 1
TYPED_EPOCHS = 1
LEARNING_RATE = 1e-4
GRADIENT_ACCUMULATION_STEPS = 16
TRAIN_MAX_IMAGE_SIDE = 896
INFERENCE_MAX_IMAGE_SIDE = 1344
MAX_SEQUENCE_LENGTH = 4096

# 선택적 OCR crop
MAX_OCR_CROPS = 1
OCR_CROP_MAX_SIDE = 672
OCR_VERSION = "PP-OCRv5"
OCR_DETECTION_MODEL = "PP-OCRv5_server_det"
OCR_RECOGNITION_MODEL = "korean_PP-OCRv5_mobile_rec"
OCR_MAX_IMAGE_SIDE = 4000
OCR_MIN_SCORE = 0.30
OCR_CROP_MIN_SCORE = 0.80
OCR_CROP_MIN_HANGUL_CHARS = 2
OCR_RAG_SCORE = 0.50
OCR_APPLY_MODE = "selective_mc_sa"
RESET_OCR_CACHE = False
OCR_CACHE_SAVE_EVERY = 20

# 이 제출이 OCR팀 최고점으로 선정된 뒤에만 True로 바꾸고 RAG export 셀을 실행합니다.
EXPORT_RAG_AFTER_LEADERBOARD = False

RESET_PREDICTION_CACHE = False
SAVE_EVERY_N_PREDICTIONS = 20
SEED = 42
REFERENCE_BASELINE_SCORE = 44.8395439

COLAB_WORK_DIR = Path("/content/kculture_kanana15v_paddle_crop")
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

print("ZIP 폴더:", DRIVE_ZIP_DIR)
print("결과 저장 폴더:", RUN_OUTPUT_DIR)
print("모델:", MODEL_ID)
print("학습: Shared", SHARED_EPOCHS, "epoch + 유형별", TYPED_EPOCHS, "epoch")
print("OCR 검출:", OCR_DETECTION_MODEL)
print("OCR 인식:", OCR_RECOGNITION_MODEL)
print("입력 방식: OCR 후보 MC·SA만 원본+crop, LA는 원본, OCR 문자열 주입 없음")
""",
)

set_cell(
    notebook,
    7,
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
    raise RuntimeError(
        "Kanana-V 3B 일반 LoRA와 OCR 다중 이미지 학습에는 20GB 이상의 VRAM을 권장합니다. "
        "Colab A100 또는 L4 런타임으로 변경하세요."
    )

print("일반 LoRA 메모리 점검 통과")
""",
)

# 기존 Qwen v2 OCR 캐시를 우선 재사용합니다.
ocr_source = "".join(notebook["cells"][12]["source"])
old_cache_block = r'''OCR_CACHE_PATH = RUN_OUTPUT_DIR / "paddleocr_cache.json"
V1_OCR_CACHE_PATH = (
    DRIVE_PROJECT_DIR / "outputs" / "qwen3vl_8b_lora_paddle_crop_v1"
    / "paddleocr_cache.json"
)

if not OCR_CACHE_PATH.exists() and V1_OCR_CACHE_PATH.exists():
    shutil.copy2(V1_OCR_CACHE_PATH, OCR_CACHE_PATH)
    print("v1 PaddleOCR 캐시를 v2 실행 폴더로 복사했습니다:", OCR_CACHE_PATH)'''
new_cache_block = r'''OCR_CACHE_PATH = RUN_OUTPUT_DIR / "paddleocr_cache.json"
CANDIDATE_OCR_CACHE_PATHS = [
    DRIVE_PROJECT_DIR / "outputs" / "qwen3vl_8b_lora_paddle_crop_v2_selective" / "paddleocr_cache.json",
    DRIVE_PROJECT_DIR / "outputs" / "qwen3vl_8b_lora_paddle_crop_v1" / "paddleocr_cache.json",
]

if not OCR_CACHE_PATH.exists():
    for candidate_path in CANDIDATE_OCR_CACHE_PATHS:
        if candidate_path.exists():
            shutil.copy2(candidate_path, OCR_CACHE_PATH)
            print("기존 PaddleOCR 캐시를 Kanana 실행 폴더로 복사했습니다:", candidate_path)
            break'''
if old_cache_block not in ocr_source:
    raise RuntimeError("OCR 캐시 교체 대상을 찾지 못했습니다.")
ocr_source = ocr_source.replace(old_cache_block, new_cache_block, 1)
set_cell(notebook, 13, ocr_source)

set_cell(
    notebook,
    16,
    r"""
### 6. 문항 유형별 프롬프트와 Kanana conversation
""",
)

set_cell(
    notebook,
    17,
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


def build_conversation(record, image_count, answer=None, add_answer=False, retry=False):
    # Kanana 공식 형식: 실제 PIL 이미지는 sample['image'], 대화에는 같은 수의 <image> 토큰을 둡니다.
    # OCR 인식 문자열은 대화에 추가하지 않습니다.
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": " ".join(["<image>"] * image_count)},
        {"role": "user", "content": build_user_text(record, retry=retry)},
    ]
    if add_answer:
        conversation.append({"role": "assistant", "content": str(answer)})
    return conversation


for form in ["MC", "SA", "LA"]:
    example = next(row for row in records["train"] if row["metadata"]["question_form"] == form)
    print("=" * 80)
    print(form, example["metadata"]["question_id"])
    print(build_user_text(example))
    print("정답:", example["model_output"]["answer"])
""",
)

set_cell(
    notebook,
    18,
    r"""
### 7. Kanana-1.5-V 3B BF16/FP16 로드
""",
)

set_cell(
    notebook,
    19,
    r"""
from transformers import AutoModelForVision2Seq, AutoProcessor


compute_dtype = torch.bfloat16 if bf16_supported else torch.float16

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
processor.tokenizer.padding_side = "right"
if processor.tokenizer.pad_token_id is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token


def load_base_model():
    return AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        dtype=compute_dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )


def set_model_cache(enabled, target_model=None):
    target = target_model if target_model is not None else model
    if hasattr(target.config, "use_cache"):
        target.config.use_cache = enabled
    if hasattr(target.config, "text_config"):
        target.config.text_config.use_cache = enabled
    language_model = getattr(target, "language_model", None)
    if language_model is not None and hasattr(language_model, "config"):
        language_model.config.use_cache = enabled


model = load_base_model()
set_model_cache(not DO_TRAIN)
torch.set_float32_matmul_precision("high")

print("모델 로드 완료:", MODEL_ID)
print("학습 방식: 일반 LoRA (양자화 없음)")
print("원본 모델 dtype:", compute_dtype)
print("모델 입력 장치:", next(model.parameters()).device)
""",
)

set_cell(
    notebook,
    20,
    r"""
### 8. 이미지 로더·응답 정규화·Kanana 추론 함수
""",
)

set_cell(
    notebook,
    21,
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


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


INFERENCE_USES_TYPED_ADAPTERS = False
ACTIVE_ADAPTER_NAME = None


def activate_adapter(form):
    global ACTIVE_ADAPTER_NAME
    if not INFERENCE_USES_TYPED_ADAPTERS:
        return
    if ACTIVE_ADAPTER_NAME != form:
        model.language_model.set_adapter(form)
        ACTIVE_ADAPTER_NAME = form


@torch.inference_mode()
def generate_raw(record, split, retry=False):
    form = record["metadata"]["question_form"]
    activate_adapter(form)
    images = build_input_images(record, split, INFERENCE_MAX_IMAGE_SIDE)
    sample = {
        "image": images,
        "conv": build_conversation(record, len(images), retry=retry),
    }
    inputs = processor.batch_encode_collate(
        [sample],
        padding_side="left",
        add_generation_prompt=True,
        max_length=MAX_SEQUENCE_LENGTH,
    )
    device = next(model.parameters()).device
    inputs = move_to_device(inputs, device)
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens_for(form),
        temperature=0,
        top_p=1.0,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    # Kanana-V는 이미지 입력을 inputs_embeds로 변환하므로 generate 결과가 새 토큰만 포함됩니다.
    raw_text = processor.tokenizer.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    del images, sample, inputs, generated
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

set_cell(
    notebook,
    24,
    r"""
### 9. Shared LoRA 및 유형별 추가 학습 준비
""",
)

set_cell(
    notebook,
    25,
    r"""
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import Dataset


class MultimodalQADataset(Dataset):
    def __init__(self, split_records, max_samples=None, question_form=None):
        rows = list(split_records)
        if question_form is not None:
            rows = [row for row in rows if row["metadata"]["question_form"] == question_form]
        if max_samples is not None:
            rows = rows[:max_samples]
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class KananaVLCollator:
    def __init__(self, processor, split, max_length, max_image_side):
        self.processor = processor
        self.split = split
        self.max_length = max_length
        self.max_image_side = max_image_side

    def __call__(self, features):
        if len(features) != 1:
            raise RuntimeError("Kanana-V collator는 per_device_train_batch_size=1을 전제로 합니다.")

        record = features[0]
        images = build_input_images(record, self.split, self.max_image_side)
        answer = record["model_output"]["answer"]

        full_sample = {
            "image": images,
            "conv": build_conversation(
                record, len(images), answer=answer, add_answer=True
            ),
        }
        prompt_sample = {
            "image": images,
            "conv": build_conversation(record, len(images), add_answer=False),
        }

        full_batch = self.processor.batch_encode_collate(
            [full_sample],
            padding_side="right",
            add_generation_prompt=False,
            max_length=self.max_length,
        )
        prompt_batch = self.processor.batch_encode_collate(
            [prompt_sample],
            padding_side="right",
            add_generation_prompt=True,
            max_length=self.max_length,
        )

        labels = full_batch["input_ids"].clone()
        labels[full_batch["attention_mask"] == 0] = -100
        labels[full_batch["input_ids"] < 0] = -100
        prompt_length = int(prompt_batch["attention_mask"].sum().item())
        labels[:, :prompt_length] = -100
        if torch.all(labels == -100):
            raise RuntimeError("정답 토큰이 학습 label에 포함되지 않았습니다.")

        full_batch["labels"] = labels
        return full_batch


train_dataset = MultimodalQADataset(records["train"], TRAIN_MAX_SAMPLES)
typed_train_datasets = {
    form: MultimodalQADataset(records["train"], question_form=form)
    for form in ["MC", "SA", "LA"]
}
train_collator = KananaVLCollator(
    processor=processor,
    split="train",
    max_length=MAX_SEQUENCE_LENGTH,
    max_image_side=TRAIN_MAX_IMAGE_SIDE,
)


def available_language_lora_targets(language_model):
    preferred = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    targets = [
        name
        for name, module in language_model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.rsplit(".", 1)[-1] in preferred
        and "lm_head" not in name
    ]
    if not targets:
        raise RuntimeError("Kanana language_model의 LoRA target module을 찾지 못했습니다.")
    return targets


target_modules = available_language_lora_targets(model.language_model)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=target_modules,
)

if DO_TRAIN:
    for parameter in model.parameters():
        parameter.requires_grad = False
    # Kanana-V 전체가 아니라 내부 Llama language_model만 PEFT로 감쌉니다.
    # 원본 Kanana 멀티모달 forward/generate와 vision encoder는 그대로 유지됩니다.
    model.language_model = get_peft_model(model.language_model, lora_config)
    model.language_model.print_trainable_parameters()
    print("학습 방식: Shared LoRA → MC·SA·LA 유형별 LoRA")
    print("LoRA target 수:", len(target_modules))
    print("유형별 학습 문항 수:", {form: len(ds) for form, ds in typed_train_datasets.items()})
else:
    print("DO_TRAIN=False: 저장된 MC·SA·LA Adapter로 추론합니다.")
""",
)

set_cell(
    notebook,
    26,
    r"""
### 학습 전 loss 스모크 테스트
""",
)

set_cell(
    notebook,
    27,
    r"""
if DO_TRAIN:
    model.train()
    one_batch = train_collator([train_dataset[0]])
    device = next(model.parameters()).device
    one_batch = move_to_device(one_batch, device)
    smoke_output = model(**one_batch)
    print("학습 loss 스모크 테스트:", float(smoke_output.loss))
    smoke_output.loss.backward()
    grad_count = sum(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if grad_count == 0:
        raise RuntimeError("LoRA parameter에 gradient가 생성되지 않았습니다.")
    model.zero_grad(set_to_none=True)
    print("gradient가 확인된 trainable tensor 수:", grad_count)
    del one_batch, smoke_output
    gc.collect()
    torch.cuda.empty_cache()
else:
    print("추론 모드이므로 학습 loss 테스트를 건너뜁니다.")
""",
)

set_cell(
    notebook,
    28,
    r"""
### 10. Shared LoRA 학습 후 MC·SA·LA Adapter 분화
""",
)

set_cell(
    notebook,
    29,
    r"""
from transformers import Trainer, TrainingArguments


def adapter_ready(path):
    return (path / "adapter_config.json").exists()


def run_training_stage(stage_model, dataset, stage_name, epochs):
    checkpoint_dir = CHECKPOINT_ROOT / stage_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    set_model_cache(False, stage_model)

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        logging_steps=10,
        # 전체 Kanana-V를 checkpoint로 저장하면 수 GB가 되므로 단계 완료 후 LoRA만 저장합니다.
        save_strategy="no",
        bf16=bf16_supported,
        fp16=not bf16_supported,
        gradient_checkpointing=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        report_to="none",
        seed=SEED,
    )
    trainer = Trainer(
        model=stage_model,
        args=training_args,
        train_dataset=dataset,
        data_collator=train_collator,
    )
    result = trainer.train()
    return trainer, result.metrics


stage_metrics = {}

if DO_TRAIN:
    # 1단계: 전체 데이터 Shared Adapter
    if REUSE_COMPLETED_ADAPTERS and adapter_ready(SHARED_ADAPTER_DIR):
        print("완료된 Shared Adapter를 재사용합니다:", SHARED_ADAPTER_DIR)
        stage_metrics["shared"] = {"reused": True}
    else:
        trainer, metrics = run_training_stage(
            model, train_dataset, "shared", SHARED_EPOCHS
        )
        model.language_model.save_pretrained(str(SHARED_ADAPTER_DIR))
        processor.save_pretrained(str(SHARED_ADAPTER_DIR))
        stage_metrics["shared"] = metrics
        print("Shared Adapter 저장:", SHARED_ADAPTER_DIR)
        del trainer

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # 2단계: Shared Adapter 가중치에서 시작해 유형별로 독립 추가 학습
    for form in ["MC", "SA", "LA"]:
        form_dir = FORM_ADAPTER_DIRS[form]
        if REUSE_COMPLETED_ADAPTERS and adapter_ready(form_dir):
            print(f"완료된 {form} Adapter를 재사용합니다:", form_dir)
            stage_metrics[form] = {"reused": True}
            continue

        typed_model = load_base_model()
        typed_model.language_model = PeftModel.from_pretrained(
            typed_model.language_model,
            str(SHARED_ADAPTER_DIR),
            is_trainable=True,
        )
        trainer, metrics = run_training_stage(
            typed_model,
            typed_train_datasets[form],
            f"typed_{form.lower()}",
            TYPED_EPOCHS,
        )
        typed_model.language_model.save_pretrained(str(form_dir))
        stage_metrics[form] = metrics
        print(f"{form} Adapter 저장:", form_dir)

        del trainer, typed_model
        gc.collect()
        torch.cuda.empty_cache()

    train_metrics_path = RUN_OUTPUT_DIR / "train_metrics.json"
    train_metrics_path.write_text(
        json.dumps(stage_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("학습 지표 저장:", train_metrics_path)
else:
    missing = [str(path) for path in FORM_ADAPTER_DIRS.values() if not adapter_ready(path)]
    if missing:
        raise FileNotFoundError("저장된 유형별 Adapter가 없습니다: " + " | ".join(missing))
    del model
    gc.collect()
    torch.cuda.empty_cache()


# validation/test에서 문항 유형에 따라 Adapter만 빠르게 전환합니다.
model = load_base_model()
model.language_model = PeftModel.from_pretrained(
    model.language_model,
    str(FORM_ADAPTER_DIRS["MC"]),
    adapter_name="MC",
    is_trainable=False,
)
for form in ["SA", "LA"]:
    model.language_model.load_adapter(
        str(FORM_ADAPTER_DIRS[form]),
        adapter_name=form,
        is_trainable=False,
    )

model.eval()
set_model_cache(True, model)
INFERENCE_USES_TYPED_ADAPTERS = True
ACTIVE_ADAPTER_NAME = None
gc.collect()
torch.cuda.empty_cache()

print("유형별 Adapter 추론 준비 완료:", list(FORM_ADAPTER_DIRS))
""",
)

# 제출 파일명과 설명을 Kanana 실험으로 분리합니다.
submission_source = "".join(notebook["cells"][34]["source"])
submission_source = submission_source.replace(
    "submission_qwen3vl_8b_lora_paddle_crop_v2_selective.json",
    "submission_kanana15v_3b_lora_paddle_crop_typed.json",
)
submission_source = submission_source.replace(
    "submission_qwen3vl_8b_lora_paddle_crop_v2_selective.zip",
    "submission_kanana15v_3b_lora_paddle_crop_typed.zip",
)
set_cell(notebook, 35, submission_source)

rag_source = "".join(notebook["cells"][36]["source"])
rag_source = rag_source.replace(
    "qwen3vl_paddle_crop_rag_v1",
    "kanana15v_paddle_crop_rag_v1",
)
set_cell(notebook, 37, rag_source)

set_cell(
    notebook,
    38,
    r"""
### 실행 결과 확인 순서

1. 대표 이미지 OCR의 한글 인식 결과 확인
2. 기존 Qwen OCR 캐시가 복사되었는지 확인
3. 학습 전 loss와 LoRA gradient 스모크 테스트 통과 확인
4. Shared Adapter와 MC·SA·LA Adapter 저장 확인
5. `validation_predictions.csv`에서 유형·OCR 적용 여부별 exact match 확인
6. `submission_kanana15v_3b_lora_paddle_crop_typed.json`을 리더보드에 제출
7. OCR 없는 Kanana 유형별 Adapter 점수 `44.8395439`와 비교
8. 최고점일 때만 `EXPORT_RAG_AFTER_LEADERBOARD=True`로 바꾸고 RAG export 실행

주의: 이 실험은 OCR 문자열을 프롬프트에 넣지 않습니다. OCR은 글자 영역 crop을 선택하는 데만 사용합니다.
""",
)

# 남아 있는 Qwen 전용 표현을 정리하되 Markdown heading 수준은 변경하지 않습니다.
for cell in notebook["cells"]:
    text = "".join(cell["source"])
    text = text.replace("Qwen3-VL 8B", "Kanana-1.5-V 3B")
    cell["source"] = source_lines(text)
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

notebook["metadata"]["colab"]["name"] = OUTPUT_PATH.name
notebook["metadata"]["colab"]["gpuType"] = "A100"

OUTPUT_PATH.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
    encoding="utf-8",
)
print(OUTPUT_PATH)
