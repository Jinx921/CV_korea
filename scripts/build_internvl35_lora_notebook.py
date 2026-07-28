from copy import deepcopy
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "notebooks" / "internvl35_8b_baseline.ipynb"
OUTPUT_PATH = ROOT / "notebooks" / "internvl35_8b_lora_colab.ipynb"


if not SOURCE_PATH.exists():
    raise FileNotFoundError(
        f"먼저 QLoRA 기준 노트북을 생성하세요: {SOURCE_PATH}"
    )

nb = deepcopy(nbf.read(SOURCE_PATH, as_version=4))
nb.metadata.setdefault("colab", {})["name"] = OUTPUT_PATH.name
nb.metadata["colab"]["gpuType"] = "A100"


def find_code_cell(marker: str):
    matches = [
        cell
        for cell in nb.cells
        if cell.cell_type == "code" and marker in cell.source
    ]
    if len(matches) != 1:
        raise RuntimeError(f"코드 셀 marker={marker!r} 후보 수: {len(matches)}")
    return matches[0]


for cell in nb.cells:
    if cell.cell_type != "markdown":
        continue
    cell.source = (
        cell.source.replace("Q-LoRA", "LoRA")
        .replace("QLoRA", "LoRA")
        .replace("4-bit 로드", "BF16/FP16 로드")
        .replace("4-bit로 로드", "BF16/FP16로 로드")
        .replace("internvl35_8b_qlora_v1", "internvl35_8b_lora_v1")
    )


install_cell = find_code_cell("%pip uninstall -q -y gradio gradio_client")
install_cell.source = install_cell.source.replace(
    '  "bitsandbytes>=0.45" \\\n', ""
)


config_cell = find_code_cell('RUN_NAME = "internvl35_8b_qlora_v1"')
config_cell.source = config_cell.source.replace(
    'RUN_NAME = "internvl35_8b_qlora_v1"',
    'RUN_NAME = "internvl35_8b_lora_v1"\nTRAINING_METHOD = "lora_bf16"',
)
config_cell.source = config_cell.source.replace(
    "# True: train 1,000건 QLoRA 학습 / False: 제로샷 베이스라인",
    "# True: train 1,000건 일반 LoRA 학습 / False: 제로샷 베이스라인",
)


gpu_cell = find_code_cell("gpu_mem_gb < 20")
old_gpu_warning = '''if DO_TRAIN and gpu_mem_gb < 20:
    print(
        "주의: 8B 이미지 모델 QLoRA는 16GB GPU에서 OOM이 날 수 있습니다. "
        "발생 시 TRAIN_MAX_PATCHES=2, TRAIN_MAX_IMAGE_SIDE=672로 낮추거나 "
        "DO_TRAIN=False로 제로샷 제출을 먼저 만드세요."
    )'''
new_gpu_warning = '''if DO_TRAIN and gpu_mem_gb < 35:
    raise RuntimeError(
        "InternVL3.5-8B 일반 LoRA는 원본 모델을 BF16/FP16으로 올리므로 "
        "A100 40GB급 GPU가 필요합니다. 현재 VRAM이 부족합니다. "
        "A100 런타임으로 변경하거나 기존 QLoRA 노트북을 사용하세요."
    )

print("일반 LoRA 메모리 점검 통과")'''
if old_gpu_warning not in gpu_cell.source:
    raise RuntimeError("GPU 경고 블록을 찾지 못했습니다.")
gpu_cell.source = gpu_cell.source.replace(old_gpu_warning, new_gpu_warning)


model_cell = find_code_cell("BitsAndBytesConfig")
model_cell.source = r'''
from transformers import AutoProcessor

try:
    from transformers import AutoModelForMultimodalLM as InternVLModelClass
except ImportError:
    from transformers import AutoModelForImageTextToText as InternVLModelClass


compute_dtype = torch.bfloat16 if bf16_supported else torch.float16

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
if getattr(processor, "tokenizer", None) is not None:
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

# 일반 LoRA: 원본 모델을 4-bit로 양자화하지 않고 BF16/FP16으로 로드합니다.
model = InternVLModelClass.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=compute_dtype,
    device_map={"": 0},
    low_cpu_mem_usage=True,
)

model.config.use_cache = not DO_TRAIN
torch.set_float32_matmul_precision("high")

print("모델 로드 완료:", MODEL_ID)
print("학습 방식: 일반 LoRA (원본 모델 양자화 없음)")
print("원본 모델 dtype:", compute_dtype)
print("모델 입력 장치:", next(model.parameters()).device)
'''.strip()


peft_cell = find_code_cell("prepare_model_for_kbit_training")
peft_cell.source = peft_cell.source.replace(
    "from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training",
    "from peft import LoraConfig, PeftModel, get_peft_model",
)

old_peft_block = '''if USE_SAVED_ADAPTER:
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
    print("DO_TRAIN=False: 제로샷 모델을 그대로 사용합니다.")'''

new_peft_block = '''if USE_SAVED_ADAPTER:
    adapter_config_path = ADAPTER_DIR / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"저장된 adapter가 없습니다: {adapter_config_path}")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR, is_trainable=DO_TRAIN)
    print("저장된 일반 LoRA adapter 로드:", ADAPTER_DIR)
elif DO_TRAIN:
    # 일반 LoRA는 k-bit 준비 과정을 사용하지 않습니다.
    # 원본 가중치를 고정하고 언어 모델의 LoRA 행렬만 학습합니다.
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
    print("DO_TRAIN=False: 제로샷 모델을 그대로 사용합니다.")'''

if old_peft_block not in peft_cell.source:
    raise RuntimeError("기존 QLoRA 준비 블록을 찾지 못했습니다.")
peft_cell.source = peft_cell.source.replace(old_peft_block, new_peft_block)


training_cell = find_code_cell('optim="paged_adamw_8bit"')
training_cell.source = (
    training_cell.source.replace('optim="paged_adamw_8bit"', 'optim="adamw_torch"')
    .replace("save_steps=100", "save_steps=10")
)


for cell in nb.cells:
    cell.source = (
        cell.source.replace(
            "submission_internvl35_8b.json", "submission_internvl35_8b_lora.json"
        )
        .replace(
            "submission_internvl35_8b.zip", "submission_internvl35_8b_lora.zip"
        )
        .replace("internvl35_8b_qlora_v1", "internvl35_8b_lora_v1")
    )


all_source = "\n".join(cell.source for cell in nb.cells)
for forbidden in [
    "BitsAndBytesConfig",
    "load_in_4bit",
    "prepare_model_for_kbit_training",
    "paged_adamw_8bit",
]:
    if forbidden in all_source:
        raise RuntimeError(f"일반 LoRA 노트북에 QLoRA 코드가 남았습니다: {forbidden}")

required = [
    'TRAINING_METHOD = "lora_bf16"',
    'RUN_NAME = "internvl35_8b_lora_v1"',
    'optim="adamw_torch"',
    "model.gradient_checkpointing_enable",
    "submission_internvl35_8b_lora.json",
]
for value in required:
    if value not in all_source:
        raise RuntimeError(f"일반 LoRA 필수 설정이 없습니다: {value}")


OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUTPUT_PATH)
print(f"Wrote {OUTPUT_PATH}")
