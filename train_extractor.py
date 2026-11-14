"""
Phase 2: Unsloth 4-bit QLoRA Instruction Fine-Tuning for Telemetry Knowledge Graph Extraction
Model: Qwen/Qwen2.5-Coder-3B
Target GPU: RTX 3050 (6GB VRAM)
"""

import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ---------------------------------------------------------------------------
# 1. Hardware & Hyperparameter Configuration (Strict 6GB VRAM Ceiling)
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-Coder-3B"
DATA_PATH = "telemetry_graph_pairs.json"
OUTPUT_DIR = "./cyber-extractor-qlora"

MAX_SEQ_LENGTH = 512
LOAD_IN_4BIT = True
LORA_R = 16
LORA_ALPHA = 16
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4

print(f"[*] Checking GPU capability...")
if torch.cuda.is_available():
    print(f"[*] Device: {torch.cuda.get_device_name(0)}")
    print(f"[*] VRAM Available: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    print("[!] Warning: CUDA is not available. Fine-tuning with Unsloth requires an NVIDIA GPU.")

# ---------------------------------------------------------------------------
# 2. Model & Tokenizer Initialization with FastLanguageModel
# ---------------------------------------------------------------------------
print(f"[*] Loading base model '{MODEL_NAME}' in 4-bit precision...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # Automatically detects bfloat16 or float16
    load_in_4bit=LOAD_IN_4BIT,
)

# ---------------------------------------------------------------------------
# 3. Apply QLoRA Parameter Efficient Adapters
# ---------------------------------------------------------------------------
print("[*] Applying QLoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,               # 0 is optimized for Unsloth fast kernels
    bias="none",
    use_gradient_checkpointing="unsloth",  # Unsloth checkpointing drastically lowers VRAM consumption
    random_state=3407,
    max_seq_length=MAX_SEQ_LENGTH,
)

# ---------------------------------------------------------------------------
# 4. Dataset Ingestion & Alpaca Formatting
# ---------------------------------------------------------------------------
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input_text, output_text in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input_text, output_text) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

print(f"[*] Loading dataset from: {DATA_PATH}")
dataset = load_dataset("json", data_files=DATA_PATH, split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

# ---------------------------------------------------------------------------
# 5. SFTTrainer Execution
# ---------------------------------------------------------------------------
print("[*] Initializing SFTTrainer with conservative memory settings...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=1,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        warmup_steps=2,
        max_steps=30,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="./training_outputs",
        report_to="none"
    ),
)

print("[*] Starting extraction model fine-tuning...")
trainer_stats = trainer.train()
print(f"[*] Training finished in: {trainer_stats.metrics['train_runtime']:.2f}s")

# ---------------------------------------------------------------------------
# 6. Save Model Artifacts
# ---------------------------------------------------------------------------
print(f"[*] Saving adapter weights and tokenizer to: {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"[+] QLoRA adapters successfully saved at: {OUTPUT_DIR}")
