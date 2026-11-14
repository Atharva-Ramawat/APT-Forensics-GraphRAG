"""
Phase 1: Embedding Fine-Tuning for Cybersecurity Domain Adaptation
Model: BAAI/bge-small-en-v1.5
Target GPU: RTX 3050 (6GB VRAM)
"""

import json
import os
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

# ---------------------------------------------------------------------------
# 1. Hardware & Hyperparameter Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "BAAI/bge-small-en-v1.5"
DATA_PATH = "cyber_training_data.jsonl"
OUTPUT_DIR = "./cyber-bge-small"
HF_HUB_REPO = "athv/cyber-bge-small"

# Strict batch size constraint for RTX 3050 (6GB VRAM) to prevent CUDA OOM
BATCH_SIZE = 16  # Reduce to 8 if encountering any memory spikes
NUM_EPOCHS = 3
WARMUP_RATIO = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Running training on device: {DEVICE}")
if DEVICE == "cuda":
    print(f"[*] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[*] Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

# ---------------------------------------------------------------------------
# 2. Data Ingestion
# ---------------------------------------------------------------------------
train_examples = []

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Training dataset not found at: {DATA_PATH}")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line_idx, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            anchor = record["anchor"]
            positive = record["positive"]
            train_examples.append(InputExample(texts=[anchor, positive]))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[!] Warning: Skipping invalid record at line {line_idx}: {e}")

print(f"[*] Successfully loaded {len(train_examples)} training pairs.")

# ---------------------------------------------------------------------------
# 3. Model & Loss Setup
# ---------------------------------------------------------------------------
print(f"[*] Loading base model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME, device=DEVICE)

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
train_loss = losses.MultipleNegativesRankingLoss(model=model)

warmup_steps = int(len(train_dataloader) * NUM_EPOCHS * WARMUP_RATIO)

# ---------------------------------------------------------------------------
# 4. Contrastive Fine-Tuning Loop
# ---------------------------------------------------------------------------
print("[*] Starting contrastive training...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=NUM_EPOCHS,
    warmup_steps=warmup_steps,
    use_amp=True,  # Automatic Mixed Precision (FP16) enabled for 6GB VRAM efficiency
    output_path=OUTPUT_DIR,
    show_progress_bar=True
)

print(f"[*] Fine-tuning complete. Model saved locally to: {OUTPUT_DIR}")

# ---------------------------------------------------------------------------
# 5. Push Fine-Tuned Weights to Hugging Face Hub
# ---------------------------------------------------------------------------
try:
    print(f"[*] Publishing fine-tuned model to Hugging Face Hub: {HF_HUB_REPO}")
    model.push_to_hub(
        repo_id=HF_HUB_REPO,
        private=False,
        commit_message="Initial release of cyber-bge-small fine-tuned on cybersecurity contrastive pairs"
    )
    print(f"[+] Successfully pushed model to: https://huggingface.co/{HF_HUB_REPO}")
except Exception as e:
    print(f"[!] Push to Hugging Face Hub failed or requires authentication: {e}")
    print("[!] Run `huggingface-cli login` in your terminal, then re-run the push step.")
