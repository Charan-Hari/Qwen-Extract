"""
LoRA fine-tuning of Qwen3-1.7B (HuggingFace weights, downloaded on first run) on the
resume-extraction task, using the train split produced by build_dataset.py.

This trains the model to go directly from resume text -> JSON matching schema.py,
formatted as a simple instruction-following example per row.

Usage:
    python scripts/train_lora.py --epochs 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from qwen_extract.schema import schema_prompt_block

BASE_MODEL = "Qwen/Qwen3-1.7B"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODEL_OUT_DIR = Path(__file__).resolve().parent.parent / "models" / "qwen3-1.7b-resume-lora"


def build_training_text(resume_text: str, ground_truth: dict, tokenizer) -> str:
    """Format one example as an instruction-following chat turn ending in the target JSON."""
    prompt = f"{schema_prompt_block()}\n\nResume text:\n{resume_text}\n\nJSON:"
    target = json.dumps(ground_truth)
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": target},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def load_split(split: str) -> list[dict]:
    path = DATA_DIR / f"{split}.jsonl"
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def main(epochs: int, lr: float, batch_size: int):
    print(f"Loading base model/tokenizer: {BASE_MODEL} (first run will download weights)...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_examples = load_split("train")
    print(f"Loaded {len(train_examples)} training examples")

    texts = [build_training_text(ex["text"], ex["ground_truth"], tokenizer) for ex in train_examples]

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=1024, padding="max_length")

    dataset = Dataset.from_dict({"text": texts})
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(MODEL_OUT_DIR / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        use_cpu=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    train_result = trainer.train()
    print(train_result)

    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MODEL_OUT_DIR))
    tokenizer.save_pretrained(str(MODEL_OUT_DIR))
    print(f"Saved LoRA adapter to {MODEL_OUT_DIR}")

    # Persist the loss curve for the writeup/eval report
    log_history = trainer.state.log_history
    with open(MODEL_OUT_DIR / "training_log.json", "w", encoding="utf-8") as f:
        json.dump(log_history, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    main(args.epochs, args.lr, args.batch_size)
