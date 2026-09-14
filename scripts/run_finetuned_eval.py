"""
Runs the fine-tuned LoRA-adapted Qwen3 model on the test set (loaded locally via
HuggingFace transformers, not Ollama) and scores it with the same eval_metrics.py used
for the baselines, so results are directly comparable.

Usage:
    python scripts/run_finetuned_eval.py --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen_extract.eval_metrics import EvalResult, extract_json, score_example
from qwen_extract.schema import schema_prompt_block

BASE_MODEL = "Qwen/Qwen3-1.7B"
ADAPTER_DIR = Path(__file__).resolve().parent.parent / "models" / "qwen3-1.7b-resume-lora"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval" / "results"


def load_model():
    print(f"Loading base model {BASE_MODEL} + LoRA adapter from {ADAPTER_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR))
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, resume_text: str, max_new_tokens: int = 512) -> str:
    prompt = f"{schema_prompt_block()}\n\nResume text:\n{resume_text}\n\nJSON:"
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def run_eval(split: str) -> dict:
    model, tokenizer = load_model()

    path = DATA_DIR / f"{split}.jsonl"
    examples = [json.loads(line) for line in open(path, encoding="utf-8")]

    result = EvalResult()
    per_example_output = []

    for i, ex in enumerate(examples):
        print(f"[{i + 1}/{len(examples)}] running fine-tuned model...", flush=True)
        raw_output = generate(model, tokenizer, ex["text"])
        pred_json = extract_json(raw_output)
        score_example(pred_json, ex["ground_truth"], result)
        per_example_output.append(
            {"text": ex["text"], "ground_truth": ex["ground_truth"], "raw_output": raw_output, "parsed": pred_json}
        )

    summary = result.summary()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"finetuned_{split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "examples": per_example_output}, f, indent=2)

    print(f"\n=== fine-tuned on {split} ===")
    print(json.dumps(summary, indent=2))
    print(f"Saved to {out_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    run_eval(args.split)
