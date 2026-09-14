"""
Runs zero-shot and few-shot Qwen3 (via Ollama) extraction on the test set and reports
JSON-validity rate + per-field precision/recall/F1, using eval_metrics.py.

Usage:
    python scripts/run_baseline_eval.py --split test --mode zero_shot
    python scripts/run_baseline_eval.py --split test --mode few_shot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qwen_extract.eval_metrics import EvalResult, extract_json, score_example
from qwen_extract.schema import SCHEMA_JSON_EXAMPLE, schema_prompt_block

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval" / "results"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:1.7b"

FEW_SHOT_EXAMPLE_TEXT = (
    "Mark Reyes works as a Product Manager at Delta Systems, a role he's held since "
    "March 2019. He holds a B.S. in Computer Science from Lakeview University, class of "
    "2015. His skill set includes Python, Agile, and Data Analysis. He can be reached at "
    "mreyes@example.com or 555-201-4488. He has about 8 years of total experience."
)
FEW_SHOT_EXAMPLE_JSON = {
    "name": "Mark Reyes",
    "email": "mreyes@example.com",
    "phone": "555-201-4488",
    "skills": ["Python", "Agile", "Data Analysis"],
    "education": [{"degree": "B.S. Computer Science", "institution": "Lakeview University", "year": "2015"}],
    "experience": [
        {
            "title": "Product Manager",
            "company": "Delta Systems",
            "start_date": "Mar 2019",
            "end_date": "Present",
            "description": None,
        }
    ],
    "total_years_experience": 8,
}


def build_prompt(resume_text: str, mode: str) -> str:
    header = schema_prompt_block()
    if mode == "zero_shot":
        return f"{header}\n\nResume text:\n{resume_text}\n\nJSON:"
    elif mode == "few_shot":
        return (
            f"{header}\n\n"
            f"Example resume text:\n{FEW_SHOT_EXAMPLE_TEXT}\n\n"
            f"Example JSON:\n{json.dumps(FEW_SHOT_EXAMPLE_JSON, indent=2)}\n\n"
            f"Now extract from this resume text:\n{resume_text}\n\nJSON:"
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


def call_model(prompt: str, timeout: int = 300) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt + " /no_think",
            "stream": False,
            "think": False,
            "options": {"temperature": 0.0},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["response"]


def run_eval(split: str, mode: str) -> dict:
    path = DATA_DIR / f"{split}.jsonl"
    examples = [json.loads(line) for line in open(path, encoding="utf-8")]

    result = EvalResult()
    per_example_output = []

    for i, ex in enumerate(examples):
        print(f"[{i + 1}/{len(examples)}] running {mode}...", flush=True)
        prompt = build_prompt(ex["text"], mode)
        raw_output = call_model(prompt)
        pred_json = extract_json(raw_output)
        score_example(pred_json, ex["ground_truth"], result)
        per_example_output.append(
            {"text": ex["text"], "ground_truth": ex["ground_truth"], "raw_output": raw_output, "parsed": pred_json}
        )

    summary = result.summary()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"baseline_{mode}_{split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "examples": per_example_output}, f, indent=2)

    print(f"\n=== {mode} on {split} ===")
    print(json.dumps(summary, indent=2))
    print(f"Saved to {out_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--mode", choices=["zero_shot", "few_shot"], required=True)
    args = parser.parse_args()

    run_eval(args.split, args.mode)
