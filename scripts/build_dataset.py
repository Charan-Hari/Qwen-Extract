"""
CLI script to generate the full synthetic dataset: ground truth + textified prose,
split into train/val/test, written to data/processed/.

Usage:
    python scripts/build_dataset.py --n 10 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qwen_extract.generate_ground_truth import generate_batch
from qwen_extract.textify import textify

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def build_dataset(n: int, seed: int, splits: tuple[float, float, float] = (0.7, 0.15, 0.15)) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    resumes = generate_batch(n, seed=seed)

    examples = []
    for i, resume in enumerate(resumes):
        print(f"[{i + 1}/{n}] textifying...", flush=True)
        for attempt in range(3):
            try:
                text = textify(resume)
                break
            except Exception as e:  # noqa: BLE001 - retry on any transient network/timeout error
                print(f"  attempt {attempt + 1} failed: {e}", flush=True)
                if attempt == 2:
                    raise
        examples.append({"text": text, "ground_truth": resume.model_dump()})

    n_train = int(n * splits[0])
    n_val = int(n * splits[1])

    train = examples[:n_train]
    val = examples[n_train:n_train + n_val]
    test = examples[n_train + n_val:]

    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = DATA_DIR / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for ex in split:
                f.write(json.dumps(ex) + "\n")
        print(f"Wrote {len(split)} examples to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Total number of examples to generate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_dataset(args.n, args.seed)
