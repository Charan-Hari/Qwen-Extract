# Qwen-Extract

> Fine-tuning Qwen3 (LoRA) to reliably extract structured JSON from messy resume text — measured, not just demoed.

## What this project does

Most people "use an LLM" by calling an API with a prompt. This project goes further: it
**fine-tunes** a small open-weight model (Qwen3-1.7B, run 100% locally via Ollama/HuggingFace,
no cloud, no API keys, no cost) on the specific task of extracting structured JSON from
free-text resumes, and **proves** the fine-tune helped with a rigorous before/after evaluation.

## The pipeline

1. **Schema definition** — a fixed, moderately rich JSON schema resumes must be extracted into
   (see [`src/qwen_extract/schema.py`](src/qwen_extract/schema.py)).
2. **Synthetic data generation** — since real resumes are hard to source publicly, we generate
   synthetic messy resume text + ground-truth JSON pairs using a local model, then split into
   train/val/test.
3. **Baselines** — Qwen3-1.7B zero-shot and few-shot performance on the test set (no fine-tuning).
4. **LoRA fine-tuning** — Qwen3-1.7B fine-tuned via HuggingFace `peft` on the train split, fully
   on CPU.
5. **Evaluation** — same test set, same metrics (JSON-validity rate, per-field precision/recall/F1)
   across zero-shot vs few-shot vs fine-tuned, so the fine-tune's contribution is measured, not
   assumed.

## Status

🚧 Under active development. See `docs/` for design notes as they land.

## Requirements

- Python 3.10+ (developed against 3.14)
- [Ollama](https://ollama.com) (free, local, no account required) — used for baselines and
  synthetic data generation
- No cloud accounts, no API keys, no credit card — everything runs locally

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
ollama pull qwen3:1.7b
```

## License

MIT
