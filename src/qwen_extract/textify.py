"""
Turns a structured, ground-truth Resume object into messy, realistic free-text resume
prose using a local Ollama model. The LLM here only rephrases/formats — it never invents
facts — so the text should always be reducible back to the exact ground-truth JSON.

This is the "reverse" direction of the task we ultimately fine-tune Qwen3 to perform
(text -> JSON), used purely to build a training/eval corpus.
"""

from __future__ import annotations

import json
import re

import requests

from qwen_extract.schema import Resume

OLLAMA_URL = "http://localhost:11434/api/generate"
TEXTIFY_MODEL = "qwen3:1.7b"

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Qwen3 emits <think>...</think> reasoning blocks; strip them from the final text."""
    return _THINK_TAG_RE.sub("", text).strip()


def _build_prompt(resume: Resume) -> str:
    data = resume.model_dump()
    return (
        "You are generating realistic, messy resume text for a dataset. "
        "Given the structured resume data below, write the resume as free-form prose "
        "the way a real person might write it — inconsistent formatting, some fields "
        "combined into sentences, some abbreviations, no markdown, no JSON. "
        "Do NOT add any facts beyond what's given. Do NOT include a <think> block, "
        "just output the resume text directly.\n\n"
        f"Structured data:\n{json.dumps(data, indent=2)}\n\n"
        "Resume text:"
    )


def textify(resume: Resume, timeout: int = 300) -> str:
    """Call local Ollama model to turn a Resume object into messy prose text."""
    prompt = _build_prompt(resume) + " /no_think"
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": TEXTIFY_MODEL,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {"temperature": 0.8},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    raw = response.json()["response"]
    return _strip_thinking(raw)
