"""
Evaluation harness: given a model's raw text output for a resume, parse it as JSON and
score it against ground truth using per-field precision/recall/F1, plus a JSON-validity rate.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from qwen_extract.schema import TOP_LEVEL_FIELDS

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def extract_json(raw_text: str) -> dict | None:
    """Best-effort extraction of a JSON object from raw model output (strips <think> blocks,
    markdown code fences, and surrounding prose)."""
    text = _THINK_TAG_RE.sub("", raw_text).strip()
    text = text.replace("```json", "").replace("```", "").strip()

    match = _JSON_BLOCK_RE.search(text)
    if not match:
        return None
    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _scalar_match(pred, truth) -> bool:
    if pred is None and truth is None:
        return True
    if pred is None or truth is None:
        return False
    return str(pred).strip().lower() == str(truth).strip().lower()


def _list_of_str_score(pred: list, truth: list) -> tuple[int, int, int]:
    """Returns (true_positives, false_positives, false_negatives) for a list-of-strings field
    (e.g. skills), compared case-insensitively as sets."""
    pred_set = {str(p).strip().lower() for p in (pred or [])}
    truth_set = {str(t).strip().lower() for t in (truth or [])}
    tp = len(pred_set & truth_set)
    fp = len(pred_set - truth_set)
    fn = len(truth_set - pred_set)
    return tp, fp, fn


def _list_of_dict_score(pred: list, truth: list, keys: list[str]) -> tuple[int, int, int]:
    """Compares list-of-dict fields (education, experience) by treating each dict as a
    tuple of its key values and matching as sets (order-independent, exact-match per item)."""
    def to_tuple(d):
        return tuple(str(d.get(k, "")).strip().lower() for k in keys)

    pred_set = {to_tuple(p) for p in (pred or []) if isinstance(p, dict)}
    truth_set = {to_tuple(t) for t in (truth or []) if isinstance(t, dict)}
    tp = len(pred_set & truth_set)
    fp = len(pred_set - truth_set)
    fn = len(truth_set - pred_set)
    return tp, fp, fn


@dataclass
class FieldScore:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, tp: int, fp: int, fn: int) -> None:
        self.tp += tp
        self.fp += fp
        self.fn += fn

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class EvalResult:
    json_valid_count: int = 0
    total_count: int = 0
    field_scores: dict[str, FieldScore] = field(default_factory=lambda: {f: FieldScore() for f in TOP_LEVEL_FIELDS})

    @property
    def json_validity_rate(self) -> float:
        return self.json_valid_count / self.total_count if self.total_count else 0.0

    @property
    def macro_f1(self) -> float:
        scores = [s.f1 for s in self.field_scores.values()]
        return sum(scores) / len(scores) if scores else 0.0

    def summary(self) -> dict:
        return {
            "json_validity_rate": round(self.json_validity_rate, 3),
            "macro_f1": round(self.macro_f1, 3),
            "per_field": {
                name: {
                    "precision": round(s.precision, 3),
                    "recall": round(s.recall, 3),
                    "f1": round(s.f1, 3),
                }
                for name, s in self.field_scores.items()
            },
        }


def score_example(pred_json: dict | None, ground_truth: dict, result: EvalResult) -> None:
    result.total_count += 1
    if pred_json is None:
        # Every field counts as a full miss (fn) if JSON failed to parse at all.
        for f_name in TOP_LEVEL_FIELDS:
            truth_val = ground_truth.get(f_name)
            if f_name == "skills":
                _, _, fn = _list_of_str_score([], truth_val)
                result.field_scores[f_name].add(0, 0, fn)
            elif f_name == "education":
                _, _, fn = _list_of_dict_score([], truth_val, ["degree", "institution", "year"])
                result.field_scores[f_name].add(0, 0, fn)
            elif f_name == "experience":
                _, _, fn = _list_of_dict_score([], truth_val, ["title", "company", "start_date", "end_date"])
                result.field_scores[f_name].add(0, 0, fn)
            else:
                if truth_val is not None:
                    result.field_scores[f_name].add(0, 0, 1)
        return

    result.json_valid_count += 1

    for f_name in TOP_LEVEL_FIELDS:
        pred_val = pred_json.get(f_name)
        truth_val = ground_truth.get(f_name)

        if f_name == "skills":
            tp, fp, fn = _list_of_str_score(pred_val, truth_val)
            result.field_scores[f_name].add(tp, fp, fn)
        elif f_name == "education":
            tp, fp, fn = _list_of_dict_score(pred_val, truth_val, ["degree", "institution", "year"])
            result.field_scores[f_name].add(tp, fp, fn)
        elif f_name == "experience":
            tp, fp, fn = _list_of_dict_score(pred_val, truth_val, ["title", "company", "start_date", "end_date"])
            result.field_scores[f_name].add(tp, fp, fn)
        else:
            match = _scalar_match(pred_val, truth_val)
            if truth_val is None and pred_val is None:
                continue  # not applicable, skip
            if match:
                result.field_scores[f_name].add(1, 0, 0)
            else:
                # wrong/missing prediction counts as both a false positive (if pred given)
                # and a false negative (ground truth was missed)
                fp = 1 if pred_val is not None else 0
                fn = 1 if truth_val is not None else 0
                result.field_scores[f_name].add(0, fp, fn)
