"""
Schema definition for structured resume extraction.

This is the single source of truth for what "correct" output looks like. It is used by:
- the synthetic data generator (to produce ground-truth JSON)
- the baseline prompts (zero-shot / few-shot instructions embed this schema)
- the fine-tuning data formatter (target JSON must conform to this schema)
- the evaluation harness (per-field precision/recall/F1 is computed against this schema)

Keeping the schema in one place means every stage of the pipeline stays consistent.
"""
