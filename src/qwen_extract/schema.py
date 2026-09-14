"""
Schema definition for structured resume extraction.

This is the single source of truth for what "correct" output looks like. It is used by:
- the synthetic data generator (to produce ground-truth JSON)
- the baseline prompts (zero-shot / few-shot instructions embed this schema)
- the fine-tuning data formatter (target JSON must conform to this schema)
- the evaluation harness (per-field precision/recall/F1 is computed against this schema)

Keeping the schema in one place means every stage of the pipeline stays consistent.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class Education(BaseModel):
    degree: str = Field(description="e.g. 'B.S. Computer Science'")
    institution: str
    year: Optional[str] = Field(default=None, description="Graduation year, e.g. '2021'")


class Experience(BaseModel):
    title: str
    company: str
    start_date: Optional[str] = Field(default=None, description="e.g. 'Jan 2020'")
    end_date: Optional[str] = Field(default=None, description="e.g. 'Mar 2023' or 'Present'")
    description: Optional[str] = Field(default=None, description="One-line summary of the role")


class Resume(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    skills: list[str] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    total_years_experience: Optional[float] = Field(
        default=None, description="Total years of professional experience, estimated if not stated"
    )


# Flat list of top-level field names, used by the eval harness to iterate per-field metrics.
TOP_LEVEL_FIELDS = [
    "name",
    "email",
    "phone",
    "skills",
    "education",
    "experience",
    "total_years_experience",
]

SCHEMA_JSON_EXAMPLE = {
    "name": "Jane Doe",
    "email": "jane.doe@example.com",
    "phone": "555-123-4567",
    "skills": ["Python", "SQL", "Project Management"],
    "education": [
        {"degree": "B.S. Computer Science", "institution": "State University", "year": "2018"}
    ],
    "experience": [
        {
            "title": "Data Analyst",
            "company": "Acme Corp",
            "start_date": "Jun 2018",
            "end_date": "Present",
            "description": "Built dashboards and automated reporting pipelines.",
        }
    ],
    "total_years_experience": 6.5,
}


def schema_prompt_block() -> str:
    """Human-readable schema description embedded into zero-shot / few-shot prompts."""
    import json

    return (
        "Extract the following fields as a single JSON object. "
        "Use null for missing scalar fields and [] for missing list fields. "
        "Do not include any text outside the JSON object.\n\n"
        f"Schema example:\n{json.dumps(SCHEMA_JSON_EXAMPLE, indent=2)}"
    )
