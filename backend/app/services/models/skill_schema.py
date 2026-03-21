from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


ALLOWED_LEVELS = {"beginner", "intermediate", "advanced", "unknown"}


class Skill(BaseModel):
    """Normalized skill item returned by the LLM extraction pipeline."""

    skill_name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    level: str
    years_experience: Optional[float] = None
    evidence_text: str = Field(min_length=1)
    confidence: float

    @field_validator("skill_name", "category", "evidence_text", mode="before")
    @classmethod
    def _strip_required_strings(cls, value: object) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("Field cannot be empty.")
        return text

    @field_validator("level", mode="before")
    @classmethod
    def _validate_level(cls, value: object) -> str:
        level = str(value).strip().lower()
        if level not in ALLOWED_LEVELS:
            raise ValueError(f"level must be one of {sorted(ALLOWED_LEVELS)}")
        return level

    @field_validator("confidence")
    @classmethod
    def _validate_confidence(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        return float(value)


class SkillExtractionResponse(BaseModel):
    """Top-level extraction response model."""

    skills: List[Skill] = Field(default_factory=list)

