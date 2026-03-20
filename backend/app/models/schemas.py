from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


SkillLevel = Literal["beginner", "intermediate", "advanced"]
SourceType = Literal["resume", "job_description"]


class SkillItem(BaseModel):
    skill_id: str
    name: str
    level: SkillLevel
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractedSkills(BaseModel):
    source: SourceType
    skills: List[SkillItem]
    meta: Dict[str, Any] = Field(default_factory=dict)


class SkillGapItem(BaseModel):
    jd_skill_id: str
    jd_skill_name: str
    required_level: SkillLevel
    matched_resume_skill_id: Optional[str] = None
    matched_resume_skill_name: Optional[str] = None
    similarity: float = Field(ge=0.0, le=1.0)
    is_missing: bool
    evidence: Dict[str, Any] = Field(default_factory=dict)


class ResourceItem(BaseModel):
    title: str
    link: str
    difficulty: SkillLevel
    estimated_time: str
    tags: List[str] = Field(default_factory=list)


class RecommendationReason(BaseModel):
    skill_id: str
    skill_name: str
    target_level: SkillLevel
    reasons: List[str]
    evidence: Dict[str, Any] = Field(default_factory=dict)


class RoadmapStep(BaseModel):
    phase: SkillLevel  # beginner -> advanced
    step_index: int
    skill_id: str
    skill_name: str
    target_level: SkillLevel
    prerequisites: List[str] = Field(default_factory=list)
    resources: List[ResourceItem] = Field(default_factory=list)
    notes: Optional[str] = None


class UploadResponse(BaseModel):
    analysis_id: str
    resume_text_preview: str
    jd_text_preview: str


class AnalyzeRequest(BaseModel):
    analysis_id: str


class AnalyzeResponse(BaseModel):
    analysis_id: str
    resume: ExtractedSkills
    job_description: ExtractedSkills
    skill_gaps: List[SkillGapItem]
    recommended_skills: List[SkillItem]
    roadmap: List[RoadmapStep]
    reasoning_trace: List[RecommendationReason]


class RoadmapResponse(BaseModel):
    analysis_id: str
    roadmap: List[RoadmapStep]
    reasoning_trace: List[RecommendationReason]

