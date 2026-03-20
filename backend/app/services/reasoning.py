from __future__ import annotations

from typing import Dict, List, Set

from ..core.skill_graph import SkillGraph
from ..models.schemas import RecommendationReason, SkillGapItem, SkillItem, SkillLevel


def _is_prerequisite_transitively(graph: SkillGraph, maybe_prereq: str, skill_id: str, seen: Set[str]) -> bool:
    if skill_id in seen:
        return False
    seen.add(skill_id)
    for prereq in graph.prerequisites(skill_id):
        if prereq == maybe_prereq:
            return True
        if _is_prerequisite_transitively(graph, maybe_prereq, prereq, seen):
            return True
    return False


def generate_reasoning_trace(
    gaps: List[SkillGapItem],
    recommended_skills: List[SkillItem],
    graph: SkillGraph,
) -> List[RecommendationReason]:
    jd_missing = {g.jd_skill_id for g in gaps if g.is_missing}
    jd_all_required = {g.jd_skill_id for g in gaps}

    # Best effort lookup for gap evidence.
    gap_by_id: Dict[str, SkillGapItem] = {g.jd_skill_id: g for g in gaps}

    recommended_ids = [s.skill_id for s in recommended_skills]
    reasons: List[RecommendationReason] = []

    for rec in recommended_skills:
        skill_id = rec.skill_id
        target_level: SkillLevel = rec.level

        rec_reasons: List[str] = []
        evidence: Dict[str, object] = {}

        if skill_id in jd_missing:
            g = gap_by_id.get(skill_id)
            rec_reasons.append("Missing and required by the job description")
            if g is not None:
                evidence["jd_required_level"] = g.required_level
                evidence["resume_similarity"] = g.similarity
                evidence["resume_match"] = {
                    "skill_id": g.matched_resume_skill_id,
                    "skill_name": g.matched_resume_skill_name,
                }
        elif skill_id in jd_all_required:
            # Present in JD but not missing; recommend because target level might be higher.
            g = gap_by_id.get(skill_id)
            if g and not g.is_missing:
                rec_reasons.append("Job-required skill is partially covered; roadmap targets a stronger level")
                evidence["resume_similarity"] = g.similarity
                evidence["resume_match"] = {
                    "skill_id": g.matched_resume_skill_id,
                    "skill_name": g.matched_resume_skill_name,
                }
            else:
                rec_reasons.append("Aligned with job requirements")
        else:
            # Prerequisite skills not explicitly listed in JD.
            dependent = [
                missing_id
                for missing_id in jd_missing
                if _is_prerequisite_transitively(graph, skill_id, missing_id, seen=set())
            ]
            if dependent:
                rec_reasons.append("Added as a prerequisite to unlock missing job skills")
                evidence["unlocks_missing_skills"] = dependent[:5]
            else:
                rec_reasons.append("Recommended to strengthen the learning foundation")

        reasons.append(
            RecommendationReason(
                skill_id=skill_id,
                skill_name=rec.name,
                target_level=target_level,
                reasons=rec_reasons,
                evidence=evidence,
            )
        )

    # Stable ordering: by target phase/level then confidence.
    order = {"beginner": 0, "intermediate": 1, "advanced": 2}
    reasons.sort(key=lambda r: (order.get(r.target_level, 0), r.skill_id))
    return reasons

