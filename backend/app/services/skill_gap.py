from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..config.settings import get_settings
from ..models.schemas import ExtractedSkills, SkillGapItem
from ..utils.embeddings import best_match_via_faiss, embed_texts


def compute_skill_gaps(resume_skills: ExtractedSkills, jd_skills: ExtractedSkills) -> List[SkillGapItem]:
    """
    Compare resume skill coverage vs JD required skills using embeddings + cosine similarity.
    """
    settings = get_settings()

    resume_items = resume_skills.skills
    jd_items = jd_skills.skills

    resume_texts = [f"{s.skill_id}:{s.name}" for s in resume_items]
    jd_texts = [f"{s.skill_id}:{s.name}" for s in jd_items]

    resume_labels = [s.skill_id for s in resume_items]
    try:
        resume_vecs = embed_texts(resume_texts)
        jd_vecs = embed_texts(jd_texts)

        matches = best_match_via_faiss(
            query_vecs=jd_vecs,
            key_vecs=resume_vecs,
            key_labels=resume_labels,
            top_k=max(1, settings.top_k_resume_matches),
        )
    except Exception as e:
        # If embeddings dependencies are missing, fall back to deterministic matching.
        matches = []
        for jd_skill in jd_items:
            sim = 1.0 if jd_skill.skill_id in resume_labels else 0.0
            matched_id = jd_skill.skill_id if sim == 1.0 else (resume_labels[0] if resume_labels else "")
            matches.append([(matched_id, sim)])

    # Build quick lookup for matched skill names.
    resume_name_by_id = {s.skill_id: s.name for s in resume_items}
    resume_level_by_id = {s.skill_id: s.level for s in resume_items}

    gaps: List[SkillGapItem] = []

    for jd_skill, top_matches in zip(jd_items, matches):
        if not top_matches:
            gaps.append(
                SkillGapItem(
                    jd_skill_id=jd_skill.skill_id,
                    jd_skill_name=jd_skill.name,
                    required_level=jd_skill.level,
                    is_missing=True,
                    similarity=0.0,
                    evidence={"top_matches": []},
                )
            )
            continue

        # Use best similarity to decide missing.
        best_id, best_sim = top_matches[0]
        matched_resume_id = best_id if best_sim > 0 else None
        is_missing = best_sim < settings.similarity_threshold

        gaps.append(
            SkillGapItem(
                jd_skill_id=jd_skill.skill_id,
                jd_skill_name=jd_skill.name,
                required_level=jd_skill.level,
                matched_resume_skill_id=matched_resume_id,
                matched_resume_skill_name=resume_name_by_id.get(matched_resume_id) if matched_resume_id else None,
                similarity=best_sim,
                is_missing=is_missing,
                evidence={
                    "top_matches": [
                        {"skill_id": sid, "skill_name": resume_name_by_id.get(sid), "similarity": sim}
                        for sid, sim in top_matches
                    ],
                    "matched_resume_level": resume_level_by_id.get(matched_resume_id) if matched_resume_id else None,
                },
            )
        )

    return gaps

