from __future__ import annotations

from typing import Dict, List, Tuple

from ..core.skill_graph import SkillGraph
from ..models.schemas import RoadmapStep, SkillGapItem, SkillItem
from .resource_recommender import recommend_resources


def generate_learning_path(
    gaps: List[SkillGapItem],
    graph: SkillGraph,
) -> Tuple[List[SkillItem], List[RoadmapStep]]:
    """
    Build a prerequisite-aware roadmap:
    missing JD skills -> prerequisites (transitive) -> ordered steps.
    """
    missing = [g for g in gaps if g.is_missing]
    missing_ids = [g.jd_skill_id for g in missing]

    expanded_ids = graph.expand_with_prerequisites(missing_ids)  # includes transitive prereqs

    # Target level for each recommended skill:
    # - If it's missing, use JD required_level.
    # - Otherwise, use graph default_level.
    required_level_by_missing: Dict[str, str] = {g.jd_skill_id: g.required_level for g in missing}

    recommended_skills: List[SkillItem] = []
    for sid in expanded_ids:
        level = required_level_by_missing.get(sid) or graph.get_default_level(sid)
        recommended_skills.append(
            SkillItem(
                skill_id=sid,
                name=graph.get_name(sid),
                level=level,
                confidence=0.85,
            )
        )

    # Prerequisite-safe order.
    ordered_ids = graph.topological_sort(expanded_ids)
    id_to_skill = {s.skill_id: s for s in recommended_skills}

    roadmap: List[RoadmapStep] = []
    for i, sid in enumerate(ordered_ids):
        skill = id_to_skill[sid]
        prereqs = graph.prerequisites(sid)
        resources = recommend_resources(skill_id=sid, target_level=skill.level, max_items=2)

        roadmap.append(
            RoadmapStep(
                phase=skill.level,
                step_index=i,
                skill_id=sid,
                skill_name=skill.name,
                target_level=skill.level,
                prerequisites=prereqs,
                resources=resources,
                notes="Targeted skill" if sid in missing_ids else "Prerequisite skill",
            )
        )

    return recommended_skills, roadmap

