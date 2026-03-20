from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ..models.schemas import ResourceItem, SkillLevel


def _load_courses_db() -> List[Dict[str, Any]]:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    return json.loads((data_dir / "courses.json").read_text(encoding="utf-8"))


def recommend_resources(skill_id: str, target_level: SkillLevel, max_items: int = 2) -> List[ResourceItem]:
    courses = _load_courses_db()
    matches = [c for c in courses if c.get("skill_id") == skill_id]
    matches.sort(key=lambda c: c.get("difficulty", "beginner"))

    def as_item(c: Dict[str, Any]) -> ResourceItem:
        return ResourceItem(
            title=c.get("title") or skill_id,
            link=c.get("link") or "",
            difficulty=c.get("difficulty") or "beginner",
            estimated_time=c.get("estimated_time") or "N/A",
            tags=c.get("tags") or [],
        )

    # Return what we have; for demo this is sufficient.
    items = [as_item(c) for c in matches][:max_items]

    if items:
        return items

    # Fallback placeholder.
    return [
        ResourceItem(
            title=f"{skill_id}: Learn and practice",
            link="https://www.youtube.com/",
            difficulty=target_level,
            estimated_time="2-6 hours",
            tags=["fallback"],
        )
    ]

