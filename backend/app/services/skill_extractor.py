from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config.settings import get_settings
from ..models.schemas import ExtractedSkills, SkillItem, SkillLevel


def _load_skills_db() -> List[Dict[str, Any]]:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    return json.loads((data_dir / "skills_db.json").read_text(encoding="utf-8"))


def _infer_level_from_context(context: str, source: str, default_level: SkillLevel) -> SkillLevel:
    ctx = context.lower()

    # Strong signals
    if any(k in ctx for k in ["expert", "deep", "mastery", "advanced", "senior"]):
        return "advanced"
    if any(k in ctx for k in ["familiar", "baseline", "exposure", "beginner"]):
        return "beginner"

    # Years signal for resume
    if source == "resume":
        m = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*(years|yrs)\b", ctx)
        if m:
            years = float(m.group(1))
            if years >= 4:
                return "advanced"
            if years >= 2:
                return "intermediate"
            return "beginner"

    # Soft signals
    if any(k in ctx for k in ["led", "designed", "built", "shipped", "production", "architect"]):
        return "intermediate"

    return default_level


def _extract_with_openai(text: str, source: str) -> Optional[List[SkillItem]]:
    settings = get_settings()
    if not settings.openai_api_key:
        return None

    # Lazy import so the backend can run without OpenAI installed.
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return None

    skills_db = _load_skills_db()
    allowed_skill_ids = [s["skill_id"] for s in skills_db]

    system = (
        "You are an expert resume and job-description parser. "
        "Return ONLY valid JSON. Output must be a list of skills with fields: "
        "skill_id, name, level, confidence (0-1). "
        f"Allowed skill_id values: {allowed_skill_ids}."
    )

    user = (
        f"Extract relevant skills from this {source.replace('_', ' ')} text for the onboarding roadmap. "
        f"Classify each skill as beginner/intermediate/advanced and provide confidence. "
        "If a skill is not present, do not include it.\n\n"
        f"TEXT:\n{text}\n\n"
        "JSON FORMAT:\n"
        "[{\"skill_id\":\"...\",\"name\":\"...\",\"level\":\"beginner|intermediate|advanced\",\"confidence\":0.0}]"
    )

    client = OpenAI(api_key=settings.openai_api_key)
    last_err: Optional[Exception] = None

    for _attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.1,
            )
            content = resp.choices[0].message.content or "[]"
            parsed = json.loads(content)
            items: List[SkillItem] = []
            for it in parsed:
                items.append(
                    SkillItem(
                        skill_id=str(it["skill_id"]),
                        name=str(it.get("name") or it["skill_id"]),
                        level=str(it["level"]),
                        confidence=float(it.get("confidence", 0.5)),
                    )
                )
            return items
        except Exception as e:
            last_err = e

    # If OpenAI extraction fails, just return None and fall back.
    return None


def extract_skills_from_text(text: str, source: str) -> ExtractedSkills:
    """
    Returns structured extracted skills.
    Uses heuristic skill detection from `skills_db.json`.
    Optionally upgrades extraction using OpenAI if configured.
    """
    text = text or ""
    skills_db = _load_skills_db()
    lower = text.lower()

    # Optional LLM extraction (best-effort, safe fallback).
    try:
        llm_items = _extract_with_openai(text=text, source=source)
        if llm_items:
            # Keep output stable/safe.
            return ExtractedSkills(source=source, skills=llm_items, meta={"method": "llm"})
    except Exception:
        pass

    extracted: List[SkillItem] = []
    for skill in skills_db:
        skill_id = skill["skill_id"]
        name = skill.get("name", skill_id)
        default_level = skill.get("default_level", "beginner")
        synonyms: List[str] = [s.lower() for s in (skill.get("synonyms") or [])]

        best_idx = None
        occurrences = 0
        for syn in synonyms:
            idx = lower.find(syn)
            if idx >= 0:
                occurrences += max(1, lower.count(syn))
                if best_idx is None or idx < best_idx:
                    best_idx = idx

        if occurrences <= 0:
            continue

        # Grab some surrounding text for level inference.
        if best_idx is None:
            context = lower
        else:
            context = lower[max(0, best_idx - 140) : min(len(lower), best_idx + 220)]

        level = _infer_level_from_context(context=context, source=source, default_level=default_level)

        # Map occurrences into a bounded confidence score.
        confidence = min(0.95, 0.25 + 0.07 * occurrences)
        extracted.append(
            SkillItem(
                skill_id=skill_id,
                name=name,
                level=level,
                confidence=float(confidence),
            )
        )

    extracted.sort(key=lambda s: s.confidence, reverse=True)
    return ExtractedSkills(source=source, skills=extracted, meta={"method": "heuristic"})

