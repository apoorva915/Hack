from __future__ import annotations

import json
import logging
from typing import List

from app.services.extractor.gemini_client import GeminiClientError, call_gemini


logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

def _build_extraction_prompt(text: str) -> str:
    safe_text = (text or "").strip()
    return f"""You are a skill extraction AI.

Extract skills from text.

Return strict JSON only, with this exact shape:

{{
  "skills": []
}}

Rules:
- Extract only explicitly mentioned skills.
- Do not include explanations.
- If no skills are present, return an empty array.

Text:

{safe_text}
"""


class GeminiSkillExtractor:
    """
    Gemini-backed extractor that extracts skills directly from resume text.
    """

    def __init__(self, approved_skills: List[str] | None = None) -> None:
        _ = approved_skills

    def set_approved_skills(self, skills: List[str]) -> None:
        _ = skills

    def extract_skills(self, text: str, seed_skills: List[str] | None = None) -> List[str]:
        if not text or not text.strip():
            logger.info("log: gemini extraction skipped (empty text)")
            return []
        _ = seed_skills
        prompt = _build_extraction_prompt(text=text)

        try:
            raw = call_gemini(prompt)
            data = json.loads(raw)
        except GeminiClientError as exc:
            logger.error("log: gemini extraction failed: %s", exc)
            return []
        except json.JSONDecodeError as exc:
            logger.error("log: gemini JSON parse error: %s", exc)
            return []
        except Exception as exc:
            logger.exception("log: gemini unexpected error: %s", exc)
            return []

        skills = data.get("skills", [])
        if not isinstance(skills, list):
            logger.warning("log: gemini returned non-list skills field")
            return []

        out: List[str] = []
        seen: set[str] = set()
        for item in skills:
            s = str(item).strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        out.sort(key=str.lower)
        logger.info("log: gemini skills extracted count=%d", len(out))
        return out
