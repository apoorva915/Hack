from __future__ import annotations

import logging


logger = logging.getLogger(__name__)


def build_skill_extraction_prompt(text: str) -> str:
    """
    Build the exact prompt for Gemini skill extraction.

    The structure intentionally mirrors the requested template so downstream
    parsing remains deterministic.
    """
    safe_text = (text or "").strip()
    logger.info("Building skill extraction prompt for text length=%d", len(safe_text))
    return f"""You are an expert AI system that extracts professional skills 
from resumes and job descriptions.

STRICT RULES:

1. Extract only explicitly mentioned skills
2. Do NOT infer missing skills
3. Return only valid JSON
4. No explanations

OUTPUT FORMAT:

{{
  "skills": [
    {{
      "skill_name": "",
      "category": "",
      "level": "",
      "years_experience": null,
      "evidence_text": "",
      "confidence": 0.0
    }}
  ]
}}

TEXT:
{safe_text}

"""

