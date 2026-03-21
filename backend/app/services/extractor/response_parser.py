from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from app.services.models.skill_schema import SkillExtractionResponse


logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)


_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _strip_markdown_code_fences(text: str) -> str:
    match = _CODE_FENCE_RE.search(text or "")
    if match:
        return match.group(1).strip()
    return (text or "").strip()


def _cleanup_common_json_issues(candidate_text: str) -> str:
    """
    Apply lightweight cleanup for common LLM JSON mistakes.

    This remains conservative so we do not silently alter semantics.
    """
    cleaned = (candidate_text or "").strip()
    # Remove trailing commas before closing braces/brackets.
    cleaned = _TRAILING_COMMA_RE.sub(r"\1", cleaned)
    return cleaned


def _extract_json_object(candidate_text: str) -> Optional[str]:
    """
    Extract the first balanced JSON object from noisy text.
    """
    text = candidate_text or ""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for idx in range(start, len(text)):
        ch = text[idx]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]

    # Partial/truncated output.
    return None


def parse_llm_response(response_text: str) -> SkillExtractionResponse:
    """
    Parse Gemini raw response text into a validated SkillExtractionResponse.

    Handles:
    - markdown code fences
    - extra text before/after JSON
    - partial or invalid JSON
    """
    if not response_text or not response_text.strip():
        logger.error("LLM response was empty.")
        return SkillExtractionResponse(skills=[])

    try:
        cleaned = _strip_markdown_code_fences(response_text)
        json_blob = _extract_json_object(cleaned)
        if not json_blob:
            logger.error("Could not extract JSON object from LLM response.")
            return SkillExtractionResponse(skills=[])

        payload_text = _cleanup_common_json_issues(json_blob)
        payload: Dict[str, Any] = json.loads(payload_text)
        parsed = SkillExtractionResponse.model_validate(payload)
        return parsed
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON from LLM response: %s", str(exc))
        return SkillExtractionResponse(skills=[])
    except Exception as exc:
        logger.error("Failed to parse/validate LLM response: %s", str(exc))
        return SkillExtractionResponse(skills=[])

