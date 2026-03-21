from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from app.services.extractor.gemini_client import GeminiClientError, call_gemini
from app.services.extractor.prompt_builder import build_skill_extraction_prompt
from app.services.extractor.response_parser import parse_llm_response
from app.services.models.skill_schema import SkillExtractionResponse


logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)


class SkillExtractor:
    """
    Orchestrates LLM-based skill extraction.

    Flow:
      build prompt -> call Gemini -> parse response -> validate schema -> return structured data
    """

    def __init__(self) -> None:
        # Keep a callable dependency for clean architecture and testability.
        self._gemini_caller = call_gemini
        logger.info("SkillExtractor initialized.")

    def extract_skills(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract structured skills from raw resume/JD text.

        Returns:
          {"skills": [...]}
        On failure:
          {"skills": []}
        """
        if not text or not text.strip():
            logger.warning("extract_skills called with empty text.")
            return {"skills": []}

        try:
            logger.info("Building extraction prompt.")
            prompt = build_skill_extraction_prompt(text=text)

            logger.info("Calling Gemini for skill extraction.")
            raw_response = self._gemini_caller(prompt=prompt)

            logger.info("Parsing Gemini response.")
            parsed: SkillExtractionResponse = parse_llm_response(raw_response)
            return parsed.model_dump()
        except GeminiClientError as exc:
            logger.error("Gemini API failure: %s", str(exc))
            return {"skills": []}
        except Exception as exc:
            logger.error("Unexpected extraction error: %s", str(exc))
            return {"skills": []}


def extract_from_resume(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convenience wrapper for resume text extraction.
    """
    return SkillExtractor().extract_skills(text=text)


def extract_from_jd(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convenience wrapper for job description text extraction.
    """
    return SkillExtractor().extract_skills(text=text)


if __name__ == "__main__":
    sample_text = """
    John Doe is a Data Scientist with:

    - Python (Advanced)
    - SQL
    - Machine Learning
    - TensorFlow
    - AWS
    """

    extractor = SkillExtractor()
    skills = extractor.extract_skills(sample_text)
    print(json.dumps(skills, indent=2))

