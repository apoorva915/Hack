from __future__ import annotations

import os
import logging
from typing import Any, Dict

from app.services.llm.llm_extractor import GeminiSkillExtractor
from app.services.matcher.keyword_matcher import KeywordMatcher
from app.services.normalizer.skill_normalizer import normalize_skills
from app.services.parser.parser import extract_text_from_pdf
from app.services.skill_extractor import SkillExtractor


logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

_EMPTY = {
    "keyword_skills": [],
    "embedding_skills": [],
    "gemini_skills": [],
    "final_skills": [],
    "final_structured": [],
}


class SkillExtractionPipeline:
    def __init__(self) -> None:
        self.keyword_matcher = KeywordMatcher()
        approved = self.keyword_matcher.approved_skill_list
        self.llm_extractor = GeminiSkillExtractor(approved_skills=approved)
        # Backward-compatible alias in case other modules still use this name.
        self.gemini_extractor = self.llm_extractor
        self.advanced_extractor = SkillExtractor()
        self.enable_embeddings = os.getenv("ENABLE_EMBEDDINGS", "true").lower() == "true"
        logger.info("SkillExtractionPipeline initialized")

    def run(self, file: Any) -> Dict[str, Any]:
        try:
            text = extract_text_from_pdf(file)
            if not text.strip():
                logger.warning("Pipeline: empty PDF text")
                return dict(_EMPTY)

            keyword_skills = self.keyword_matcher.match(text)
            logger.info("Keyword matches=%d", len(keyword_skills))

            embedding_skills: list[str] = []
            advanced_result: Dict[str, Any] = {"skills": []}
            if self.enable_embeddings:
                try:
                    advanced_result = self.advanced_extractor.extract(text)
                    embedding_skills = [
                        str(s["name"]).strip().lower()
                        for s in advanced_result.get("skills", [])
                        if isinstance(s, dict) and s.get("name")
                    ]
                    embedding_skills = sorted(set(embedding_skills))
                except Exception as e:
                    logger.warning("Embedding extraction failed: %s", e)
                    embedding_skills = []
                    advanced_result = {"skills": []}
            logger.info("Embedding matches=%d", len(embedding_skills))

            # Keep LLM vocabulary in sync with matcher (e.g. if matcher reloads later).
            self.llm_extractor.set_approved_skills(self.keyword_matcher.approved_skill_list)
            gemini_skills = self.llm_extractor.extract_skills(text)
            logger.info("Gemini matches=%d", len(gemini_skills))

            all_skills = keyword_skills + embedding_skills + gemini_skills
            final_skills = normalize_skills(all_skills)
            logger.info("Final skills=%d", len(final_skills))

            skill_levels = {
                str(s["name"]).strip().lower(): int(s.get("level", 1))
                for s in advanced_result.get("skills", [])
                if isinstance(s, dict) and s.get("name")
            }
            final_structured = [
                {"name": skill, "level": skill_levels.get(skill, 1)}
                for skill in final_skills
            ]

            logger.info("Pipeline completed successfully")
            return {
                "keyword_skills": keyword_skills,
                "embedding_skills": embedding_skills,
                "gemini_skills": gemini_skills,
                "final_skills": final_skills,
                "final_structured": final_structured,
            }
        except Exception as exc:
            logger.exception("Pipeline failed: %s", exc)
            return dict(_EMPTY)
