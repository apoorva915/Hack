from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Set, Tuple

import regex


logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

_SKILLS_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "final_skills.json"


class KeywordMatcher:
    """
    Match resume text against the processed skill dictionary (canonical + aliases).
    """

    def __init__(self) -> None:
        self._phrase_to_canonical: list[Tuple[str, str]] = []
        self._all_phrases: list[str] = []

        if not _SKILLS_PATH.exists():
            logger.warning("Skill dictionary missing: %s", _SKILLS_PATH)
            return

        with _SKILLS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            logger.warning("final_skills.json is not a dict; skipping matcher init.")
            return

        seen_pairs: Set[Tuple[str, str]] = set()
        for canonical, aliases in data.items():
            c = str(canonical).strip().lower()
            if not c:
                continue
            phrases: Set[str] = {c}
            if isinstance(aliases, list):
                for a in aliases:
                    p = str(a).strip().lower()
                    if p:
                        phrases.add(p)
            for p in phrases:
                key = (p, c)
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    self._phrase_to_canonical.append((p, c))

        # Longer phrases first to prefer multi-word skills over substrings.
        self._phrase_to_canonical.sort(key=lambda x: len(x[0]), reverse=True)
        self._all_phrases = sorted({c for _, c in self._phrase_to_canonical})

        logger.info("KeywordMatcher loaded phrases=%d canonicals=%d", len(self._phrase_to_canonical), len(self._all_phrases))

    @property
    def approved_skill_list(self) -> List[str]:
        """Flattened canonical skills for LLM grounding."""
        return list(self._all_phrases)

    def match(self, text: str) -> List[str]:
        if not text or not text.strip() or not self._phrase_to_canonical:
            logger.info("log: keyword matches count=0")
            return []

        haystack = text.lower()
        matched_canonicals: Set[str] = set()

        for phrase, canonical in self._phrase_to_canonical:
            pattern = rf"(?<!\w){regex.escape(phrase)}(?!\w)"
            if regex.search(pattern, haystack, flags=regex.IGNORECASE):
                matched_canonicals.add(canonical)

        result = sorted(matched_canonicals)
        logger.info("log: keyword matches count=%d", len(result))
        return result
