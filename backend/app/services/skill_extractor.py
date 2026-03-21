
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# sentence-transformers may indirectly import TensorFlow on this environment and crash.
# Force USE_TF=0 and USE_TORCH=1 before importing it.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")


def _load_sentence_transformer() -> Any:
    """
    Lazy import to avoid hard failures at module import-time.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer


logger = logging.getLogger(__name__)

# Ensure logs emit even if the host app hasn't configured logging yet.
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)


# ----------------------------
# Config (bonus: env overrides)
# ----------------------------

DEFAULT_SIMILARITY_THRESHOLD = 0.6
DEFAULT_MAX_SKILLS = 25
SKILLS_VOCAB_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "processed" / "skills.json"
)

SIMILARITY_THRESHOLD = float(os.getenv("SKILL_SIMILARITY_THRESHOLD", DEFAULT_SIMILARITY_THRESHOLD))
MAX_SKILLS = int(os.getenv("SKILL_MAX_SKILLS", DEFAULT_MAX_SKILLS))


# ----------------------------
# Utilities: text preprocessing
# ----------------------------

def split_into_sentences(text: str) -> List[str]:
    """
    Split resume/job-description text into normalized sentences.

    Splits on '.' and newlines, lowercases, strips, and removes empty results.
    """

    if not text:
        return []

    # Normalize whitespace a bit; keep it deterministic.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    parts: List[str] = []
    for chunk in text.split("\n"):
        # Split on period as requested.
        parts.extend(chunk.split("."))

    sentences: List[str] = []
    for p in parts:
        s = p.strip().lower()
        if s:
            sentences.append(s)
    return sentences


def _normalize_for_matching(s: str) -> str:
    # Lowercase and collapse whitespace for consistent alias matching.
    s = s.lower()
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def _is_simple_alias(alias: str) -> bool:
    """
    Whether alias is "regex-safe" to use word boundaries for.

    We only use word boundaries when alias is composed of letters/digits/spaces.
    """

    alias = alias.strip().lower()
    return bool(re.fullmatch(r"[a-z0-9 ]+", alias))


def _build_alias_regex(alias: str) -> Optional[re.Pattern]:
    """
    Build a compiled regex for high precision matching.

    If alias contains special characters (e.g., 'c++'), we fall back to substring matching.
    """

    alias = _normalize_for_matching(alias)
    if not alias:
        return None
    if not _is_simple_alias(alias):
        return None

    # Convert 'machine learning' -> r'\bmachine\s+learning\b'
    tokens = alias.split()
    if not tokens:
        return None

    pattern = r"\b" + r"\s+".join(map(re.escape, tokens)) + r"\b"
    return re.compile(pattern, flags=re.IGNORECASE)


# ----------------------------
# Skill extraction engine
# ----------------------------

@dataclass(frozen=True)
class _AliasPattern:
    canonical_skill: str
    alias: str
    regex: Optional[re.Pattern]


class SkillExtractionEngine:
    """
    Production-style skill extraction engine.

    Uses a hybrid approach:
      - Rule-based alias matching (high precision)
      - Embedding-based semantic matching (high recall)
      - Simple deterministic skill level estimation
    """

    def __init__(
        self,
        skills_vocab_path: Path,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.6,
        max_skills: int = 25,
    ) -> None:
        self.skills_vocab_path = skills_vocab_path
        self.similarity_threshold = similarity_threshold
        self.max_skills = max_skills

        self.skill_aliases: Dict[str, List[str]] = {}
        self.alias_patterns: List[_AliasPattern] = []

        self.skill_names: List[str] = []
        self._skill_embedding_matrix: Optional[np.ndarray] = None
        self._skill_embedding_norms_ready: bool = False

        self.model: Any = None

        # Thread-safety: encoding uses torch and may not be safe under concurrency.
        self._encode_lock = threading.Lock()

        self._initialize()

    def _initialize(self) -> None:
        start = time.perf_counter()

        self.skill_aliases = self._load_skills_vocab(self.skills_vocab_path)
        if not self.skill_aliases:
            logger.error(
                "Skill vocabulary is empty or missing. Expected file at %s",
                str(self.skills_vocab_path),
            )
            # Keep the engine "disabled" (deterministically return empty results).
            self.skill_names = []
            self._skill_embedding_matrix = None
            return

        # Precompute rule-based alias patterns.
        alias_seen: Set[Tuple[str, str]] = set()
        for canonical, aliases in self.skill_aliases.items():
            canonical = canonical.strip().lower()
            if not canonical:
                continue
            for alias in aliases or []:
                a = _normalize_for_matching(str(alias))
                if not a:
                    continue
                key = (canonical, a)
                if key in alias_seen:
                    continue
                alias_seen.add(key)
                self.alias_patterns.append(
                    _AliasPattern(
                        canonical_skill=canonical,
                        alias=a,
                        regex=_build_alias_regex(a),
                    )
                )

        self.skill_names = sorted(self.skill_aliases.keys())

        # Load model once and precompute skill embeddings once.
        # If embeddings fail to initialize, rule-based extraction still works.
        try:
            logger.info("Loading SentenceTransformer model: %s", "all-MiniLM-L6-v2")
            SentenceTransformer = _load_sentence_transformer()
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            # Disable dropout/ensure eval mode if supported.
            try:  # pragma: no cover
                self.model.eval()
            except Exception:
                pass

            skills_list = self.skill_names
            # normalize_embeddings=True => cosine similarity is dot product
            with self._encode_lock:
                skill_embs = self.model.encode(
                    skills_list,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            self._skill_embedding_matrix = skill_embs
            self._skill_embedding_norms_ready = True
        except Exception as e:  # pragma: no cover
            logger.error("Embedding model initialization failed; rule-based extraction only: %s", str(e))
            self.model = None
            self._skill_embedding_matrix = None
            self._skill_embedding_norms_ready = False

        elapsed = time.perf_counter() - start
        logger.info(
            "SkillExtractionEngine initialized: skills=%d, threshold=%.3f, max_skills=%d, time=%.3fs",
            len(self.skill_names),
            self.similarity_threshold,
            self.max_skills,
            elapsed,
        )

    @staticmethod
    def _load_skills_vocab(skills_vocab_path: Path) -> Dict[str, List[str]]:
        try:
            if not skills_vocab_path.exists() or skills_vocab_path.stat().st_size == 0:
                return {}
            raw = skills_vocab_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if not isinstance(data, dict):
                return {}
            # Normalize keys and alias lists.
            normalized: Dict[str, List[str]] = {}
            for skill, aliases in data.items():
                if not isinstance(skill, str):
                    continue
                if not isinstance(aliases, list):
                    continue
                canonical = _normalize_for_matching(skill)
                if not canonical:
                    continue
                normalized[canonical] = [str(a).strip().lower() for a in aliases if str(a).strip()]
            return normalized
        except Exception as e:  # pragma: no cover
            logger.exception("Failed to load skills vocabulary from %s: %s", str(skills_vocab_path), str(e))
            return {}

    def is_vocab_disabled(self) -> bool:
        return not self.skill_names

    def is_embeddings_ready(self) -> bool:
        return self._skill_embedding_matrix is not None and self.model is not None

    def detect_skills_rule_based(self, sentences: List[str]) -> Set[str]:
        """
        High-precision alias matching.

        If an alias appears in a sentence => mark the canonical skill.
        """

        if self.is_vocab_disabled():
            return set()

        detected: Set[str] = set()
        for sentence in sentences:
            # sentence already lowercased by split_into_sentences; normalize spaces for alias matching.
            s = _normalize_for_matching(sentence)

            for ap in self.alias_patterns:
                if ap.regex is not None:
                    if ap.regex.search(s):
                        detected.add(ap.canonical_skill)
                        continue
                # Substring fallback (handles non-word characters like 'c++')
                else:
                    if ap.alias in s:
                        detected.add(ap.canonical_skill)
        return detected

    @lru_cache(maxsize=2048)
    def _encode_sentence_cached(self, sentence: str) -> np.ndarray:
        # Encodes a single sentence; caching helps across repeated requests.
        with self._encode_lock:
            emb = self.model.encode(
                [sentence],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return emb[0]

    def detect_skills_embedding(self, sentences: List[str]) -> Dict[str, float]:
        """
        High-recall embedding matching.

        Returns:
            dict: skill -> best_similarity_score (max over sentences)
        """

        if not self.is_embeddings_ready():
            return {}

        if not sentences:
            return {}

        skill_scores: Dict[str, float] = {name: 0.0 for name in self.skill_names}

        # For each sentence, compute similarities to all skills.
        # Embeddings are normalized => cosine similarity is dot product.
        skill_matrix = self._skill_embedding_matrix
        assert skill_matrix is not None  # for type-checkers

        # Ensure deterministic processing order.
        for sentence in sentences:
            s = _normalize_for_matching(sentence)
            if not s:
                continue
            sent_emb = self._encode_sentence_cached(s)
            sims = np.dot(skill_matrix, sent_emb)  # shape: (num_skills,)

            # Track max similarity per skill.
            # Use deterministic loop order by skill_names.
            for idx, skill_name in enumerate(self.skill_names):
                sim = float(sims[idx])
                # Track best similarity for ranking even if below threshold.
                if sim > skill_scores[skill_name]:
                    skill_scores[skill_name] = sim

        # Return max similarity for all skills; caller applies thresholding/selection.
        return skill_scores

    @staticmethod
    def estimate_skill_level_from_signals(text: str, skill: str, aliases: List[str]) -> int:
        """
        Estimate deterministic skill level based on simple keyword and pattern heuristics.

        IMPORTANT: `text` is expected to be a *context snippet* around the skill mention,
        not the full resume/JD text. This keeps "advanced/expert" rules localized.
        """

        t = text.lower()
        # Collapse whitespace to make multi-word alias matching robust to newlines.
        t_norm = re.sub(r"\s+", " ", t).strip()

        def alias_in_text(a: str) -> bool:
            a_norm = re.sub(r"\s+", " ", _normalize_for_matching(a)).strip()
            return bool(a_norm) and a_norm in t_norm

        skill_present = any(alias_in_text(a) for a in aliases if a)
        if not skill_present:
            # Even if called for a detected skill, keep deterministic fallback.
            return 1

        # High precision overrides.
        if re.search(r"\b(expert|advanced)\b", t):
            return 3
        if re.search(r"\bintermediate\b", t):
            return 2

        # Years patterns: "3 years of python" / "2 yrs python"
        # Note: we check each alias.
        years_matches: List[int] = []
        for alias in aliases:
            if not alias or not str(alias).strip():
                continue

            alias_norm = re.sub(r"\s+", " ", _normalize_for_matching(alias)).strip()
            if not alias_norm:
                continue

            # Allow flexible whitespace for multi-word aliases.
            tokens = alias_norm.split()
            if _is_simple_alias(alias_norm) and len(tokens) > 1:
                alias_regex = r"\b" + r"\s+".join(map(re.escape, tokens)) + r"\b"
            else:
                alias_regex = re.escape(alias_norm)

            # 'of' is optional
            m = re.search(rf"(\d+)\s*(?:\+)?\s*(?:years?|yrs?)\s+(?:of\s+)?{alias_regex}", t_norm)
            if m:
                years_matches.append(int(m.group(1)))

        if years_matches:
            # Deterministic: take max years mentioned.
            years = max(years_matches)
            if years >= 5:
                return 3
            if years >= 3:
                return 2
            return 1

        # Strong/proficient patterns (bonus heuristic).
        if "strong" in t_norm or "proficient" in t_norm or "extensive" in t_norm:
            return 2

        return 1

    def _find_first_alias_index(self, text: str, aliases: List[str]) -> int:
        """
        Find first occurrence index of any alias (normalized whitespace).
        Returns -1 if no alias is found.
        """
        # Normalize whitespace in the haystack for deterministic matching.
        hay = re.sub(r"\s+", " ", (text or "").lower()).strip()
        if not hay:
            return -1

        best_idx = -1
        for a in aliases:
            if not a:
                continue
            a_norm = re.sub(r"\s+", " ", _normalize_for_matching(a)).strip()
            if not a_norm:
                continue
            idx = hay.find(a_norm)
            if idx == -1:
                continue
            if best_idx == -1 or idx < best_idx:
                best_idx = idx
        return best_idx

    def estimate_skill_level(self, text: str, skill: str) -> int:
        """
        Estimate skill level using a context window around the first alias mention.
        """
        aliases = self.skill_aliases.get(skill, [skill])

        idx = self._find_first_alias_index(text=text, aliases=aliases)
        if idx == -1:
            context_snippet = text
        else:
            # Use the original `text` slice (not the normalized one) for more natural keyword matches.
            # This is deterministic even if whitespace differs slightly.
            window = 220
            start = max(0, idx - window)
            end = min(len(text), idx + window)
            context_snippet = text[start:end]

        return self.estimate_skill_level_from_signals(
            text=context_snippet,
            skill=skill,
            aliases=aliases,
        )

    def extract_skills(self, text: str) -> Dict[str, Any]:
        """
        Main extraction function.

        Returns:
            {"skills": [{"name": <skill>, "level": <1|2|3>}, ...]}
        """

        started = time.perf_counter()

        if not text or not text.strip():
            return {"skills": []}

        sentences = split_into_sentences(text)
        if not sentences:
            return {"skills": []}

        if self.is_vocab_disabled():
            # Vocabulary missing/empty; deterministic empty output.
            return {"skills": []}

        # 1) Rule-based
        rule_detected = self.detect_skills_rule_based(sentences)

        # 2) Embedding-based (best similarity per skill)
        embedding_scores = self.detect_skills_embedding(sentences)
        # Apply thresholding only for "detected" skills; we still keep raw similarities
        # for ranking of rule-based hits.
        embedding_detected = {
            k for k, v in embedding_scores.items() if v >= self.similarity_threshold
        }

        # 3) Combine (merge + remove duplicates)
        combined: Set[str] = set(rule_detected) | set(embedding_detected)

        if not combined:
            logger.info("Skill extraction finished: skills=0, time=%.3fs", time.perf_counter() - started)
            return {"skills": []}

        # Score map for deterministic top-N ranking.
        # - Rule-based skills get a fixed high score so they rank above "weak embedding" matches.
        # - Embedding scores use similarity.
        skill_scores: Dict[str, float] = {}
        for s in combined:
            if s in rule_detected:
                # Rule-based hits are high precision; break ties using embedding similarity.
                sim = float(embedding_scores.get(s, 0.0))
                sim = max(0.0, sim)
                skill_scores[s] = 2.0 + sim
            else:
                # Embedding-based detected skills.
                sim = float(embedding_scores.get(s, 0.0))
                skill_scores[s] = sim

        # Deterministic ordering:
        # 1) ranking score (rule-based vs embedding)
        # 2) local skill level (context-aware)
        # 3) skill name
        levels_cache: Dict[str, int] = {}
        for s in combined:
            levels_cache[s] = self.estimate_skill_level(text=text, skill=s)

        ranked = sorted(
            combined,
            key=lambda k: (-skill_scores.get(k, 0.0), -levels_cache.get(k, 1), k),
        )

        # Apply top-N limit when explicitly configured.
        # Convention: <= 0 => no limit.
        if self.max_skills and self.max_skills > 0:
            ranked = ranked[: self.max_skills]

        results: List[Dict[str, Any]] = []
        for skill in ranked:
            level = self.estimate_skill_level(text=text, skill=skill)
            results.append({"name": skill, "level": int(level)})

        elapsed = time.perf_counter() - started
        logger.info("Skill extraction finished: skills=%d, time=%.3fs", len(results), elapsed)
        return {"skills": results}


# ----------------------------
# Engine singleton + exported API
# ----------------------------

_ENGINE: Optional[SkillExtractionEngine] = None
_ENGINE_LOCK = threading.Lock()


def _get_engine() -> SkillExtractionEngine:
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = SkillExtractionEngine(
                skills_vocab_path=SKILLS_VOCAB_PATH,
                similarity_threshold=SIMILARITY_THRESHOLD,
                max_skills=MAX_SKILLS,
            )
        return _ENGINE


class SkillExtractor:
    """
    Pipeline-facing facade around the module-level singleton engine.
    """

    def extract(self, text: str) -> dict:
        return extract_skills(text=text)


def extract_skills(text: str) -> dict:
    """
    Public service-layer function.

    Deterministic hybrid skill extraction using the dataset-driven vocabulary from:
      app/data/processed/skills.json
    """

    try:
        engine = _get_engine()
        return engine.extract_skills(text=text)
    except Exception as e:  # pragma: no cover
        logger.exception("Skill extraction failed: %s", str(e))
        return {"skills": []}


def estimate_skill_level(text: str, skill: str) -> int:
    """
    Estimate a single skill level for a given skill name.

    Required helper by spec; delegates to the singleton engine heuristics.
    """

    try:
        engine = _get_engine()
        return int(engine.estimate_skill_level(text=text, skill=skill.strip().lower()))
    except Exception as e:  # pragma: no cover
        logger.exception("Skill level estimation failed: %s", str(e))
        return 1

