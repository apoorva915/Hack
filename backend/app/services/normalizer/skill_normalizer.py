from __future__ import annotations

import difflib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# Transformers can optionally import TensorFlow; on some environments this can crash.
# We force USE_TF=0 and USE_TORCH=1 before importing sentence-transformers.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")


def _load_sentence_transformer() -> Any:
    """
    Lazy import to avoid hard failures at module import-time.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer


logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)


# ----------------------------
# Config
# ----------------------------

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_SIMILARITY_THRESHOLD = 0.6
DEFAULT_FUZZY_RATIO = 0.92

SKILLS_VOCAB_PRIMARY_PATH = Path(__file__).resolve().parents[2] / "data" / "skills.json"
SKILLS_VOCAB_FALLBACK_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "skills.json"

SIMILARITY_THRESHOLD = float(os.getenv("SKILL_NORMALIZER_THRESHOLD", DEFAULT_SIMILARITY_THRESHOLD))
MODEL_NAME = os.getenv("SKILL_NORMALIZER_MODEL", DEFAULT_MODEL_NAME)
FUZZY_RATIO = float(os.getenv("SKILL_NORMALIZER_FUZZY_RATIO", DEFAULT_FUZZY_RATIO))


# ----------------------------
# Helpers: normalization + scoring
# ----------------------------

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_skill_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = _WHITESPACE_RE.sub(" ", s)
    return s


def _compute_context_score(context: str) -> int:
    """
    Deterministic level estimation from context.

    Returns:
      - 3 for "advanced"/"expert"
      - 2 for "intermediate"
      - 1 otherwise
    """
    t = (context or "").lower()
    t_norm = _WHITESPACE_RE.sub(" ", t)

    # High-confidence wording.
    if re.search(r"\b(expert|advanced)\b", t_norm):
        return 3
    if re.search(r"\b(intermediate)\b", t_norm):
        return 2
    return 1


def _deduplicate_skills(
    matched: Sequence[Dict[str, Any]],
    boost_multiple_sources: bool = True,
) -> List[Dict[str, Any]]:
    """
    Deduplicate normalized skills.

    If multiple raw skills map to the same canonical skill, keep the highest confidence
    and increase score deterministically based on match count.
    """
    # canonical -> {best_confidence, max_base_score, sources_set, count}
    agg: Dict[str, Dict[str, Any]] = {}

    for m in matched:
        skill = m["skill"]
        if skill not in agg:
            agg[skill] = {
                "best_confidence": float(m["confidence"]),
                "max_base_score": int(m["score"]),
                "sources": set(m.get("sources", [])),
                "count": 0,
            }
        agg[skill]["best_confidence"] = max(agg[skill]["best_confidence"], float(m["confidence"]))
        agg[skill]["max_base_score"] = max(agg[skill]["max_base_score"], int(m["score"]))
        agg[skill]["sources"].update(m.get("sources", []))
        agg[skill]["count"] += 1

    results: List[Dict[str, Any]] = []
    for skill, info in agg.items():
        score = int(info["max_base_score"])
        if boost_multiple_sources and info["count"] > 1:
            # Deterministic boost: add (count-1), clamped to 3.
            score = min(3, score + (info["count"] - 1))

        results.append(
            {
                "skill": skill,
                "score": score,
                "confidence": float(info["best_confidence"]),
                "sources": sorted(info["sources"]),
            }
        )

    # Deterministic ordering: confidence desc, then score desc, then skill asc.
    results.sort(key=lambda d: (-d["confidence"], -d["score"], d["skill"]))
    return results


# ----------------------------
# Embedding + matching engine (singleton)
# ----------------------------


@dataclass(frozen=True)
class _SkillVocab:
    # canonical -> list of aliases (normalized)
    aliases_by_canonical: Dict[str, List[str]]
    # reverse lookup alias -> set(canonical)
    canonicals_by_alias: Dict[str, Set[str]]
    canonical_names: List[str]


class SkillNormalizerEngine:
    def __init__(
        self,
        skills_vocab_path: Path,
        model_name: str = DEFAULT_MODEL_NAME,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        self.skills_vocab_path = skills_vocab_path
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold

        self.vocab: Optional[_SkillVocab] = None
        self.model: Optional[Any] = None
        self.canonical_embeddings: Optional[np.ndarray] = None

        # raw_text -> embedding vector
        self._raw_embedding_cache: Dict[str, np.ndarray] = {}
        self._raw_cache_lock = threading.Lock()

        self._init_lock = threading.Lock()
        self._initialized = False

        self._initialize()

    def _initialize(self) -> None:
        with self._init_lock:
            if self._initialized:
                return

            start = time.perf_counter()
            self.vocab = self._load_skill_dictionary()
            if not self.vocab.canonical_names:
                # Keep embeddings unset; normalizer will return empty.
                logger.error("Skill vocabulary is empty: %s", str(self.skills_vocab_path))
                self._initialized = True
                return

            # Model: load once.
            logger.info("SkillNormalizerEngine loading model: %s", self.model_name)
            SentenceTransformer = _load_sentence_transformer()
            self.model = SentenceTransformer(self.model_name)
            try:  # pragma: no cover
                self.model.eval()
            except Exception:
                pass

            # Precompute canonical skill embeddings (normalized for cosine similarity).
            canonical_names = self.vocab.canonical_names
            logger.info("Precomputing canonical skill embeddings: %d", len(canonical_names))
            canonical_embs = self.model.encode(
                canonical_names,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self.canonical_embeddings = canonical_embs
            self._initialized = True

            elapsed = time.perf_counter() - start
            logger.info(
                "SkillNormalizerEngine ready: skills=%d threshold=%.3f time=%.2fs",
                len(canonical_names),
                self.similarity_threshold,
                elapsed,
            )

    def _load_skill_dictionary(self) -> _SkillVocab:
        """
        Load skill dictionary from JSON:
          { "python": ["python","py"], "machine learning": ["ml","machine learning"], ... }
        """
        if not self.skills_vocab_path.exists() or self.skills_vocab_path.stat().st_size == 0:
            logger.warning("skills.json missing/empty at %s", str(self.skills_vocab_path))
            return _SkillVocab(aliases_by_canonical={}, canonicals_by_alias={}, canonical_names=[])

        data = json.loads(self.skills_vocab_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return _SkillVocab(aliases_by_canonical={}, canonicals_by_alias={}, canonical_names=[])

        aliases_by_canonical: Dict[str, List[str]] = {}
        canonicals_by_alias: Dict[str, Set[str]] = {}

        for canonical, aliases in data.items():
            canonical_norm = _normalize_skill_text(str(canonical))
            if not canonical_norm:
                continue
            if not isinstance(aliases, list):
                continue

            normalized_aliases: List[str] = []
            for a in aliases:
                a_norm = _normalize_skill_text(str(a))
                if not a_norm:
                    continue
                normalized_aliases.append(a_norm)

            # Always include the canonical itself as an alias.
            if canonical_norm not in normalized_aliases:
                normalized_aliases.append(canonical_norm)

            aliases_by_canonical[canonical_norm] = sorted(set(normalized_aliases))

            for alias in aliases_by_canonical[canonical_norm]:
                canonicals_by_alias.setdefault(alias, set()).add(canonical_norm)

        canonical_names = sorted(aliases_by_canonical.keys())
        return _SkillVocab(
            aliases_by_canonical=aliases_by_canonical,
            canonicals_by_alias=canonicals_by_alias,
            canonical_names=canonical_names,
        )

    def _raw_to_text_for_embedding(self, raw_skill: Dict[str, Any]) -> str:
        name = str(raw_skill.get("name", "") or "")
        context = str(raw_skill.get("context", "") or "")
        # Deterministic combined string for embedding.
        return f"{name}. {context}".strip()

    def _encode_raw_cached(self, raw_text: str) -> np.ndarray:
        """
        Encode a raw skill/context string with caching.
        """
        with self._raw_cache_lock:
            if raw_text in self._raw_embedding_cache:
                return self._raw_embedding_cache[raw_text]

        assert self.model is not None  # engine init ensures this.
        emb = self.model.encode(
            [raw_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        with self._raw_cache_lock:
            self._raw_embedding_cache[raw_text] = emb
        return emb

    def _embedding_match(self, raw_text: str) -> Tuple[Optional[str], float]:
        """
        Return (best_canonical, similarity_score) if above threshold else (None, best_score).
        """
        if self.canonical_embeddings is None or self.vocab is None:
            return None, 0.0

        raw_emb = self._encode_raw_cached(raw_text)  # normalized
        # canonical_embeddings are normalized => cosine similarity == dot product.
        sims = np.dot(self.canonical_embeddings, raw_emb)  # (num_skills,)

        best_idx = int(np.argmax(sims))
        best_skill = self.vocab.canonical_names[best_idx]
        best_sim = float(sims[best_idx])
        if best_sim >= self.similarity_threshold:
            return best_skill, best_sim
        return None, best_sim

    def _rule_match(self, raw_name: str) -> Optional[str]:
        """
        High-precision canonical mapping using:
          - exact match (canonical key)
          - alias match via reverse index
          - optional fuzzy match (bonus)
        """
        raw_norm = _normalize_skill_text(raw_name)
        if not raw_norm or self.vocab is None:
            return None

        # Exact canonical match.
        if raw_norm in self.vocab.aliases_by_canonical:
            return raw_norm

        # Alias exact match.
        if raw_norm in self.vocab.canonicals_by_alias:
            canonicals = self.vocab.canonicals_by_alias[raw_norm]
            # Deterministic choice: smallest canonical name.
            return sorted(canonicals)[0] if canonicals else None

        # Optional fuzzy match (bonus): limited to candidates with token overlap.
        raw_tokens = set(raw_norm.split())
        if not raw_tokens:
            return None

        best: Optional[str] = None
        best_ratio = 0.0
        for canonical in self.vocab.canonical_names:
            cand_tokens = set(canonical.split())
            if not cand_tokens.intersection(raw_tokens):
                continue
            ratio = difflib.SequenceMatcher(None, raw_norm, canonical).ratio()
            if ratio >= FUZZY_RATIO and ratio > best_ratio:
                best_ratio = ratio
                best = canonical
        return best

    def match_skill(self, raw_skill: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Match one raw skill entry to a canonical skill.

        Returns:
          {
            "skill": <canonical>,
            "score": <1|2|3>,
            "confidence": <float>,
            "sources": [<raw_name>]
          }
        """
        if self.vocab is None or not self.vocab.canonical_names:
            return None

        raw_name = str(raw_skill.get("name", "") or "")
        if not raw_name.strip():
            return None

        context = str(raw_skill.get("context", "") or "")
        raw_text = self._raw_to_text_for_embedding(raw_skill)

        # Rule match provides a canonical candidate (high precision).
        rule_canonical = self._rule_match(raw_name)

        # Embedding match provides similarity + confidence.
        emb_canonical, emb_sim = self._embedding_match(raw_text)

        # Decide which canonical to keep.
        selected_canonical: Optional[str] = None
        selected_confidence: float = 0.0

        if rule_canonical is not None:
            # If rule hit, we still prefer embedding confidence if similarity clears threshold;
            # otherwise, fall back to embedding similarity (even if below threshold) as confidence.
            selected_canonical = rule_canonical
            # Use embedding similarity to the canonical: compute if we can.
            selected_confidence = self._confidence_for_canonical(raw_text=raw_text, canonical=rule_canonical)
        elif emb_canonical is not None:
            selected_canonical = emb_canonical
            selected_confidence = emb_sim
        else:
            # Neither rule nor embedding threshold succeeded.
            return None

        # Discard if similarity is very low (safety against fuzzy false positives).
        if selected_confidence < (self.similarity_threshold * 0.5):
            return None

        base_score = _compute_context_score(context=context)
        return {
            "skill": selected_canonical,
            "score": base_score,
            "confidence": float(selected_confidence),
            "sources": [raw_name.strip()],
        }

    def _confidence_for_canonical(self, raw_text: str, canonical: str) -> float:
        """
        Compute cosine similarity between raw_text embedding and a canonical embedding.
        """
        if self.vocab is None or self.canonical_embeddings is None:
            return 0.0
        raw_emb = self._encode_raw_cached(raw_text)
        # Find canonical index deterministically via vocab list.
        try:
            idx = self.vocab.canonical_names.index(canonical)
        except ValueError:
            return 0.0
        sim = float(np.dot(self.canonical_embeddings[idx], raw_emb))
        return sim

    def normalize_and_score_skills(self, raw_skills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize and score a list of raw skill/context entries.
        """
        if not raw_skills:
            return []

        matches: List[Dict[str, Any]] = []
        for rs in raw_skills:
            m = self.match_skill(rs)
            if m is not None:
                matches.append(m)

        if not matches:
            return []

        deduped = _deduplicate_skills(matches)
        # Deterministic limit: keep top-N if provided.
        max_out = int(os.getenv("SKILL_NORMALIZER_MAX_SKILLS", "0"))
        if max_out > 0:
            deduped = deduped[:max_out]
        return deduped


# ----------------------------
# Singleton export
# ----------------------------

_ENGINE: Optional[SkillNormalizerEngine] = None
_ENGINE_LOCK = threading.Lock()


def _get_engine() -> SkillNormalizerEngine:
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            skills_path = SKILLS_VOCAB_PRIMARY_PATH if SKILLS_VOCAB_PRIMARY_PATH.exists() else SKILLS_VOCAB_FALLBACK_PATH
            _ENGINE = SkillNormalizerEngine(
                skills_vocab_path=skills_path,
                model_name=MODEL_NAME,
                similarity_threshold=SIMILARITY_THRESHOLD,
            )
        return _ENGINE


# ----------------------------
# Public API (as requested)
# ----------------------------

def load_skill_dictionary() -> Dict[str, List[str]]:
    """
    Load the skill dictionary as:
      canonical -> aliases list
    """
    skills_path = SKILLS_VOCAB_PRIMARY_PATH if SKILLS_VOCAB_PRIMARY_PATH.exists() else SKILLS_VOCAB_FALLBACK_PATH
    if not skills_path.exists() or skills_path.stat().st_size == 0:
        return {}
    data = json.loads(skills_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    cleaned: Dict[str, List[str]] = {}
    for k, v in data.items():
        if not isinstance(k, str) or not isinstance(v, list):
            continue
        canonical = _normalize_skill_text(k)
        if not canonical:
            continue
        cleaned[canonical] = [_normalize_skill_text(str(a)) for a in v if str(a).strip()]
    return cleaned


def build_skill_embeddings() -> Any:
    """
    Force-build embeddings by creating the singleton engine.
    Returns the internal canonical embedding matrix.
    """
    engine = _get_engine()
    return engine.canonical_embeddings


def match_skill(raw_skill: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Match one raw skill to canonical.
    """
    engine = _get_engine()
    return engine.match_skill(raw_skill)


def compute_score(context: str) -> int:
    """
    Deterministic scoring based on context.
    """
    return _compute_context_score(context=context)


def deduplicate_skills(matched: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate matched skills with deterministic score boost.
    """
    return _deduplicate_skills(matched)


def normalize_and_score_skills(raw_skills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize and score raw skills into canonical skills.

    Args:
      raw_skills: List of dicts with shape:
        {"name": <raw name>, "context": <text snippet>}
    """
    if not raw_skills:
        return []

    started = time.perf_counter()
    engine = _get_engine()
    out = engine.normalize_and_score_skills(raw_skills=raw_skills)
    elapsed = time.perf_counter() - started

    logger.info(
        "normalize_and_score_skills: raw=%d matched=%d time=%.3fs",
        len(raw_skills),
        len(out),
        elapsed,
    )
    return out


def normalize_skills(keyword_skills: List[str], gemini_skills: Optional[List[str]] = None) -> List[str]:
    """
    Merge keyword + Gemini outputs: dedupe, lowercase canonical form, sort.
    """
    if gemini_skills is None:
        return normalize_skill_list(keyword_skills)
    return normalize_skill_list(keyword_skills + gemini_skills)


@lru_cache(maxsize=1)
def _alias_to_canonical_map() -> Dict[str, str]:
    out: Dict[str, str] = {}
    d = load_skill_dictionary()
    for canonical, aliases in d.items():
        out[canonical] = canonical
        for alias in aliases:
            out[alias] = canonical
    return out


def normalize_skill_list(skills: List[str]) -> List[str]:
    """
    Normalize a flat skill list:
      - lowercase and strip
      - drop generic filler tokens
      - map aliases to canonical names using the skill dictionary
      - dedupe + sort
    """
    generic_words = {
        "skill",
        "skills",
        "experience",
        "responsibility",
        "responsibilities",
        "requirement",
        "requirements",
        "team",
        "work",
        "project",
        "projects",
    }
    alias_map = _alias_to_canonical_map()
    merged: Set[str] = set()
    for s in skills:
        t = str(s).strip().lower()
        if not t or t in generic_words:
            continue
        merged.add(alias_map.get(t, t))
    result = sorted(merged)
    logger.info("log: normalized final skills count=%d", len(result))
    return result


# ----------------------------
# Basic test
# ----------------------------

if __name__ == "__main__":
    sample = [
        {"name": "PyTorch", "context": "Used PyTorch for advanced deep learning models"},
        {"name": "ML", "context": "Worked on intermediate ML models"},
        {"name": "Python", "context": "Advanced Python development"},
    ]
    print(json.dumps(normalize_and_score_skills(sample), indent=2))

