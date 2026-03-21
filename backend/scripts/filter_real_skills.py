#!/usr/bin/env python3
"""
Second-pass filtering for a noisy skill ontology.

Input:
  cleaned_skills.json (mapping canonical_skill -> [aliases])

Output:
  final_skills.json (same structure, but only real professional skills)

This script is purely rule-based (no LLM, no embeddings).
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

try:
    import nltk  # type: ignore
    from nltk import pos_tag  # type: ignore

    # We don't automatically download corpora during normal runs; if the tagger isn't
    # available, we'll fall back to pure heuristics.
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger.zip")
        _POS_TAGGER_AVAILABLE = True
    except LookupError:
        _POS_TAGGER_AVAILABLE = False
except Exception:
    _POS_TAGGER_AVAILABLE = False

logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Never remove: explicit real-tech whitelist.
# ---------------------------------------------------------------------------
TECH_WHITELIST: Set[str] = {
    "python",
    "java",
    "javascript",
    "sql",
    "aws",
    "docker",
    "kubernetes",
    "linux",
    "react",
    "angular",
    "django",
    "flask",
    "node",
    "git",
    "tensorflow",
    "pytorch",
    "pandas",
    "numpy",
    "tableau",
    "power bi",
    "mysql",
    "postgres",
    "mongodb",
    "jira",
    "salesforce",
    "sap",
    "quickbooks",
    "oracle",
    # Commonly-real variants / related systems
    "gcp",
    "azure",
    "azure devops",
}


VERB_SUFFIX_RE = re.compile(r"(ed|ing)$", re.IGNORECASE)
TECH_NUMBER_RE = re.compile(r"\d")
HTML_CSS_VERSION_RE = re.compile(r"\b(html|css)\s*\d+(?:\.\d+)?\b", re.IGNORECASE)
ACRONYM_LIKE_RE = re.compile(r"\b(api|sql|etl|sdk|ci|cd|devops|nosql|ml)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Noise lists from the prompt.
# ---------------------------------------------------------------------------
EDUCATION_TERMS: Set[str] = {
    "bachelor",
    "bachelors",
    "masters",
    "master",
    "degree",
    "coursework",
    "student",
    "school",
    "college",
}

RESUME_SECTION_WORDS: Set[str] = {
    "summary",
    "objective",
    "responsibilities",
    "responsibility",
    "profile",
    "resume",
}

LOCATION_WORDS: Set[str] = {
    "states",
    "city",
    "country",
    "india",
    "usa",
    "united states",
}

GENERIC_ACTION_WORDS: Set[str] = {
    "manage",
    "work",
    "perform",
    "complete",
    "conduct",
    "assist",
    "provide",
    "support",
    "handle",
}

ADJECTIVE_NOISE: Set[str] = {
    "accurate",
    "efficient",
    "strong",
    "excellent",
    "fast",
    "good",
    "dynamic",
    "motivated",
}


MULTIWORD_PROFESSIONAL_KEEP: Set[str] = {
    "data analysis",
    "project management",
    "financial reporting",
    "customer service",
    "risk management",
    # allow close variants
    "customer support",
    "business analysis",
}


OFFICE_PAIR_NOISE: Set[Tuple[str, str]] = {("word", "excel")}


def _normalize(skill: str) -> str:
    """Lowercase, strip, collapse whitespace. Input is expected already normalized."""
    s = str(skill).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def is_verb(word: str) -> bool:
    """
    Heuristic verb detector used for second-pass filtering.

    Removes tokens/skills that end with `ed` or `ing`, except for known tech exceptions.
    """
    w = _normalize(word)
    if not w:
        return False

    # Exceptions for real tech keywords that end with "ing".
    # (Keeps "spring" framework etc. if present in cleaned data.)
    if w in {"spring", "testing", "inc"}:
        return False

    # Only apply verb suffix rule to the final token to avoid dropping phrases like "cloud computing".
    last_tok = w.split()[-1]
    if last_tok in {"spring"}:
        return False
    return bool(VERB_SUFFIX_RE.search(last_tok))


def is_noise_phrase(skill: str) -> bool:
    """
    Return True if the skill looks like a resume/section/filler phrase or duplicated structure.
    """
    s = _normalize(skill)
    if not s:
        return True

    # Remove anything containing explicit noise tokens (whole-token match for speed).
    tokens = s.split()
    token_set = set(tokens)

    # Remove education/resume/location terms appearing anywhere as tokens.
    if token_set & EDUCATION_TERMS:
        return True
    if token_set & RESUME_SECTION_WORDS:
        return True
    if token_set & LOCATION_WORDS:
        return True

    # Remove exact generic action words.
    if s in GENERIC_ACTION_WORDS:
        return True

    # Duplicated structures like "client clients" / "sales sales" / "word excel".
    # 1) Adjacent repetition (or singular/plural near-repetition) patterns.
    if len(tokens) >= 2:
        for i in range(len(tokens) - 1):
            a, b = tokens[i], tokens[i + 1]
            if a == b:
                return True
            if b == f"{a}s" or a == f"{b}s":
                return True

    # 2) Explicit known noisy pair patterns.
    if len(tokens) == 2:
        a, b = tokens
        if (a, b) in OFFICE_PAIR_NOISE:
            return True

    return False


def looks_like_technology(skill: str) -> bool:
    """
    Heuristic: does this look like a real technical/professional skill?

    Uses patterns requested in the prompt:
    - numbers (css3, html5)
    - acronyms (api, sql)
    - multi-word professional phrases (data analysis, project management, etc.)
    - whitelist exact matches
    """
    s = _normalize(skill)
    if not s:
        return False

    # Always keep known real-tech exact names.
    if s in TECH_WHITELIST:
        return True

    # Numbers (css3, html5, office 365-like, etc.)
    if TECH_NUMBER_RE.search(s):
        return True

    # HTML/CSS version formats.
    if HTML_CSS_VERSION_RE.search(s):
        return True

    # Acronym-like tokens
    if ACRONYM_LIKE_RE.search(s):
        return True

    # Multi-word known professional phrases (explicit keep list)
    if s in MULTIWORD_PROFESSIONAL_KEEP:
        return True

    # Broader professional phrase heuristics. These are common resume skills and should
    # not be dropped just because they aren't narrowly "technologies".
    professional_patterns = (
        r"\bproject\s+(management|coordination|planning)\b",
        r"\bdata\s+analysis\b",
        r"\bfinancial\s+reporting\b",
        r"\bcustomer\s+(service|support)\b",
        r"\brisk\s+management\b",
        r"\b\w+\s+management\b",  # e.g., account/client/project management
        r"\b\w+\s+reporting\b",
        r"\b\w+\s+analysis\b",
        r"\b\w+\s+support\b",
        r"\b\w+\s+service\b",
    )
    if any(re.search(pat, s) for pat in professional_patterns):
        return True

    # Common tech keywords (broad but still tech-shaped; avoids verbs/adjectives already filtered)
    tech_keywords = {
        # Data / analysis
        "data",
        "analysis",
        "analytics",
        "data science",
        "machine learning",
        "deep learning",
        "data analysis",

        # SDLC / engineering
        "development",
        "engineering",
        "deployment",
        "automation",
        "testing",

        # Architecture / design
        "architecture",
        "architect",
        "design",

        # Agile / delivery
        "agile",
        "scrum",
        "kanban",
        "ci cd",
        "continuous integration",
        "continuous delivery",

        # API / platform
        "rest",
        "graphql",
        "microservices",
        "api",

        "linux",
        "unix",
        "windows",
        "macos",
        "docker",
        "kubernetes",
        "aws",
        "azure",
        "gcp",
        "react",
        "angular",
        "django",
        "flask",
        "node",
        "express",
        "spring",
        "hibernate",
        "laravel",
        "rails",
        "ruby",
        "elasticsearch",
        "kafka",
        "redis",
        "postgres",
        "mysql",
        "oracle",
        "mongodb",
        "mssql",
        "tableau",
        "power bi",
        "git",
        "github",
        "gitlab",
        "jira",
        "salesforce",
        "sap",
        "tensorflow",
        "pytorch",
        "scikit-learn",
        "sklearn",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "excel",
        # Common noun-forms that otherwise match the verb suffix heuristic.
        "programming",
        "networking",
        "android",
        "ios",
    }

    if any(k in s for k in tech_keywords):
        return True

    # Multi-word skills: if they look like "X Y" and X is known tech marker, keep.
    # Example: "data analysis" already handled, but this catches "cloud security" etc.
    marker_tokens = {
        "data",
        "cloud",
        "software",
        "risk",
        "financial",
        "customer",
        "network",
        "security",
        "project",
        "system",
        "information",
        "machine",
    }
    if len(s.split()) >= 2 and any(m in s.split() for m in marker_tokens):
        # Avoid keeping pure action/education phrases; those get filtered by is_noise_phrase/is_valid_skill.
        return True

    return False


def is_valid_skill(skill: str) -> bool:
    """
    Final validation gate for canonical/alias tokens.

    - Never remove explicit `TECH_WHITELIST`.
    - Otherwise filter out verbs/adjectives/generic action/education/resume/location noise.
    - Then keep only if it looks like technology (or explicit multi-word professional keep phrases).
    """
    s = _normalize(skill)
    if not s:
        return False

    if s in TECH_WHITELIST:
        return True

    # Duplicate structures and resume noise
    if is_noise_phrase(s):
        return False

    # Adjectives
    if s in ADJECTIVE_NOISE:
        return False

    # Generic action words
    if s in GENERIC_ACTION_WORDS:
        return False

    # Verbs: remove verb-like forms unless they also clearly look like
    # a real technology/professional skill (e.g., "programming", "networking").
    if is_verb(s) and not looks_like_technology(s):
        return False

    # If it looks like a technology / known professional phrase, keep immediately.
    if looks_like_technology(s):
        return True

    # Otherwise, use a lightweight POS-based noun check to preserve real professional
    # noun skills (e.g., "accountability", "analysis") while dropping adjective/adverb
    # noise. This is still rule-based.
    if _POS_TAGGER_AVAILABLE:
        tokens = s.split()
        # POS tagging works on tokens; cache results for speed.
        cache_key = " ".join(tokens)
        # Many resume fillers are short single tokens (e.g., "access", "active").
        # Require a minimum length for single-word skills.
        if len(tokens) == 1 and len(s) < 8:
            return False
        if not hasattr(is_valid_skill, "_pos_cache"):
            setattr(is_valid_skill, "_pos_cache", {})  # type: ignore[attr-defined]
        pos_cache: Dict[str, bool] = getattr(is_valid_skill, "_pos_cache")  # type: ignore[attr-defined]
        if cache_key in pos_cache:
            return pos_cache[cache_key]

        try:
            tagged = pos_tag(tokens)  # type: ignore[misc]
        except Exception:
            pos_cache[cache_key] = False
            return False

        # Keep if we have at least one noun token and we didn't see verb/adverb tokens.
        has_noun = False
        for tok, tag in tagged:
            tag = str(tag)
            if tok in TECH_WHITELIST:
                continue
            if tag.startswith("VB"):
                pos_cache[cache_key] = False
                return False
            if tag.startswith("RB"):
                pos_cache[cache_key] = False
                return False
            if tag.startswith("NN"):
                has_noun = True
            # Allow JJ only for known methodology-ish tokens.
            if tag.startswith("JJ") and tok in {"agile", "scrum", "kanban", "rest"}:
                has_noun = True

        pos_cache[cache_key] = has_noun
        return has_noun

    # Fallback if POS tagging isn't available:
    # - Keep multi-word skills (more specific; often true skills)
    # - Keep longer single-word terms
    return len(s.split()) >= 2 or len(s) >= 8


def _filter_aliases(aliases: Iterable[Any], canonical: str) -> List[str]:
    """Filter alias list; keep canonical; sort and dedupe."""
    canon = _normalize(canonical)
    out: Set[str] = set()
    for a in aliases or []:
        if not isinstance(a, str):
            a = str(a)
        aa = _normalize(a)
        if not aa:
            continue
        if is_valid_skill(aa) or aa == canon:
            out.add(aa)

    out.discard("")
    if canon:
        out.add(canon)

    final_aliases = sorted(out)
    if canon in out:
        final_aliases = [canon] + [x for x in final_aliases if x != canon]
    return final_aliases


def filter_real_skills(input_file: str | Path, output_file: str | Path) -> Tuple[int, int, int]:
    """
    Load cleaned_skills.json and write final_skills.json with only real skills.

    Returns:
      (original_count, removed_count, final_count)
    """
    inp = Path(input_file)
    outp = Path(output_file)
    if not inp.exists():
        raise FileNotFoundError(f"Input file not found: {inp}")

    try:
        raw = inp.read_text(encoding="utf-8")
        data: Any = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {inp}: {e}") from e

    if not isinstance(data, dict):
        raise ValueError("Expected cleaned_skills.json to be a JSON object mapping skill -> [aliases].")

    original_count = len(data)
    removed = 0
    final: Dict[str, List[str]] = {}

    for canonical, aliases in data.items():
        canon_norm = _normalize(canonical)
        if not canon_norm:
            removed += 1
            continue

        if not is_valid_skill(canon_norm):
            removed += 1
            continue

        final_aliases = _filter_aliases(aliases, canonical=canon_norm)
        if not final_aliases:
            removed += 1
            continue

        final[canon_norm] = final_aliases

    # Ensure output directory exists
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(final, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    final_count = len(final)
    print(
        "Original skill count: "
        f"{original_count}\nRemoved skill count: {removed}\nFinal skill count: {final_count}"
    )
    logger.info("Original skill count: %d", original_count)
    logger.info("Removed skill count: %d", removed)
    logger.info("Final skill count: %d", final_count)
    return original_count, removed, final_count


def main() -> None:
    filter_real_skills("cleaned_skills.json", "final_skills.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Common workflow: run from backend/, while cleaned data is in app/data/processed/.
        if isinstance(e, FileNotFoundError) and not Path("cleaned_skills.json").exists():
            backend_root = Path(__file__).resolve().parents[1]
            fallback_in = backend_root / "app" / "data" / "processed" / "cleaned_skills.json"
            fallback_out = fallback_in.parent / "final_skills.json"
            filter_real_skills(fallback_in, fallback_out)
        else:
            logger.error("%s", e)
            sys.exit(1)

