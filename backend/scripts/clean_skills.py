#!/usr/bin/env python3
"""
Rule-based cleaning of a noisy skill vocabulary (skills.json) into cleaned_skills.json.

No LLM usage. Designed for 10k+ entries with O(1) set lookups.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import string
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Whitelist: canonical skill names that must never be dropped as keys.
# ---------------------------------------------------------------------------
IMPORTANT_SKILLS: Set[str] = {
    "python",
    "sql",
    "java",
    "javascript",
    "react",
    "node",
    "aws",
    "docker",
    "kubernetes",
    "tensorflow",
    "machine learning",
    "deep learning",
    "data science",
    "pandas",
    "numpy",
}

# Short aliases allowed only when attached to a whitelisted or otherwise valid canonical.
# (RULE 2: "ml" keep only if alias exists — we allow these as aliases anywhere.)
# Two-letter (or single-letter) tech abbreviations allowed as aliases / short keys.
ALLOWED_SHORT_ALIASES: Set[str] = {
    "ml",
    "ai",
    "ui",
    "ux",
    "go",
    "r",
    "c",
    "js",
    "ts",
    "py",
    "cs",
    "vb",
    "qt",
    "io",
    "ci",
    "cd",
    "bi",
}

# Multi-word / phrase noise: if ANY of these appear as whole tokens in the phrase, drop.
NOISE_SUBSTRINGS: Set[str] = {
    "company",
    "state",
    "year",
    "years",
    "month",
    "months",
    "week",
    "weeks",
    "day",
    "days",
    "preferred",
    "university",
    "experience",
    "salary",
    "location",
    "remote",
    "full time",
    "part time",
}

# Common verb / action stems (fragments).
VERB_LIKE: Set[str] = {
    "work",
    "worked",
    "working",
    "develop",
    "developed",
    "developing",
    "design",
    "designed",
    "make",
    "made",
    "making",
    "use",
    "used",
    "using",
    "help",
    "helped",
    "helping",
    "manage",
    "managed",
    "managing",
    "lead",
    "led",
    "leading",
    "responsible",
    "ensure",
    "ensuring",
    "support",
    "supported",
    "collaborate",
    "collaborated",
    "communicate",
    "communicated",
    "build",
    "built",
    "building",
    "create",
    "created",
    "creating",
    "implement",
    "implemented",
    "maintain",
    "maintained",
    "perform",
    "performed",
    "provide",
    "provided",
    "review",
    "reviewed",
    "analyze",
    "analyzed",
    "deliver",
    "delivered",
}

MONTHS: Set[str] = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
}

# Custom stopwords + generic job words (superset when NLTK unavailable).
CUSTOM_STOPWORDS: Set[str] = {
    "able",
    "about",
    "above",
    "across",
    "after",
    "again",
    "against",
    "all",
    "also",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "done",
    "each",
    "few",
    "for",
    "from",
    "further",
    "get",
    "got",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "like",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "good",
    "great",
    "best",
    "strong",
    "excellent",
    "team",
    "teams",
    "time",
    "times",
    "year",
    "years",
    "month",
    "months",
    "week",
    "weeks",
    "day",
    "days",
    "hour",
    "hours",
    "job",
    "jobs",
    "role",
    "roles",
    "position",
    "candidate",
    "candidates",
    "employer",
    "employee",
    "office",
    "skills",
    "skill",
    "ability",
    "abilities",
    "experience",
    "experiences",
    "responsibilities",
    "responsibility",
    "requirements",
    "required",
    "preferred",
    "including",
    "including",
    "etc",
    "various",
    "multiple",
    "several",
    "many",
    "someone",
    "something",
    "someone",
    "well",
    "highly",
    "effective",
    "efficient",
    "successful",
    "new",
    "old",
    "current",
    "previous",
    "prior",
    "first",
    "second",
    "third",
    "last",
    "next",
    "other",
    "another",
    "such",
    "same",
    "different",
    "similar",
    "related",
    "general",
    "specific",
    "various",
    "key",
    "main",
    "primary",
    "secondary",
}

_STOPWORDS_CACHE: Optional[Set[str]] = None


def _load_stopwords() -> Set[str]:
    """NLTK English stopwords if available; else CUSTOM_STOPWORDS only."""
    global _STOPWORDS_CACHE
    if _STOPWORDS_CACHE is not None:
        return _STOPWORDS_CACHE

    words: Set[str] = set(CUSTOM_STOPWORDS)
    words |= MONTHS
    words |= VERB_LIKE
    try:
        import nltk  # type: ignore

        from nltk.corpus import stopwords  # type: ignore

        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)
        words |= set(stopwords.words("english"))
        logger.info("Loaded NLTK English stopwords.")
    except Exception:
        logger.info("NLTK stopwords not available; using custom stopword set only.")

    _STOPWORDS_CACHE = {w.lower() for w in words if w}
    return _STOPWORDS_CACHE


# Preserve common tech punctuation inside tokens (c++, c#, .net style handled after normalize).
_PUNCT_REMOVE = set(string.punctuation) - {"+", "#", "."}


def clean_skill_name(skill: str) -> str:
    """
    Normalize a skill or alias string: lowercase, strip, collapse whitespace, remove punctuation
    except + # . (for c++, c#, .net-like tokens).
    """
    if skill is None:
        return ""
    s = str(skill).lower().strip()
    s = re.sub(r"\s+", " ", s)
    out: List[str] = []
    for ch in s:
        if ch in _PUNCT_REMOVE:
            continue
        out.append(ch)
    return "".join(out).strip()


def _is_pure_numeric(s: str) -> bool:
    return bool(s) and bool(re.fullmatch(r"\d+", s))


def _phrase_has_noise_token(phrase: str) -> bool:
    """RULE 8: drop multi-word phrases containing noise substrings as tokens or bigrams."""
    tokens = phrase.split()
    token_set = set(tokens)
    if token_set & NOISE_SUBSTRINGS:
        return True
    # bigrams
    for i in range(len(tokens) - 1):
        bigram = f"{tokens[i]} {tokens[i + 1]}"
        if bigram in NOISE_SUBSTRINGS:
            return True
    # substring scan for multi-word noise phrases
    for noise in ("full time", "part time", "year experience", "years experience", "state university"):
        if noise in phrase:
            return True
    return False


def is_valid_skill(skill: str, *, is_canonical: bool, stopwords: Set[str]) -> bool:
    """
    Return True if this token/phrase should be kept as a canonical skill key or alias.

    IMPORTANT_SKILLS canonical names are always valid when is_canonical and name matches whitelist.
    """
    name = clean_skill_name(skill)
    if not name:
        return False

    if is_canonical and name in IMPORTANT_SKILLS:
        return True

    if _is_pure_numeric(name):
        return False

    tokens = name.split()
    # Single token length rules
    if len(tokens) == 1:
        tok = tokens[0]
        if len(tok) < 3 and tok not in ALLOWED_SHORT_ALIASES:
            return False
    else:
        # Multi-word: noise check
        if _phrase_has_noise_token(name):
            return False

    first = tokens[0]
    if first in stopwords or name in stopwords:
        return False
    if name in MONTHS:
        return False
    if name in VERB_LIKE:
        return False

    # Single-token verb-like endings (light heuristic)
    if len(tokens) == 1 and len(first) > 4 and first.endswith("ing") and first not in {"spring", "string"}:
        if first in stopwords or first in VERB_LIKE:
            return False

    return True


def _short_token_ok(tok: str, *, canonical_clean: str) -> bool:
    """RULE 2: drop single-token skills/aliases shorter than 3 unless allowed or important canonical."""
    if len(tok) >= 3:
        return True
    if tok in ALLOWED_SHORT_ALIASES:
        return True
    if canonical_clean in IMPORTANT_SKILLS:
        return True
    return False


def clean_aliases(
    alias_list: Iterable[Any],
    *,
    canonical: str,
    stopwords: Set[str],
) -> List[str]:
    """
    Clean, filter, dedupe aliases for a canonical skill.

    Short tokens (<3 chars) are kept only if in ALLOWED_SHORT_ALIASES or the canonical is IMPORTANT_SKILLS.
    """
    canonical_clean = clean_skill_name(canonical)

    seen: Set[str] = set()
    out: List[str] = []

    for raw in alias_list or []:
        a = clean_skill_name(str(raw))
        if not a:
            continue
        if _is_pure_numeric(a):
            continue
        parts = a.split()
        first_tok = parts[0] if parts else ""
        if a in stopwords or first_tok in stopwords:
            continue
        if _phrase_has_noise_token(a):
            continue

        if len(parts) == 1 and not _short_token_ok(parts[0], canonical_clean=canonical_clean):
            continue

        if not is_valid_skill(a, is_canonical=False, stopwords=stopwords):
            # Still allow known short abbreviations as aliases for any valid canonical
            if not (len(parts) == 1 and parts[0] in ALLOWED_SHORT_ALIASES):
                continue

        if a not in seen:
            seen.add(a)
            out.append(a)

    out.sort()
    return out


def clean_skills(input_file: str | Path, output_file: str | Path) -> Tuple[int, int, int]:
    """
    Load skills.json, apply rules, write cleaned_skills.json.

    Returns:
        (original_count, removed_count, final_count)
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    stopwords = _load_stopwords()

    try:
        raw_text = input_path.read_text(encoding="utf-8")
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {input_path}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("skills.json must be a JSON object mapping skill -> [aliases].")

    original_count = len(data)
    cleaned: Dict[str, List[str]] = {}
    removed = 0
    # Merge all raw rows that normalize to the same canonical key (RULE 5 / duplicates).
    buckets: DefaultDict[str, List[Any]] = defaultdict(list)

    for key, aliases in data.items():
        canon = clean_skill_name(str(key))
        if not canon:
            removed += 1
            continue

        if canon not in IMPORTANT_SKILLS:
            if not is_valid_skill(canon, is_canonical=True, stopwords=stopwords):
                removed += 1
                continue

        alias_src: List[Any] = list(aliases) if isinstance(aliases, list) else []
        alias_src.append(canon)
        buckets[canon].extend(alias_src)

    for canon, combined in buckets.items():
        uniq_aliases = clean_aliases(combined, canonical=canon, stopwords=stopwords)

        rest = [a for a in uniq_aliases if a != canon]
        final_aliases = [canon] + sorted(rest) if canon in uniq_aliases else sorted(uniq_aliases)

        if not final_aliases:
            continue

        cleaned[canon] = final_aliases

    # Dedupe keys that normalized to same string (last wins — rare)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, sort_keys=True, ensure_ascii=False)

    final_count = len(cleaned)
    msg = (
        f"Original skill count: {original_count}\n"
        f"Removed skill count: {removed}\n"
        f"Final skill count: {final_count}"
    )
    print(msg)
    logger.info("Original skill count: %d", original_count)
    logger.info("Removed skill count: %d", removed)
    logger.info("Final skill count: %d", final_count)

    return original_count, removed, final_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean noisy skills.json into cleaned_skills.json")
    parser.add_argument(
        "input",
        nargs="?",
        default="skills.json",
        help="Path to input skills.json (default: skills.json)",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="cleaned_skills.json",
        help="Path to output cleaned_skills.json (default: cleaned_skills.json)",
    )
    args = parser.parse_args()
    clean_skills(args.input, args.output)


if __name__ == "__main__":
    main()
