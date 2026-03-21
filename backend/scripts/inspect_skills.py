#!/usr/bin/env python3
"""
Inspect cleaned_skills.json: summary stats, top skills by alias count, random sample.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def load_cleaned(path: Path) -> Dict[str, List[str]]:
    """Load cleaned skills JSON; validate top-level shape."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    text = path.read_text(encoding="utf-8")
    data: Any = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object mapping skill -> [aliases].")
    out: Dict[str, List[str]] = {}
    for k, v in data.items():
        if not isinstance(k, str):
            continue
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            out[k] = list(v)
        else:
            logger.warning("Skipping malformed entry for key %r", k)
    return out


def inspect_skills(path: str | Path, *, top_n: int = 50, random_n: int = 20, seed: int | None = None) -> None:
    """Print total count, first ``top_n`` skills alphabetically, and a random sample."""
    p = Path(path)
    data = load_cleaned(p)
    total = len(data)
    print(f"File: {p.resolve()}")
    print(f"Total skill count: {total}")
    print()

    sorted_keys = sorted(data.keys())
    top = sorted_keys[:top_n]
    print(f"Top {top_n} skills (alphabetically):")
    for i, skill in enumerate(top, start=1):
        cnt = len(data[skill])
        print(f"  {i:2}. {skill!r}  ({cnt} aliases)")
    print()

    keys = list(data.keys())
    if not keys:
        print("Random sample: (empty)")
        return
    rng = random.Random(seed)
    sample_size = min(random_n, len(keys))
    sample = rng.sample(keys, sample_size)
    print(f"Random {sample_size} skills:")
    for s in sorted(sample):
        aliases = data[s]
        preview = aliases[:5]
        extra = "..." if len(aliases) > 5 else ""
        print(f"  {s!r}: {preview}{extra}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect cleaned_skills.json")
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to cleaned_skills.json (default: cleaned_skills.json in CWD, or processed/)",
    )
    parser.add_argument("--top", type=int, default=50, help="How many top skills to show (default: 50)")
    parser.add_argument("--random", type=int, default=20, dest="random_n", help="Random sample size (default: 20)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducible random sample")
    args = parser.parse_args()

    if args.path:
        path = Path(args.path)
    else:
        cwd = Path("cleaned_skills.json")
        backend = Path(__file__).resolve().parents[1]
        fallback = backend / "app" / "data" / "processed" / "cleaned_skills.json"
        path = cwd if cwd.exists() else fallback

    try:
        inspect_skills(path, top_n=args.top, random_n=args.random_n, seed=args.seed)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        logger.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
