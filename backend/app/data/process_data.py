
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import logging

import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer

# Allow importing `app.*` when running this script directly:
#   python app/data/process_data.py
BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.pdfplumber_parser import extract_text_from_pdf  # noqa: E402

# Dataset generation can involve hundreds/thousands of PDFs; silence per-PDF logs
# so the script output remains readable and only includes summary logs.
logging.getLogger("app.services.pdfplumber_parser").setLevel(logging.ERROR)


RAW_DIR = Path(__file__).resolve().parent / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent / "processed"
SKILLS_JSON_PATH = PROCESSED_DIR / "skills.json"


# ----------------------------
# Logging
# ----------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


# ----------------------------
# Text cleaning
# ----------------------------

_PUNCT_KEEP = r"\+\#\."  # keep common programming-language tokens
_NON_ALNUM_RE = re.compile(rf"[^a-z0-9\s{_PUNCT_KEEP}]")
_MULTISPACE_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"\b\d+\b")


def clean_text(text: str) -> str:
    """
    Deterministic text cleaning:
      - lowercase
      - remove punctuation (while preserving +, #, . for programming tokens)
      - remove numbers
      - normalize whitespace
    """
    if text is None:
        return ""

    text = str(text).lower()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _NON_ALNUM_RE.sub(" ", text)
    text = _NUM_RE.sub(" ", text)
    text = _MULTISPACE_RE.sub(" ", text).strip()
    return text


def html_to_text(html: str) -> str:
    """
    Convert HTML string to text using BeautifulSoup.
    """
    if not html:
        return ""
    try:
        soup = BeautifulSoup(html, "lxml")
        return soup.get_text(" ")
    except Exception:
        # Fallback: crude stripping if parser fails.
        return BeautifulSoup(html, "html.parser").get_text(" ")


# ----------------------------
# Loading sources
# ----------------------------

def _read_dataframe(path: Path) -> pd.DataFrame:
    """
    Read .xlsx/.xls or fallback to .csv based on extension.
    """
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(str(path))
    if suffix == ".csv":
        return pd.read_csv(str(path))
    raise ValueError(f"Unsupported file type: {path}")


def load_resume_corpus() -> List[str]:
    """
    Load resume text corpus from:
      - raw/Resume/Resume.xlsx (preferred)
      - raw/Resume/Resume.csv (fallback)
    """
    resume_dir = RAW_DIR / "Resume"
    xlsx = resume_dir / "Resume.xlsx"
    csv = resume_dir / "Resume.csv"

    path = xlsx if xlsx.exists() else csv if csv.exists() else None
    if path is None:
        log(f"[WARN] No resume file found under: {resume_dir}")
        return []

    df = _read_dataframe(path)

    # Common column names in your dataset:
    #   Resume_str, Resume_html (csv) or Resume (xlsx)
    candidate_cols = [
        "Resume_str",
        "Resume",
        "resume",
        "text",
        "Skills",
        "skills",
        "Resume_html",
        "resume_html",
        "ResumeHTML",
    ]

    corpus: List[str] = []
    for _, row in df.iterrows():
        text_val: Optional[str] = None
        for col in candidate_cols:
            if col in df.columns and pd.notna(row.get(col)):
                text_val = row.get(col)
                break
        if text_val is None:
            continue

        # If we selected an HTML-like column, strip to text.
        col_used = next((c for c in candidate_cols if c in df.columns and pd.notna(row.get(c))), None)
        if col_used and "html" in col_used.lower():
            text_val = html_to_text(str(text_val))
        corpus.append(clean_text(str(text_val)))

    # Filter empties deterministically
    return [t for t in corpus if t]


def load_job_description_corpus() -> List[str]:
    """
    Load job description corpus from:
      - raw/job_title_des.xlsx (preferred)
      - raw/job_title_des.csv (fallback)
    """
    xlsx = RAW_DIR / "job_title_des.xlsx"
    csv = RAW_DIR / "job_title_des.csv"
    path = xlsx if xlsx.exists() else csv if csv.exists() else None

    if path is None:
        log(f"[WARN] No job description file found in: {RAW_DIR}")
        return []

    df = _read_dataframe(path)

    # Your csv header is: ',Job Title,Job Description' => columns might include an empty string.
    # We primarily want "Job Description" column.
    desc_col_candidates = ["Job Description", "job_description", "description", "JobDescription"]
    desc_col = next((c for c in desc_col_candidates if c in df.columns), None)

    # Fallback: pick the last column if no explicit match.
    if desc_col is None and len(df.columns) > 0:
        desc_col = df.columns[-1]

    corpus: List[str] = []
    for _, row in df.iterrows():
        val = row.get(desc_col) if desc_col is not None else None
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        corpus.append(clean_text(str(val)))

    return [t for t in corpus if t]


def load_html_corpus() -> List[str]:
    """
    Load additional text from job documents organized by category folders.

    In your current repo, these documents are stored as `.pdf` files (even though they may
    originate from HTML). If `.html` files are found, we parse them; otherwise we fall back
    to extracting text from `.pdf` files.
    """
    root = RAW_DIR / "data" / "data"
    if not root.exists():
        log(f"[INFO] No HTML corpus directory found at: {root} (skipping)")
        return []

    corpus: List[str] = []

    # Prefer HTML if present.
    html_files: List[Path] = []
    html_files.extend(sorted(root.rglob("*.html")))
    html_files.extend(sorted(root.rglob("*.htm")))

    if html_files:
        max_html_files = int(os.getenv("HTML_MAX_FILES", "0"))
        html_files = html_files[:max_html_files] if max_html_files > 0 else html_files
        log(f"[INFO] Found {len(html_files)} HTML files under {root}")
        for p in html_files:
            try:
                html = p.read_text(encoding="utf-8", errors="ignore")
                corpus.append(clean_text(html_to_text(html)))
            except Exception:
                continue
        return [t for t in corpus if t]

    # Fallback: extract text from PDFs.
    pdf_files: List[Path] = sorted(root.rglob("*.pdf"))
    if not pdf_files:
        log(f"[INFO] No .html or .pdf files found under: {root} (skipping)")
        return []

    max_pdf_files = int(os.getenv("PDF_MAX_FILES", "0"))
    pdf_files = pdf_files[:max_pdf_files] if max_pdf_files > 0 else pdf_files
    log(f"[INFO] No HTML found; extracting from {len(pdf_files)} PDF files under {root}")

    for idx, p in enumerate(pdf_files, start=1):
        try:
            with p.open("rb") as f:
                result = extract_text_from_pdf(f)
            if result.get("status") == "success" and result.get("text"):
                corpus.append(clean_text(result["text"]))
        except Exception:
            continue

        # Light progress logging to make long runs feel responsive.
        if idx % int(os.getenv("PDF_PROGRESS_EVERY", "25")) == 0:
            log(f"[INFO] PDF corpus progress: {idx}/{len(pdf_files)}")

    return [t for t in corpus if t]


# ----------------------------
# Skill candidate extraction
# ----------------------------

GENERIC_TERMS = {
    "company",
    "role",
    "job",
    "jobs",
    "description",
    "responsibilities",
    "responsibility",
    "requirements",
    "required",
    "experience",
    "skill",
    "skills",
    "candidate",
    "position",
    "work",
    "workplace",
    "team",
    "teams",
    "training",
    "education",
    "degree",
    "company's",
    "requirements",
    "including",
    "ability",
    "using",
    "develop",
    "development",
    "including",
}


def _keep_candidate_term(term: str, generic: Set[str]) -> bool:
    """
    Filter out:
      - generic terms
      - terms with too-short tokens
    """
    term = term.strip().lower()
    if not term:
        return False
    if term in generic:
        return False

    tokens = term.split()
    # For unigrams: keep if length > 2
    # For bigrams: keep if all tokens length > 2
    if any(len(tok) <= 2 for tok in tokens):
        return False
    return True


def extract_skills_from_corpus(
    all_texts: Sequence[str],
    min_df: int,
    max_terms: int,
) -> Dict[str, List[str]]:
    """
    Deterministically extract candidate unigrams/bigrams as skills.
    """
    if not all_texts:
        return {}

    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        min_df=min_df,
    )

    X = vectorizer.fit_transform(all_texts)

    terms = vectorizer.get_feature_names_out()
    counts = X.sum(axis=0).A1  # total frequency across documents

    # Sort deterministically: count desc, then term asc.
    idxs = list(range(len(terms)))
    idxs.sort(key=lambda i: (-float(counts[i]), str(terms[i])))

    skills: Dict[str, List[str]] = {}
    kept = 0
    for i in idxs:
        term = str(terms[i]).strip().lower()
        if _keep_candidate_term(term, GENERIC_TERMS):
            skills[term] = [term]
            kept += 1
            if kept >= max_terms:
                break

    return skills


def merge_manual_skills(skills: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Add manual important skills and merge aliases deterministically.
    """
    manual: Dict[str, List[str]] = {
        "python": ["python", "py"],
        "machine learning": ["machine learning", "ml"],
        "sql": ["sql", "mysql", "postgres"],
    }

    # Merge: if canonical exists, extend aliases; else insert.
    for canonical, aliases in manual.items():
        canonical_norm = canonical.strip().lower()
        alias_list = [a.strip().lower() for a in aliases if a and str(a).strip()]
        if not canonical_norm:
            continue
        if canonical_norm in skills:
            merged = sorted(set([canonical_norm] + skills[canonical_norm] + alias_list))
            skills[canonical_norm] = merged
        else:
            skills[canonical_norm] = sorted(set(alias_list))
    return skills


# ----------------------------
# Main
# ----------------------------

def run() -> None:
    start = time.perf_counter()

    min_df = int(os.getenv("SKILL_MIN_DF", "5"))
    max_terms = int(os.getenv("SKILL_MAX_TERMS", "2000"))

    log("[INFO] Loading raw resume corpus...")
    resume_texts = load_resume_corpus()
    log(f"[INFO] Resume texts loaded: {len(resume_texts)}")

    log("[INFO] Loading raw job-description corpus...")
    job_texts = load_job_description_corpus()
    log(f"[INFO] Job-description texts loaded: {len(job_texts)}")

    log("[INFO] Loading HTML corpus (optional)...")
    html_texts = load_html_corpus()
    log(f"[INFO] HTML/PDF job texts loaded: {len(html_texts)}")

    all_texts = []
    all_texts.extend(resume_texts)
    all_texts.extend(job_texts)
    all_texts.extend(html_texts)

    # Deterministic ordering.
    all_texts = [t for t in all_texts if t]
    log(f"[INFO] Total cleaned texts for vectorization: {len(all_texts)}")

    skills = extract_skills_from_corpus(
        all_texts=all_texts,
        min_df=min_df,
        max_terms=max_terms,
    )
    skills = merge_manual_skills(skills)

    # Ensure deterministic output ordering.
    out: Dict[str, List[str]] = {k: sorted(set(v)) for k, v in skills.items()}

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with SKILLS_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    elapsed = time.perf_counter() - start
    keys_sorted = sorted(out.keys())

    log(f"[INFO] skills.json saved to: {SKILLS_JSON_PATH}")
    log(f"[INFO] Total extracted skills: {len(keys_sorted)}")
    log(f"[INFO] Top 20 skills: {keys_sorted[:20]}")
    log(f"[INFO] Done in {elapsed:.2f}s")


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        sys.exit(130)
