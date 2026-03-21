from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.pipeline.extraction_pipeline import SkillExtractionPipeline
from app.services.gap_analyzer import SkillGapAnalyzer


logger = logging.getLogger(__name__)

router = APIRouter()
_pipeline = SkillExtractionPipeline()
_gap_analyzer = SkillGapAnalyzer()

_EMPTY = {"keyword_skills": [], "gemini_skills": [], "final_skills": []}
_GAP_EMPTY = {
    "resume_skills": [],
    "jd_skills": [],
    "missing_skills": [],
    "matching_skills": [],
}


def _is_pdf(upload: UploadFile) -> bool:
    return bool(upload.filename and upload.filename.lower().endswith(".pdf"))


@router.post("")
async def analyze_resume(file: UploadFile = File(...)):
    """
    Accept a PDF upload and return keyword, Gemini, and merged skill lists.
    On any failure, returns empty lists.
    """
    if not _is_pdf(file):
        logger.warning("Rejected non-PDF upload for /analyze")
        return _EMPTY

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            if not content:
                return _EMPTY
            tmp.write(content)
            temp_path = Path(tmp.name)

        return _pipeline.run(temp_path)
    except Exception as exc:
        logger.exception("analyze endpoint error: %s", exc)
        return _EMPTY
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


@router.post("/resume-jd")
async def analyze_resume_and_jd(
    resume_file: UploadFile = File(...),
    jd_file: UploadFile = File(...),
):
    """
    Parse resume + JD PDFs, extract skills via existing pipeline, and compute gap.
    """
    if not _is_pdf(resume_file) or not _is_pdf(jd_file):
        logger.warning("Rejected non-PDF upload for /analyze/resume-jd")
        raise HTTPException(status_code=400, detail="Both resume_file and jd_file must be PDF files.")

    resume_bytes = await resume_file.read()
    jd_bytes = await jd_file.read()

    if not resume_bytes or not jd_bytes:
        logger.warning("Empty resume/jd file upload for /analyze/resume-jd")
        raise HTTPException(status_code=400, detail="Both resume_file and jd_file must be non-empty.")

    try:
        resume_result = _pipeline.run(resume_bytes)
        jd_result = _pipeline.run(jd_bytes)

        gap_result = _gap_analyzer.compute_gap(
            resume_result.get("final_skills", []),
            jd_result.get("final_skills", []),
        )

        logger.info(
            "Gap analysis completed: resume=%d jd=%d missing=%d",
            len(gap_result["resume_skills"]),
            len(gap_result["jd_skills"]),
            len(gap_result["missing_skills"]),
        )

        return {
            "resume": {"final_skills": resume_result.get("final_skills", [])},
            "jd": {"final_skills": jd_result.get("final_skills", [])},
            "gap": {
                "missing_skills": gap_result["missing_skills"],
                "matching_skills": gap_result["matching_skills"],
            },
        }
    except Exception as exc:
        logger.exception("resume-jd analysis error: %s", exc)
        return {
            "resume": {"final_skills": []},
            "jd": {"final_skills": []},
            "gap": {
                "missing_skills": _GAP_EMPTY["missing_skills"],
                "matching_skills": _GAP_EMPTY["matching_skills"],
            },
        }
