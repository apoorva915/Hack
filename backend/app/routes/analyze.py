from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.pipeline.extraction_pipeline import SkillExtractionPipeline
from app.services.gap_analyzer import SkillGapAnalyzer
from app.services.graph.path_generator import AdaptivePathGenerator


logger = logging.getLogger(__name__)

router = APIRouter()
_pipeline = SkillExtractionPipeline()
_gap_analyzer = SkillGapAnalyzer()
_path_generator = AdaptivePathGenerator()

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
        resume_skills = resume_result.get("final_skills", [])
        jd_skills = jd_result.get("final_skills", [])
        logger.info("Resume skills=%d", len(resume_skills))
        logger.info("JD skills=%d", len(jd_skills))

        gap_result = _gap_analyzer.compute_gap(
            resume_skills,
            jd_skills,
        )
        missing_skills = gap_result["missing_skills"]
        logger.info("Missing skills=%d", len(missing_skills))
        path_result = _path_generator.generate_path(
            user_skills=gap_result["resume_skills"],
            target_skills=missing_skills,
        )
        learning_path = path_result["ordered_path"]
        phases_dynamic = path_result["phases"]
        phases = _path_generator.build_phase_map(phases_dynamic)
        graph_payload = _path_generator.build_graph_payload(learning_path)
        estimated_learning_time_hours = _path_generator.estimate_learning_time(learning_path)
        logger.info("Learning path size=%d", len(learning_path))
        logger.info("Phase count=%d", len(phases_dynamic))

        logger.info(
            "Gap analysis completed: resume=%d jd=%d missing=%d learning_path=%d phases=%d",
            len(gap_result["resume_skills"]),
            len(gap_result["jd_skills"]),
            len(gap_result["missing_skills"]),
            len(learning_path),
            len(phases_dynamic),
        )

        return {
            "resume": {"final_skills": resume_result.get("final_skills", [])},
            "jd": {"final_skills": jd_result.get("final_skills", [])},
            "gap": {
                "missing_skills": missing_skills,
                "matching_skills": gap_result["matching_skills"],
            },
            "adaptive_learning": {
                "missing_skills": missing_skills,
                "learning_path": learning_path,
                "phases": phases,
                "phases_dynamic": phases_dynamic,
                "graph": graph_payload,
                "estimated_learning_time_hours": estimated_learning_time_hours,
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
            "adaptive_learning": {
                "missing_skills": _GAP_EMPTY["missing_skills"],
                "learning_path": [],
                "phases": {
                    "Phase 1": [],
                    "Phase 2": [],
                    "Phase 3": [],
                },
                "graph": {
                    "nodes": [],
                    "edges": [],
                },
                "estimated_learning_time_hours": 0,
            },
        }
