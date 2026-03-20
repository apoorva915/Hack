from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from ..models.schemas import UploadResponse
from ..services.parser import parse_job_description, parse_resume


router = APIRouter(tags=["upload"])


@router.post("/upload", response_model=UploadResponse)
async def upload(
    request: Request,
    resume_file: UploadFile = File(...),
    job_description_text: Optional[str] = Form(None),
    job_description_file: Optional[UploadFile] = File(None),
):
    if resume_file is None:
        raise HTTPException(status_code=400, detail="resume_file is required")

    jd_text = parse_job_description(text=job_description_text, file=job_description_file)
    if not jd_text.strip():
        if job_description_text and job_description_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not parse the job description text. Please check the input and try again.",
            )
        if job_description_file is not None:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from the job description file. Try a different file (PDF/DOCX).",
            )
        raise HTTPException(
            status_code=400, detail="Provide job description text or a job_description_file"
        )

    resume_text = parse_resume(resume_file)
    if not resume_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from resume. Try a different file.")

    store = request.app.state.store
    analysis_id = store.create(resume_text=resume_text, jd_text=jd_text)

    return UploadResponse(
        analysis_id=analysis_id,
        resume_text_preview=resume_text[:800],
        jd_text_preview=jd_text[:800],
    )

