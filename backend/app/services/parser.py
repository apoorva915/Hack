from __future__ import annotations

from typing import Optional

from fastapi import UploadFile

from ..utils.file_loader import clean_text, extract_text_from_bytes, read_upload_file


def parse_resume(file: UploadFile) -> str:
    """
    Extract text from an uploaded resume PDF/DOCX.
    """
    raw = read_upload_file(file)
    text = extract_text_from_bytes(raw, filename=file.filename or "")
    return clean_text(text)


def parse_job_description(text: Optional[str] = None, file: Optional[UploadFile] = None) -> str:
    """
    Extract text for job description from either text field or uploaded file.
    """
    if text and text.strip():
        return clean_text(text)
    if file is not None:
        raw = read_upload_file(file)
        extracted = extract_text_from_bytes(raw, filename=file.filename or "")
        return clean_text(extracted)
    return ""

