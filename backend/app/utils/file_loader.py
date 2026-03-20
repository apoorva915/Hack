from __future__ import annotations

import io
import re
from typing import BinaryIO, Optional


def clean_text(text: str) -> str:
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_pdf_with_pdfplumber(data: bytes) -> str:
    import pdfplumber  # type: ignore

    with pdfplumber.open(io.BytesIO(data)) as pdf:
        pages = []
        for p in pdf.pages:
            pages.append(p.extract_text() or "")
        return "\n".join(pages)


def _extract_pdf_with_pymupdf(data: bytes) -> str:
    import pymupdf  # type: ignore  # PyMuPDF

    doc = pymupdf.open(stream=data, filetype="pdf")
    pages = []
    for page in doc:
        pages.append(page.get_text() or "")
    return "\n".join(pages)


def _extract_pdf_with_pypdf(data: bytes) -> str:
    """
    Lowest-dependency fallback extractor (often works even when pdfplumber/PyMuPDF fail).
    """
    from pypdf import PdfReader  # type: ignore

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)


def extract_text_from_bytes(data: bytes, filename: str = "") -> str:
    """
    Extract readable text from PDF/DOCX/TXT-like inputs.
    """
    filename_lower = (filename or "").lower()

    # DOCX
    if filename_lower.endswith(".docx") or filename_lower.endswith(".doc"):
        try:
            from docx import Document  # type: ignore

            doc = Document(io.BytesIO(data))
            paras = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
            return clean_text("\n".join(paras))
        except Exception:
            return ""

    # PDF
    if filename_lower.endswith(".pdf"):
        # Try pdfplumber first; fallback to PyMuPDF; then pypdf.
        # We intentionally catch all errors so the API never 500s due to optional deps.
        try:
            return clean_text(_extract_pdf_with_pdfplumber(data))
        except Exception:
            pass

        try:
            return clean_text(_extract_pdf_with_pymupdf(data))
        except Exception:
            pass

        try:
            return clean_text(_extract_pdf_with_pypdf(data))
        except Exception:
            pass

        return ""

    # Fallback: treat as plain text.
    try:
        return clean_text(data.decode("utf-8", errors="ignore"))
    except Exception:
        return ""


def read_upload_file(upload_file) -> bytes:
    # FastAPI UploadFile provides a file-like object in .file.
    return upload_file.file.read()

