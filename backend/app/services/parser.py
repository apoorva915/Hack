
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Dict, List

import pdfplumber

try:
    # Raised by pdfminer when the PDF file is syntactically invalid/corrupted.
    from pdfminer.pdfparser import PDFSyntaxError
except Exception:  # pragma: no cover
    # Fallback: keep the explicit exception handling contract even if the import path differs.
    # Note: This fallback won't match the real exception type, but it prevents misrouting ValueError.
    class PDFSyntaxError(Exception):  # type: ignore[no-redef]
        pass


logger = logging.getLogger(__name__)

# Ensure logs emit even if the host app hasn't configured logging yet.
# (FastAPI/Uvicorn typically configures this, so this usually becomes a no-op.)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)


def clean_text(text: str) -> str:
    """
    Normalize extracted PDF text into a cleaner, more consistent representation.

    - Normalizes unicode forms (NFKC)
    - Removes excessive spaces/tabs
    - Collapses repeated newlines
    - Trims leading/trailing whitespace
    """

    if not text:
        return ""

    # Normalize unicode and common whitespace characters.
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")  # non-breaking spaces
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse spaces/tabs but preserve newlines for readability.
    text = re.sub(r"[ \t]+", " ", text)

    # Reduce runs of blank lines. Keep at most 2 consecutive newlines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _error_response(num_pages: int, message: str) -> Dict[str, Any]:
    return {
        "text": "",
        "num_pages": int(num_pages),
        "status": "error",
        "error": message,
    }


def _success_response(text: str, num_pages: int) -> Dict[str, Any]:
    return {
        "text": text,
        "num_pages": int(num_pages),
        "status": "success",
        "error": None,
    }


def _extract_page_text(page: Any) -> str:
    """
    Extract text from a single pdfplumber page.

    Resume PDFs often have complex layouts; using layout-aware and text-flow settings
    usually yields more complete extraction than defaults.
    """

    # Try a couple of strategies and keep the longest result.
    strategies = [
        # Layout + text flow tends to preserve reading order.
        {"layout": True, "use_text_flow": True, "x_tolerance": 2, "y_tolerance": 2},
        {"layout": True, "use_text_flow": True, "x_tolerance": 3, "y_tolerance": 3},
        # Fallback to defaults for compatibility.
        {"layout": False, "use_text_flow": False, "x_tolerance": 3, "y_tolerance": 3},
    ]

    best_text = ""
    for kwargs in strategies:
        try:
            text = page.extract_text(**kwargs) or ""
        except Exception:
            # If one strategy fails on a particular PDF, try the next.
            continue

        if len(text) > len(best_text):
            best_text = text

    return best_text


def extract_text_from_pdf(file: Any) -> dict:
    """
    Extract text from a PDF using pdfplumber.

    Parameters:
        file: A file-like object (e.g. FastAPI UploadFile.file or an object with .read()).

    Returns:
        dict with keys:
          - text: str
          - num_pages: int
          - status: "success" | "error"
          - error: Optional[str]
    """

    # FastAPI's UploadFile exposes the real stream at `.file`.
    stream = getattr(file, "file", file)

    # pdfplumber expects a readable, typically seekable stream.
    try:
        if hasattr(stream, "seek"):
            stream.seek(0)
    except Exception:
        # If we can't seek, we'll still attempt extraction from the current cursor position.
        pass

    num_pages = 0
    extracted_parts: List[str] = []
    try:
        with pdfplumber.open(stream) as pdf:
            # pdfplumber.pages is a list of Page objects; it also works for empty PDFs.
            for page in pdf.pages:
                num_pages += 1
                page_text = _extract_page_text(page)

                # Preserve page boundaries with newlines; final whitespace normalization happens later.
                if page_text:
                    # Keep a delimiter between pages; clean_text will normalize excessive whitespace.
                    extracted_parts.append(page_text)
    except PDFSyntaxError as e:
        logger.error("PDFSyntaxError while parsing PDF: %s", str(e))
        return _error_response(num_pages=0, message="Corrupted PDF file")
    except ValueError as e:
        logger.error("ValueError while parsing PDF: %s", str(e))
        return _error_response(num_pages=num_pages, message=str(e) or "Invalid PDF")
    except Exception as e:
        logger.exception("Unexpected error while parsing PDF")
        return _error_response(num_pages=num_pages, message=str(e) or "Failed to parse PDF")

    # Normal extraction path: build cleaned text and handle scanned/empty PDFs.
    extracted_text = clean_text("\n".join(extracted_parts))
    logger.info("PDF parsed successfully: num_pages=%d", num_pages)

    if not extracted_text:
        # pdfplumber succeeded but couldn't find extractable text.
        logger.warning("No extractable text found (possibly scanned PDF). num_pages=%d", num_pages)
        return _error_response(
            num_pages=num_pages,
            message="No extractable text found (possibly scanned PDF)",
        )

    return _success_response(text=extracted_text, num_pages=num_pages)

