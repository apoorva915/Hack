from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, BinaryIO, Union

import fitz  # PyMuPDF
import regex


logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)


def _as_readable_stream(file: Any) -> tuple[Union[BinaryIO, Path], bool]:
    """
    Normalize FastAPI UploadFile, file-like, bytes, or path to something fitz can open.

    Returns:
        (stream_or_path, owns_temp_file)
    """
    # FastAPI UploadFile
    inner = getattr(file, "file", None)
    if inner is not None:
        try:
            if hasattr(inner, "seek"):
                inner.seek(0)
        except Exception:
            pass
        return inner, False

    if isinstance(file, (bytes, bytearray)):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(bytes(file))
        tmp.flush()
        tmp.close()
        return Path(tmp.name), True

    if isinstance(file, str):
        return Path(file), False

    # pathlib.Path / WindowsPath (used by routes that save uploads to a temp file)
    if isinstance(file, Path):
        return file, False

    # Other path-like objects (e.g. some framework types)
    if isinstance(file, os.PathLike):
        return Path(file), False

    if hasattr(file, "read"):
        try:
            if hasattr(file, "seek"):
                file.seek(0)
        except Exception:
            pass
        return file, False

    raise TypeError("Unsupported file type for PDF extraction.")


def extract_text_from_pdf(file: Any) -> str:
    """
    Extract full text from a PDF using PyMuPDF.

    Accepts:
        - FastAPI UploadFile
        - file-like object with .read()
        - bytes / bytearray
        - str path to a PDF file

    Returns:
        Cleaned plain text, or empty string on failure.
    """
    temp_path: Path | None = None
    try:
        stream_or_path, owns_temp = _as_readable_stream(file)
        doc_path: str | Path | BinaryIO
        if owns_temp and isinstance(stream_or_path, Path):
            temp_path = stream_or_path
            doc_path = str(temp_path)
        else:
            doc_path = stream_or_path

        text_parts: list[str] = []
        with fitz.open(doc_path) as doc:
            for page in doc:
                text_parts.append(page.get_text("text") or "")

        combined = "\n".join(text_parts).strip()
        combined = regex.sub(r"[ \t]+", " ", combined)
        combined = regex.sub(r"\n{3,}", "\n\n", combined)
        combined = regex.sub(r"^\s+|\s+$", "", combined, flags=regex.MULTILINE)

        logger.info("log: PDF parsed successfully")
        return combined
    except Exception as exc:
        logger.exception("PDF parsing failed: %s", exc)
        return ""
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
