from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv
from requests import HTTPError, Response


logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

load_dotenv()


class GeminiClientError(Exception):
    """Raised when Gemini API invocation fails after retries."""


def _build_request_payload(prompt: str) -> Dict[str, Any]:
    return {
        "contents": [
            {
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
        },
    }


def _extract_response_text(data: Dict[str, Any]) -> str:
    """
    Extract raw text from Gemini response payload.
    """
    candidates = data.get("candidates")
    if not candidates:
        return ""

    first = candidates[0] or {}
    content = first.get("content") or {}
    parts = content.get("parts") or []
    if not parts:
        return ""

    text = parts[0].get("text", "")
    return str(text).strip()


def _safe_http_error_message(response: Optional[Response]) -> str:
    """Log details without ever echoing the API key (URLs in str(exc) include ?key=...)."""
    if response is None:
        return "no response"
    snippet = ""
    try:
        snippet = (response.text or "")[:300].replace("\n", " ")
    except Exception:
        pass
    return f"HTTP {response.status_code} {snippet}".strip()


def _backoff_seconds_for_retry(
    attempt: int,
    max_retries: int,
    response: Optional[Response],
) -> None:
    """Sleep before the next attempt; 429 gets longer waits + Retry-After support."""
    if attempt >= max_retries:
        return

    if response is not None and response.status_code == 429:
        ra = response.headers.get("Retry-After")
        if ra:
            try:
                wait = min(float(ra), 120.0)
                logger.warning("Gemini 429: honoring Retry-After=%.1fs", wait)
                time.sleep(wait)
                return
            except ValueError:
                pass
        base = float(os.getenv("GEMINI_429_BACKOFF_SECONDS", "10"))
        wait = min(base * attempt, 90.0)
        logger.warning("Gemini 429: backing off %.1fs before retry", wait)
        time.sleep(wait)
        return

    time.sleep(min(2 ** (attempt - 1), 8))


def call_gemini(prompt: str) -> str:
    """
    Send prompt to Gemini API and return raw response text.

    Behavior:
    - Loads API key from GEMINI_API_KEY
    - Retries on failure (extra-long backoff on HTTP 429)
    - Never logs full request URLs (avoids leaking API keys)

    Env:
    - GEMINI_MODEL: default gemini-3-flash-preview (override in .env if needed)
    - GEMINI_MAX_RETRIES: default 5, max 8
    - GEMINI_TIMEOUT_SECONDS: default 60 (large prompts + skill list)
    """
    if not prompt or not prompt.strip():
        raise GeminiClientError("Prompt is empty.")

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise GeminiClientError("Missing GEMINI_API_KEY environment variable.")

    model = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview").strip()

    timeout_s = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "60"))
    max_retries = min(max(1, int(os.getenv("GEMINI_MAX_RETRIES", "5"))), 8)

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    params = {"key": api_key}
    payload = _build_request_payload(prompt=prompt)

    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Calling Gemini API model=%s (attempt %d/%d)", model, attempt, max_retries)
            response = requests.post(
                url,
                params=params,
                json=payload,
                timeout=timeout_s,
            )
            last_response = response

            if response.status_code == 429:
                last_error = HTTPError(f"429 Too Many Requests: {_safe_http_error_message(response)}")
                logger.error("Gemini rate limited: %s", _safe_http_error_message(response))
                _backoff_seconds_for_retry(attempt, max_retries, response)
                continue

            response.raise_for_status()
            data = response.json()
            text = _extract_response_text(data)
            if not text:
                raise GeminiClientError("Gemini response was empty.")
            return text

        except requests.Timeout as exc:
            last_error = exc
            logger.error("Gemini timeout on attempt %d: %s", attempt, exc.__class__.__name__)
        except HTTPError as exc:
            last_error = exc
            logger.error(
                "Gemini HTTP error on attempt %d: %s",
                attempt,
                _safe_http_error_message(exc.response),
            )
            _backoff_seconds_for_retry(attempt, max_retries, exc.response)
            continue
        except requests.RequestException as exc:
            last_error = exc
            # Avoid str(exc) — often contains ?key= in URL
            logger.error("Gemini request error on attempt %d: %s", attempt, exc.__class__.__name__)
        except ValueError as exc:
            last_error = exc
            logger.error("Gemini response parse error on attempt %d: %s", attempt, str(exc))
        except GeminiClientError as exc:
            last_error = exc
            logger.error("Gemini returned invalid payload on attempt %d: %s", attempt, str(exc))

        _backoff_seconds_for_retry(attempt, max_retries, None)

    raise GeminiClientError(
        f"Gemini API call failed after {max_retries} retries: {last_error!r}"
    )

