from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class UploadedState:
    resume_text: str
    jd_text: str
    created_at: float
    analysis_result: Optional[object] = None


class AnalysisStore:
    """
    Demo-friendly in-memory store.
    For production, replace with Redis/Postgres.
    """

    def __init__(self, ttl_seconds: int):
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._store: Dict[str, UploadedState] = {}

    def create(self, resume_text: str, jd_text: str) -> str:
        analysis_id = str(uuid.uuid4())
        now = time.time()
        with self._lock:
            self._store[analysis_id] = UploadedState(
                resume_text=resume_text, jd_text=jd_text, created_at=now
            )
        return analysis_id

    def get(self, analysis_id: str) -> UploadedState:
        self._purge_expired()
        with self._lock:
            if analysis_id not in self._store:
                raise KeyError("analysis_id not found")
            return self._store[analysis_id]

    def set_result(self, analysis_id: str, result: object) -> None:
        # Avoid calling `self.get()` while holding the lock (would deadlock).
        self._purge_expired()
        with self._lock:
            if analysis_id not in self._store:
                raise KeyError("analysis_id not found")
            self._store[analysis_id].analysis_result = result

    def _purge_expired(self) -> None:
        if self._ttl_seconds <= 0:
            return
        cutoff = time.time() - self._ttl_seconds
        with self._lock:
            expired = [k for k, v in self._store.items() if v.created_at < cutoff]
            for k in expired:
                del self._store[k]

