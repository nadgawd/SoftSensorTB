"""Request limits for the public deployment.

Chat turns spend LLM credit and uploads spend storage, so both are capped per
user. The window is kept in process memory: the backend runs as one replica,
and a restart forgetting the counts is harmless.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque

from fastapi import HTTPException


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def max_upload_bytes() -> int:
    return _env_int("MAX_UPLOAD_MB", 25) * 1024 * 1024


def chat_limit_per_hour() -> int:
    """0 disables the limit. Defaults on only when requests are authenticated."""
    from backend.auth import auth_required

    return _env_int("CHAT_RATE_LIMIT_PER_HOUR", 40 if auth_required() else 0)


def manual_limit_per_hour() -> int:
    """Manual tool runs cost no LLM credit but do cost CPU. 0 disables the limit."""
    from backend.auth import auth_required

    return _env_int("MANUAL_RATE_LIMIT_PER_HOUR", 600 if auth_required() else 0)


class SlidingWindowLimiter:
    def __init__(self, window_s: float = 3600.0) -> None:
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, limit: int, what: str = "Chat limit reached ({limit} messages per hour)") -> None:
        if limit <= 0:
            return
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > self.window_s:
            hits.popleft()
        if len(hits) >= limit:
            retry = int(self.window_s - (now - hits[0])) + 1
            raise HTTPException(
                status_code=429,
                detail=f"{what.format(limit=limit)}. Try again in {retry // 60 + 1} min.",
                headers={"Retry-After": str(retry)},
            )
        hits.append(now)

    def reset(self) -> None:
        self._hits.clear()


chat_limiter = SlidingWindowLimiter()
manual_limiter = SlidingWindowLimiter()
