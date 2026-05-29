"""Rate limiting.

Iki katman:
1) IP bazli: slowapi (dakika/saat/gun)
2) Session bazli: in-memory sayac (dakika)

slowapi decorator'larini FastAPI route'una API tarafinda baglariz.
Burada sadece session bazli kontrolu ve slowapi limiter instance'ini saglayacagiz.
"""
from __future__ import annotations
import time
from collections import defaultdict, deque
from threading import Lock
from typing import Deque, Dict

from slowapi import Limiter
from slowapi.util import get_remote_address

from core.config import SESSION_LIMIT_PER_MINUTE


# IP bazli (slowapi)
limiter = Limiter(key_func=get_remote_address)


class SessionRateLimiter:
    """Dakikada N istek per session_id."""

    def __init__(self, per_minute: int = SESSION_LIMIT_PER_MINUTE):
        self.per_minute = per_minute
        self._buckets: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, session_id: str) -> bool:
        """True = izin, False = limit asildi."""
        now = time.time()
        with self._lock:
            bucket = self._buckets[session_id]
            cutoff = now - 60
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self.per_minute:
                return False
            bucket.append(now)
            return True

    def cleanup_old(self, max_age_seconds: int = 600) -> None:
        """10 dakikadir kullanilmayan session'larin sayacini sil."""
        now = time.time()
        cutoff = now - max_age_seconds
        with self._lock:
            stale = [
                sid
                for sid, bucket in self._buckets.items()
                if not bucket or bucket[-1] < cutoff
            ]
            for sid in stale:
                self._buckets.pop(sid, None)


session_limiter = SessionRateLimiter()
