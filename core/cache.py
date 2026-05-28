"""Q&A LRU cache.

Ayni soru icin LLM'i tekrar tekrar cagirma. Sadece STATELESS sorularda kullan
(yani gecmis baglami yokken).
"""
from __future__ import annotations
import hashlib
from typing import Any, Dict, Optional, Tuple

from cachetools import TTLCache

from core.config import CACHE_TTL_SECONDS, CACHE_MAX_SIZE


class QACache:
    def __init__(self):
        self._cache: TTLCache = TTLCache(
            maxsize=CACHE_MAX_SIZE, ttl=CACHE_TTL_SECONDS
        )
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _key(question: str) -> str:
        return hashlib.sha256(question.lower().strip().encode("utf-8")).hexdigest()

    def get(self, question: str) -> Optional[Tuple[str, list]]:
        key = self._key(question)
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def set(self, question: str, answer: str, sources: list) -> None:
        self._cache[self._key(question)] = (answer, sources)

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        rate = (self._hits / total) if total > 0 else 0.0
        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(rate, 3),
        }


qa_cache = QACache()
