"""Gunluk token butcesi.

Groq quota'sini patlatmamak icin gunluk toplam token sayisini izle.
Limit asildiginda yeni istekleri reddet.
"""
from __future__ import annotations
import time
from threading import Lock
from typing import Dict

import tiktoken

from core.config import DAILY_TOKEN_BUDGET


class TokenBudget:
    def __init__(self, daily_budget: int = DAILY_TOKEN_BUDGET):
        self.daily_budget = daily_budget
        self._used = 0
        self._day_key = self._today()
        self._lock = Lock()
        # Llama 3 icin cl100k_base yaklasik dogru (tiktoken'da llama yok)
        self._encoder = tiktoken.get_encoding("cl100k_base")

    @staticmethod
    def _today() -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())

    def _maybe_rollover(self) -> None:
        today = self._today()
        if today != self._day_key:
            self._day_key = today
            self._used = 0

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            return len(self._encoder.encode(text))
        except Exception:
            return len(text) // 4  # kaba fallback

    def can_spend(self, estimated_tokens: int) -> bool:
        with self._lock:
            self._maybe_rollover()
            return (self._used + estimated_tokens) <= self.daily_budget

    def add(self, tokens: int) -> None:
        with self._lock:
            self._maybe_rollover()
            self._used += max(0, tokens)

    def stats(self) -> Dict:
        with self._lock:
            self._maybe_rollover()
            remaining = max(0, self.daily_budget - self._used)
            return {
                "date": self._day_key,
                "used": self._used,
                "budget": self.daily_budget,
                "remaining": remaining,
                "usage_ratio": round(self._used / self.daily_budget, 4)
                if self.daily_budget > 0
                else 0,
            }


token_budget = TokenBudget()
