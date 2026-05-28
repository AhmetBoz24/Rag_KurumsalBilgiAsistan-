"""Servis istatistikleri (admin paneli icin).

Hafif in-memory metrik toplama. Restart'ta sifirlanir.
"""
from __future__ import annotations
import time
from collections import deque
from threading import Lock
from typing import Deque, Dict


class StatsCollector:
    def __init__(self):
        self._chat_requests = 0
        self._stream_requests = 0
        self._rejected_input = 0
        self._rejected_rate_limit = 0
        self._rejected_budget = 0
        self._errors = 0
        self._latencies: Deque[int] = deque(maxlen=1000)
        self._started_at = time.time()
        self._lock = Lock()

    def record_chat(self, latency_ms: int) -> None:
        with self._lock:
            self._chat_requests += 1
            self._latencies.append(latency_ms)

    def record_stream(self, latency_ms: int) -> None:
        with self._lock:
            self._stream_requests += 1
            self._latencies.append(latency_ms)

    def record_rejected_input(self) -> None:
        with self._lock:
            self._rejected_input += 1

    def record_rejected_rate(self) -> None:
        with self._lock:
            self._rejected_rate_limit += 1

    def record_rejected_budget(self) -> None:
        with self._lock:
            self._rejected_budget += 1

    def record_error(self) -> None:
        with self._lock:
            self._errors += 1

    def stats(self) -> Dict:
        with self._lock:
            n = len(self._latencies)
            avg_latency = int(sum(self._latencies) / n) if n else 0
            sorted_lat = sorted(self._latencies)
            p95 = sorted_lat[int(n * 0.95) - 1] if n >= 20 else (sorted_lat[-1] if n else 0)
            uptime = int(time.time() - self._started_at)
            return {
                "uptime_seconds": uptime,
                "chat_requests": self._chat_requests,
                "stream_requests": self._stream_requests,
                "rejected_input_guard": self._rejected_input,
                "rejected_rate_limit": self._rejected_rate_limit,
                "rejected_budget": self._rejected_budget,
                "errors": self._errors,
                "avg_latency_ms": avg_latency,
                "p95_latency_ms": p95,
            }


stats = StatsCollector()
