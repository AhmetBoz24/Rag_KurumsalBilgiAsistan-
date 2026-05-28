"""Konusma gecmisi yoneticisi.

In-memory, TTL'li, LRU-benzeri.
- TTL: SESSION_TTL_SECONDS inaktiflikten sonra session silinir
- Max: SESSION_MAX_COUNT'i asarsa en eski son-erisim olan silinir
- Turn limit: MAX_HISTORY_TURNS * 2 mesajdan fazlasi tutulmaz
"""
from __future__ import annotations
import time
import uuid
from collections import OrderedDict
from threading import Lock
from typing import Dict, List, Optional

from core.config import (
    SESSION_TTL_SECONDS,
    SESSION_MAX_COUNT,
    MAX_HISTORY_TURNS,
)


class SessionStore:
    def __init__(
        self,
        ttl: int = SESSION_TTL_SECONDS,
        max_sessions: int = SESSION_MAX_COUNT,
        max_turns: int = MAX_HISTORY_TURNS,
    ):
        self._ttl = ttl
        self._max_sessions = max_sessions
        self._max_msgs = max_turns * 2
        self._data: "OrderedDict[str, Dict]" = OrderedDict()
        self._lock = Lock()

    def _now(self) -> float:
        return time.time()

    def _evict_expired(self) -> None:
        now = self._now()
        expired = [sid for sid, s in self._data.items() if now - s["last_seen"] > self._ttl]
        for sid in expired:
            self._data.pop(sid, None)

    def _evict_overflow(self) -> None:
        while len(self._data) > self._max_sessions:
            self._data.popitem(last=False)  # en eski

    def new_session(self) -> str:
        sid = str(uuid.uuid4())
        with self._lock:
            self._data[sid] = {
                "messages": [],
                "created_at": self._now(),
                "last_seen": self._now(),
            }
            self._evict_overflow()
        return sid

    def get_or_create(self, sid: Optional[str]) -> str:
        if sid and sid in self._data:
            with self._lock:
                if sid in self._data:
                    self._data[sid]["last_seen"] = self._now()
                    self._data.move_to_end(sid)
                    return sid
        return self.new_session()

    def get_history(self, sid: str) -> List[Dict]:
        with self._lock:
            self._evict_expired()
            session = self._data.get(sid)
            if not session:
                return []
            session["last_seen"] = self._now()
            return list(session["messages"])

    def append(self, sid: str, role: str, content: str) -> None:
        with self._lock:
            if sid not in self._data:
                self._data[sid] = {
                    "messages": [],
                    "created_at": self._now(),
                    "last_seen": self._now(),
                }
            session = self._data[sid]
            session["messages"].append({"role": role, "content": content})
            if len(session["messages"]) > self._max_msgs:
                session["messages"] = session["messages"][-self._max_msgs :]
            session["last_seen"] = self._now()
            self._data.move_to_end(sid)
            self._evict_overflow()

    def delete(self, sid: str) -> bool:
        with self._lock:
            return self._data.pop(sid, None) is not None

    def active_count(self) -> int:
        with self._lock:
            self._evict_expired()
            return len(self._data)

    def stats(self) -> Dict:
        with self._lock:
            self._evict_expired()
            return {
                "active_sessions": len(self._data),
                "total_messages": sum(len(s["messages"]) for s in self._data.values()),
            }


session_store = SessionStore()
