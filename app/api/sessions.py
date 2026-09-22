"""In-memory session store: one active dataset and one Agent per browser tab.

Deliberately small and deliberately not persistent. A session is a UUID the
client generates; the server creates one the first time a dataset is loaded
under that id. Sessions expire so a long-running server does not accumulate
DataFrames, and the oldest is dropped when the cap is reached.

The four methods below are the whole interface, so replacing this with Redis
later is a single-file change.
"""

import threading
import time
from dataclasses import dataclass, field

from app.agent import Agent

MAX_SESSIONS = 50
TTL_SECONDS = 2 * 60 * 60


@dataclass
class Session:
    agent: Agent | None = None
    last_used: float = field(default_factory=time.time)


class SessionStore:
    def __init__(self, max_sessions: int = MAX_SESSIONS, ttl_seconds: int = TTL_SECONDS) -> None:
        self._sessions: dict[str, Session] = {}
        self._max = max_sessions
        self._ttl = ttl_seconds
        self._lock = threading.Lock()  # handlers run in a threadpool

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            self._expire()
            session = self._sessions.get(session_id)
            if session:
                session.last_used = time.time()
            return session

    def get_or_create(self, session_id: str) -> Session:
        with self._lock:
            self._expire()
            session = self._sessions.get(session_id)
            if session is None:
                session = self._sessions[session_id] = Session()
                self._evict_if_full()
            session.last_used = time.time()
            return session

    def discard(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def _expire(self) -> None:
        cutoff = time.time() - self._ttl
        for session_id in [k for k, v in self._sessions.items() if v.last_used < cutoff]:
            del self._sessions[session_id]

    def _evict_if_full(self) -> None:
        while len(self._sessions) > self._max:
            oldest = min(self._sessions, key=lambda k: self._sessions[k].last_used)
            del self._sessions[oldest]
