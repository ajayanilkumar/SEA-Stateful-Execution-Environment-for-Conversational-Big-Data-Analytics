# sea/core/session.py
"""
SessionManager – manages the lifecycle of analytical sessions.

A session encapsulates:
    - The Planner's memory M (sliding window of raw/enriched query history)
    - Both DAG state representations (delegated to DualStateStore)

Session lifecycle (Section 3.4 of the paper):
    - A session is a continuous dialogue on a single analytical topic.
    - When the Planner selects the root node (topic switch), a purge event
      clears all state and history, seeding memory with only the new query.
"""

from __future__ import annotations

from typing import Dict

from sea.core.schemas import HistoryItem, SessionData
from sea.core.state import DualStateStore

# Global in-memory history store.  Swap for a persistent backend if needed.
_HISTORY_STORE: Dict[str, SessionData] = {}  # session_id -> SessionData

_HISTORY_WINDOW = 3  # sliding-window size (matches paper implementation)


class SessionManager:
    """
    Manages per-session conversation history and delegates state I/O to
    DualStateStore.
    """

    def __init__(self, state_store: DualStateStore | None = None) -> None:
        self._store = state_store or DualStateStore()

    @property
    def state_store(self) -> DualStateStore:
        return self._store

    # ------------------------------------------------------------------
    # Session retrieval
    # ------------------------------------------------------------------

    def get_session(self, session_id: str) -> SessionData:
        """Return the SessionData for a session, creating it if absent."""
        if session_id not in _HISTORY_STORE:
            _HISTORY_STORE[session_id] = SessionData()
        session = _HISTORY_STORE[session_id]
        # Attach live DAG snapshots for read-only consumers (e.g. Planner)
        session.main_dag_state = self._store.get_main_dag(session_id)
        session.summary_dag_state = self._store.get_summary_dag(session_id)
        return session

    # ------------------------------------------------------------------
    # History management (Planner memory M)
    # ------------------------------------------------------------------

    def append_history(
        self, session_id: str, *, raw_query: str, enriched_query: str
    ) -> None:
        """
        Append a new turn to the session's conversation history.

        Implements the sliding window: older turns beyond the window size are
        dropped to prevent unbounded context growth.
        """
        session = _HISTORY_STORE.setdefault(session_id, SessionData())
        session.conversation_history.append(
            HistoryItem(raw=raw_query, enriched=enriched_query)
        )
        if len(session.conversation_history) > _HISTORY_WINDOW:
            session.conversation_history.pop(0)

    # ------------------------------------------------------------------
    # Purge event (Algorithm 1, lines 12-16)
    # ------------------------------------------------------------------

    def clear_session(self, session_id: str) -> None:
        """
        Purge all state and history for a session (topic-switch reset).

        The caller is responsible for re-seeding history with the current
        query after calling this method.
        """
        _HISTORY_STORE.pop(session_id, None)
        self._store.clear(session_id)
