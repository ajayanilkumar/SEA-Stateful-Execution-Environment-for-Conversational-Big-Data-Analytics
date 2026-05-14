# sea/core/state.py
"""
DualStateStore – in-memory implementation of the Dual-Representation State
Model described in Section 3.2 of the paper.

    Main State DAG  (S)  – caches the full, heavy computational artifact from
                           each node's last execution (DataFrames, image bytes,
                           etc.).  Accessed only by the Executor and tools.

    Twin Summary DAG (S') – lightweight symbolic metadata for each node.
                            This is what the Planner receives in its context
                            window, keeping token usage low.

Both representations are stored per session_id so multiple concurrent
conversations are fully isolated.
"""

from __future__ import annotations

from typing import Any, Dict

from sea.core.schemas import ToolResult


# Global in-memory store (equivalent to SESSION_STORE in the original code).
# Replace with a persistent backend (Redis, Postgres JSON, etc.) if needed.
_MAIN_STORE: Dict[str, Dict[str, Any]] = {}       # session_id -> node_id -> artifact
_SUMMARY_STORE: Dict[str, Dict[str, Any]] = {}    # session_id -> node_id -> summary


class DualStateStore:
    """
    Thread-safe (GIL-protected for CPython) in-memory dual-representation state
    store.  Exposes a simple read/write/clear interface used by SessionManager
    and the Executor.
    """

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def update(self, session_id: str, node_id: str, result: ToolResult) -> None:
        """
        Store both representations for a node after it executes.

        Called by the Executor after each successful tool invocation
        (Algorithm 1, lines 9-10).
        """
        _MAIN_STORE.setdefault(session_id, {})[node_id] = result.artifact
        _SUMMARY_STORE.setdefault(session_id, {})[node_id] = result.summary

    def update_error(self, session_id: str, node_id: str, error_msg: str) -> None:
        """Record a failed node execution in both DAGs."""
        _MAIN_STORE.setdefault(session_id, {})[node_id] = {"error": error_msg}
        _SUMMARY_STORE.setdefault(session_id, {})[node_id] = {"status": "ERROR", "error": error_msg}

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_artifact(self, session_id: str, node_id: str) -> Dict[str, Any]:
        """Fetch the full artifact from the Main State DAG."""
        return _MAIN_STORE.get(session_id, {}).get(node_id, {})

    def get_summary(self, session_id: str, node_id: str) -> Dict[str, Any]:
        """Fetch the lightweight summary from the Twin Summary DAG."""
        return _SUMMARY_STORE.get(session_id, {}).get(node_id, {})

    def get_main_dag(self, session_id: str) -> Dict[str, Any]:
        """Full Main State DAG for a session."""
        return dict(_MAIN_STORE.get(session_id, {}))

    def get_summary_dag(self, session_id: str) -> Dict[str, Any]:
        """Full Twin Summary DAG for a session (given to the Planner)."""
        return dict(_SUMMARY_STORE.get(session_id, {}))

    def has_artifact(self, session_id: str, node_id: str) -> bool:
        return node_id in _MAIN_STORE.get(session_id, {})

    # ------------------------------------------------------------------
    # Clear (Purge Event)
    # ------------------------------------------------------------------

    def clear(self, session_id: str) -> None:
        """
        Purge all state for a session.

        Triggered when the Planner selects the root node as v_entry, indicating
        a topic switch (Algorithm 1, line 14).
        """
        _MAIN_STORE.pop(session_id, None)
        _SUMMARY_STORE.pop(session_id, None)
