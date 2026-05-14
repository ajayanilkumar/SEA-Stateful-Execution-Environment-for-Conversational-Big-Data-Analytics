# sea/__init__.py
"""
SEA – Stateful Execution Environment for Conversational Big Data Analytics.

Quick start (GloboMart analytics workflow):
-------------------------------------------
    from sea.workflows.analytics import build_analytics_sea

    sea = build_analytics_sea(
        google_api_key="...",
        databricks_hostname="dbc-xxx.cloud.databricks.com",
        databricks_http_path="/sql/1.0/warehouses/...",
        databricks_token="...",
        databricks_table_prefix="workspace.globomart_db_final",
        vector_db_path="./vector_db_globomart",
        vector_collection="globomart_sea",
        json_summary_dir="./json_summaries",
    )

    result = sea.chat("Show me the top 5 products last quarter.", session_id="s1")

Custom graph:
-------------
    from sea import SEA, SEAGraph, Tool, ToolResult
    from sea.core.planner import GeminiPlanner

    class MyTool(Tool):
        name = "my_tool"
        description = "Does something useful."

        def run(self, query, upstream_artifacts, **kwargs):
            ...
            return ToolResult(artifact={...}, summary={...})

    graph = SEAGraph()
    graph.add_node("my_tool", tool=MyTool())

    sea = SEA(graph=graph, planner=GeminiPlanner(api_key="..."))
    result = sea.chat("Hello", session_id="s1")
"""

from __future__ import annotations

import logging
from typing import Optional

from sea.core.dag import SEAGraph
from sea.core.executor import Executor
from sea.core.planner import GeminiPlanner, Planner
from sea.core.schemas import Plan, ToolResult
from sea.core.session import SessionManager
from sea.core.tool import Tool

logger = logging.getLogger(__name__)

__version__ = "0.1.0"

__all__ = [
    "SEA",
    "SEAGraph",
    "Tool",
    "ToolResult",
    "Planner",
    "GeminiPlanner",
    "SessionManager",
]


class SEA:
    """
    Stateful Execution Environment.

    Orchestrates the Planner–Executor loop described in Algorithm 1 of the
    paper.  Manages session lifecycle including the purge event triggered
    when the Planner selects the root node (topic switch).

    Args:
        graph:           A configured SEAGraph (built via the builder API or a
                         workflow preset).
        planner:         Any Planner subclass (e.g. GeminiPlanner).
        session_manager: Optional; a new SessionManager is created if not
                         supplied.
    """

    def __init__(
        self,
        graph: SEAGraph,
        planner: Planner,
        *,
        session_manager: Optional[SessionManager] = None,
    ) -> None:
        self.graph = graph
        self.planner = planner
        self.session_manager = session_manager or SessionManager()
        self._executor = Executor(graph=graph, session_manager=self.session_manager)

    # ------------------------------------------------------------------
    # Main conversational turn (Algorithm 1)
    # ------------------------------------------------------------------

    def chat(self, query: str, session_id: str) -> dict:
        """
        Process one conversational turn.

        Implements the full operational loop from Algorithm 1:
          1. Update Planner memory M.
          2. Planner selects v_entry and generates plan P_t.
          3. State carry-forward (S_t = S_{t-1}).
          4. Executor runs each tool, updating both DAG representations.
          5. Session lifecycle check: purge if v_entry == v0.

        Args:
            query:      Raw user query q_t.
            session_id: Identifies the conversation session.

        Returns:
            dict with:
                tool_calls        – ordered list of executed node_ids
                final_dag_state   – Twin Summary DAG (S') after this turn
                execution_summary – per-node SUCCESS/ERROR status
        """
        session = self.session_manager.get_session(session_id)

        # --- Planner phase ---
        plan: Plan = self.planner.plan(query=query, session=session, graph=self.graph)
        logger.info("Plan start_node='%s', steps=%s",
                    plan.start_node, [s.node_id for s in plan.execution_plan])

        # --- Append to history (M ← M ∪ {q_t}) ---
        self.session_manager.append_history(
            session_id, raw_query=query, enriched_query=plan.enriched_query
        )

        # --- Session lifecycle: purge on topic switch (v_entry == v0) ---
        if plan.start_node == self.graph.root and session.main_dag_state:
            logger.info(
                "PURGE EVENT: topic switch detected for session '%s'. "
                "Resetting state and history.", session_id
            )
            self.session_manager.clear_session(session_id)
            # Re-seed history with the triggering query
            self.session_manager.append_history(
                session_id, raw_query=query, enriched_query=plan.enriched_query
            )

        # --- Executor phase ---
        execution_summary = self._executor.execute(plan=plan, session_id=session_id)

        # --- Collect final state ---
        final_session = self.session_manager.get_session(session_id)

        return {
            "tool_calls": [step.node_id for step in plan.execution_plan],
            "final_dag_state": final_session.summary_dag_state,
            "execution_summary": execution_summary,
        }
