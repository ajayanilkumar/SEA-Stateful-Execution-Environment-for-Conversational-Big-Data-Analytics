# sea/core/executor.py
"""
Deterministic Executor (X) – Section 3.3 of the paper.

The Executor receives the Planner's Plan and deterministically walks the
execution groups in order, calling each Tool with its upstream artifacts and
writing results to both DAG representations.

Key behaviours (Algorithm 1, lines 7-11):
- Sequential execution of dependency-ordered groups.
- Parallel execution of independent siblings within the same depth group
  (e.g. ChartGenerator ∥ InsightSummarizer).
- On any tool error the sequential chain halts (parallel siblings still
  complete; only further sequential steps are skipped).
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
from typing import TYPE_CHECKING, Dict, List

from sea.core.schemas import Plan, Step, ToolResult
from sea.core.session import SessionManager

if TYPE_CHECKING:
    from sea.core.dag import SEAGraph

logger = logging.getLogger(__name__)


class Executor:
    """
    Stateful executor that runs a Plan against a SEAGraph and updates both
    DAG representations through the SessionManager's DualStateStore.
    """

    def __init__(self, graph: "SEAGraph", session_manager: SessionManager) -> None:
        self._graph = graph
        self._sm = session_manager

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def execute(self, plan: Plan, session_id: str) -> Dict[str, str]:
        """
        Execute a Plan for one conversational turn.

        Returns a dict mapping node_id -> "SUCCESS" | "ERROR: <msg>".
        """
        # Compute execution order as parallel groups from the start node.
        groups: List[List[str]] = self._graph.execution_groups(plan.start_node)

        # Filter groups to only nodes listed in the plan, preserving order.
        planned_nodes = {step.node_id for step in plan.execution_plan}
        step_index: Dict[str, Step] = {s.node_id: s for s in plan.execution_plan}

        results: Dict[str, str] = {}

        for group in groups:
            group_nodes = [nid for nid in group if nid in planned_nodes]
            if not group_nodes:
                continue

            if len(group_nodes) == 1:
                node_id = group_nodes[0]
                outcome = self._run_node(node_id, step_index[node_id], session_id, plan.enriched_query)
                results.update(outcome)
                if "ERROR" in list(outcome.values())[0]:
                    logger.warning("Halting execution after error in node '%s'.", node_id)
                    break
            else:
                # Parallel group
                logger.info("Executing parallel group: %s", group_nodes)
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    futures = {
                        pool.submit(
                            self._run_node, nid, step_index[nid], session_id, plan.enriched_query
                        ): nid
                        for nid in group_nodes
                    }
                    for future in concurrent.futures.as_completed(futures):
                        results.update(future.result())

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_node(
        self, node_id: str, step: Step, session_id: str, enriched_query: str
    ) -> Dict[str, str]:
        """Execute a single node, update both DAGs, return status dict."""
        logger.info("Executing node: %s", node_id)
        tool = self._graph.get_tool(node_id)
        store = self._sm.state_store

        # Build upstream artifact dict for this tool.
        deps = self._graph.dependencies_of(node_id)
        upstream_artifacts = {dep: store.get_artifact(session_id, dep) for dep in deps}

        # Parse extra kwargs from the planner's Step.inputs JSON string.
        try:
            extra_kwargs = json.loads(step.inputs) if step.inputs else {}
            if not isinstance(extra_kwargs, dict):
                extra_kwargs = {}
        except (json.JSONDecodeError, ValueError):
            extra_kwargs = {}

        try:
            result: ToolResult = tool.run(
                query=enriched_query,
                upstream_artifacts=upstream_artifacts,
                **extra_kwargs,
            )
            store.update(session_id, node_id, result)
            logger.info("Node '%s' completed successfully.", node_id)
            return {node_id: "SUCCESS"}
        except Exception as exc:
            error_msg = f"ERROR: {exc}"
            logger.exception("Node '%s' failed: %s", node_id, exc)
            store.update_error(session_id, node_id, error_msg)
            return {node_id: error_msg}
