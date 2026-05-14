# sea/core/tool.py
"""
Tool protocol for the SEA framework.

Every analytics tool (SchemaRetriever, QuerySynthesizer, …) is a subclass of
Tool.  The executor calls Tool.run() and automatically stores the returned
ToolResult in both DAG representations – callers never touch the state store
directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from sea.core.schemas import ToolResult


class Tool(ABC):
    """
    Base class for a node in the SEAGraph.

    Subclasses must implement:
        name        – unique string identifier (matches the node_id in SEAGraph)
        description – one-line description shown to the Planner in its prompt
        run()       – executes the tool and returns a ToolResult
    """

    # --- identity (override in subclass) ---

    @property
    def name(self) -> str:
        raise NotImplementedError("Tool subclasses must set a `name` property.")

    @property
    def description(self) -> str:
        """Shown to the Planner so it can decide when to call this node."""
        return f"Tool: {self.name}"

    # --- core method ---

    @abstractmethod
    def run(
        self,
        query: str,
        upstream_artifacts: Dict[str, Dict[str, Any]],
        **kwargs: Any,
    ) -> ToolResult:
        """
        Execute the tool.

        Args:
            query:              The enriched user query for this turn.
            upstream_artifacts: Artifacts from all directly-upstream nodes,
                                keyed by node_id.  The tool extracts whatever
                                it needs from these dicts – no framework-level
                                path mapping required.
            **kwargs:           Extra key/value pairs forwarded from the
                                planner's Step.inputs JSON for this node.

        Returns:
            ToolResult with:
                artifact – full computational output (stored in Main State DAG)
                summary  – compact metadata (stored in Twin Summary DAG, given
                           to the Planner on the next turn)
        """

    # --- optional helpers ---

    def build_summary(self, artifact: Dict[str, Any]) -> Dict[str, Any]:
        """
        Default summary builder – subclasses can override for richer summaries.
        The default returns only a SUCCESS status so the Planner knows the node
        completed.
        """
        return {"status": "SUCCESS"}
