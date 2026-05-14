# sea/core/dag.py
"""
SEAGraph – user-facing builder for the Stateful Execution DAG.

The graph encodes the analytical workflow as a Directed Acyclic Graph (DAG)
where each node corresponds to a Tool and directed edges represent immutable
data dependencies (Section 3.2 of the paper).

Usage
-----
    from sea.core.dag import SEAGraph
    from my_tools import SchemaRetriever, QuerySynthesizer

    graph = SEAGraph()
    graph.add_node("schema_retriever", tool=SchemaRetriever())
    graph.add_node("query_synthesizer", tool=QuerySynthesizer(),
                   depends_on=["schema_retriever"])
    graph.add_node("ai_analytics",     tool=AIAnalytics(),
                   depends_on=["query_synthesizer"])
    graph.add_node("chart_generator",  tool=ChartGenerator(),
                   depends_on=["ai_analytics"])
    graph.add_node("insight_summarizer", tool=InsightSummarizer(),
                   depends_on=["ai_analytics"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from sea.core.tool import Tool


@dataclass
class Node:
    """A vertex V in the Stateful Execution DAG."""
    node_id: str
    tool: Tool
    depends_on: List[str] = field(default_factory=list)


class SEAGraph:
    """
    Directed Acyclic Graph representing the analytical workflow.

    Key responsibilities
    --------------------
    - Store nodes (tools) and their dependency edges.
    - Provide the root node (v0) whose selection by the Planner triggers a
      purge event (session reset).
    - Compute execution order from any entry-point (v_entry) as an ordered
      list of parallel-execution groups so the Executor can parallelise
      independent siblings (e.g. ChartGenerator ∥ InsightSummarizer).
    - Surface a node-description dict for the Planner's system prompt.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, Node] = {}

    # ------------------------------------------------------------------
    # Builder API
    # ------------------------------------------------------------------

    def add_node(
        self,
        node_id: str,
        tool: Tool,
        depends_on: Optional[List[str]] = None,
    ) -> "SEAGraph":
        """
        Register a tool as a node in the DAG.

        Args:
            node_id:    Unique identifier for this node.
            tool:       Tool instance that will be called by the Executor.
            depends_on: List of node_ids that must complete before this node.

        Returns:
            self – for method chaining.
        """
        if node_id in self._nodes:
            raise ValueError(f"Node '{node_id}' already exists in the graph.")
        for dep in (depends_on or []):
            if dep not in self._nodes:
                raise ValueError(
                    f"Dependency '{dep}' not yet registered. "
                    "Add nodes in dependency order."
                )
        self._nodes[node_id] = Node(node_id=node_id, tool=tool, depends_on=depends_on or [])
        return self

    def add_edge(self, from_node: str, to_node: str) -> "SEAGraph":
        """
        Declare that `to_node` depends on `from_node`.
        Both nodes must already be registered.
        """
        if from_node not in self._nodes:
            raise ValueError(f"Node '{from_node}' not registered.")
        if to_node not in self._nodes:
            raise ValueError(f"Node '{to_node}' not registered.")
        if from_node not in self._nodes[to_node].depends_on:
            self._nodes[to_node].depends_on.append(from_node)
        return self

    # ------------------------------------------------------------------
    # Graph introspection
    # ------------------------------------------------------------------

    @property
    def nodes(self) -> Dict[str, Node]:
        return self._nodes

    @property
    def root(self) -> str:
        """
        The root node (v0) – the unique node with no dependencies.
        Selecting this as v_entry triggers a session purge (Algorithm 1, line 12).
        """
        roots = [nid for nid, node in self._nodes.items() if not node.depends_on]
        if len(roots) != 1:
            raise ValueError(
                f"SEAGraph must have exactly one root node; found {roots}."
            )
        return roots[0]

    def get_tool(self, node_id: str) -> Tool:
        return self._nodes[node_id].tool

    def dependencies_of(self, node_id: str) -> List[str]:
        return self._nodes[node_id].depends_on

    # ------------------------------------------------------------------
    # Execution-order computation
    # ------------------------------------------------------------------

    def subgraph_from(self, start_node_id: str) -> List[str]:
        """
        Return a topologically-sorted list of ALL nodes reachable from
        `start_node_id` (including itself).

        This is the full execution subgraph G(v_entry) described in the paper.
        """
        if start_node_id not in self._nodes:
            raise ValueError(f"Node '{start_node_id}' not in graph.")
        visited: List[str] = []
        self._dfs(start_node_id, set(), visited)
        return visited

    def _dfs(self, node_id: str, seen: Set[str], result: List[str]) -> None:
        if node_id in seen:
            return
        seen.add(node_id)
        # Only recurse into children that start at or after start_node in the
        # topological order.  Since we build the graph in dep order this is
        # equivalent to forward BFS/DFS from the start node.
        for child_id, child_node in self._nodes.items():
            if node_id in child_node.depends_on and child_id not in seen:
                self._dfs(child_id, seen, result)
        result.insert(0, node_id)  # prepend to get topo order

    def execution_groups(self, start_node_id: str) -> List[List[str]]:
        """
        Return the subgraph as ordered *groups* of node_ids.

        Nodes within the same group share the same dependency depth and can
        be executed in parallel by the Executor (e.g. ChartGenerator and
        InsightSummarizer both depend on AIAnalytics and can run concurrently).

        Example for the analytics DAG starting from root:
            [["schema_retriever"],
             ["query_synthesizer"],
             ["ai_analytics"],
             ["chart_generator", "insight_summarizer"]]
        """
        ordered = self.subgraph_from(start_node_id)
        depth: Dict[str, int] = {}

        for nid in ordered:
            preds = [p for p in self._nodes[nid].depends_on if p in ordered]
            depth[nid] = (max(depth[p] for p in preds) + 1) if preds else 0

        max_depth = max(depth.values(), default=0)
        groups: List[List[str]] = [[] for _ in range(max_depth + 1)]
        for nid in ordered:
            groups[depth[nid]].append(nid)
        return [g for g in groups if g]

    # ------------------------------------------------------------------
    # Planner context
    # ------------------------------------------------------------------

    @property
    def node_descriptions(self) -> Dict[str, str]:
        """node_id -> description string, used to build the Planner prompt."""
        return {nid: node.tool.description for nid, node in self._nodes.items()}

    def dependency_description(self) -> str:
        """Human-readable dependency block for the Planner system prompt."""
        lines = []
        for nid, node in self._nodes.items():
            deps = node.depends_on
            dep_str = ", ".join(f"`{d}`" for d in deps) if deps else "None"
            lines.append(f"- `{nid}`: Requires {dep_str}.")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        nodes_repr = ", ".join(self._nodes.keys())
        return f"SEAGraph(nodes=[{nodes_repr}])"
