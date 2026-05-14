"""
examples/minimal_local.py
--------------------------
Demonstrates the SEA module without any external services (no Databricks,
no ChromaDB, no API keys required).

All tools are replaced with lightweight mock implementations so the full
Planner–Executor–StateDAG loop can be exercised end-to-end on any machine.

Run with:
    cd /path/to/SEA
    pip install -e .
    python examples/minimal_local.py
"""

from __future__ import annotations

import json
from typing import Any, Dict

from sea import SEA, SEAGraph, Tool, ToolResult
from sea.core.planner import Planner
from sea.core.schemas import Plan, SessionData, Step


# ---------------------------------------------------------------------------
# Mock tools – no external dependencies
# ---------------------------------------------------------------------------

class MockSchemaRetriever(Tool):
    name = "schema_retriever"
    description = (
        "Finds the relevant table schemas. No dependencies – always the first step."
    )

    def run(self, query: str, upstream_artifacts: Dict[str, Any], **kwargs) -> ToolResult:
        print(f"  [SchemaRetriever] query='{query}'")
        artifact = {
            "selected_tables": ["sales", "products"],
            "reasoning": "Query mentions sales figures which live in these two tables.",
            "table_summaries": {
                "sales":    {"name": "sales",    "fields": ["sale_id", "product_id", "revenue"]},
                "products": {"name": "products", "fields": ["product_id", "product_name", "category"]},
            },
        }
        return ToolResult(
            artifact=artifact,
            summary={
                "status": "SUCCESS",
                "selected_tables": artifact["selected_tables"],
                "reasoning": artifact["reasoning"],
            },
        )


class MockQuerySynthesizer(Tool):
    name = "query_synthesizer"
    description = (
        "Generates and runs SQL to fetch a data subset. "
        "Requires schema_retriever output."
    )

    def run(self, query: str, upstream_artifacts: Dict[str, Any], **kwargs) -> ToolResult:
        schema = upstream_artifacts.get("schema_retriever", {})
        tables = schema.get("selected_tables", [])
        print(f"  [QuerySynthesizer] tables={tables}, query='{query}'")

        # Simulate a small result DataFrame as JSON
        fake_data = json.dumps([
            {"product_name": "Widget A", "category": "Electronics", "revenue": 15000},
            {"product_name": "Gadget B", "category": "Electronics", "revenue": 12000},
            {"product_name": "Doohickey C", "category": "Accessories", "revenue": 8500},
        ])
        artifact = {
            "sql_query": f"SELECT * FROM sales JOIN products USING (product_id) WHERE ...",
            "sql_result": {"status": "success", "data": fake_data, "row_count": 3},
        }
        return ToolResult(
            artifact=artifact,
            summary={
                "status": "SUCCESS",
                "sql_query": artifact["sql_query"],
                "row_count": 3,
            },
        )


class MockAIAnalytics(Tool):
    name = "ai_analytics"
    description = (
        "Analyses the SQL subset with Python/pandas. "
        "Requires query_synthesizer output."
    )

    def run(self, query: str, upstream_artifacts: Dict[str, Any], **kwargs) -> ToolResult:
        sql_art = upstream_artifacts.get("query_synthesizer", {})
        row_count = sql_art.get("sql_result", {}).get("row_count", 0)
        print(f"  [AIAnalytics] analysing {row_count} rows, query='{query}'")

        result_data = json.dumps([
            {"product_name": "Widget A", "revenue": 15000},
            {"product_name": "Gadget B", "revenue": 12000},
            {"product_name": "Doohickey C", "revenue": 8500},
        ])
        artifact = {
            "result_df": result_data,
            "result_df_summary": {"name": "analysis_result", "fields": ["product_name", "revenue"]},
            "code_generated": "transformed_df = data.sort_values('revenue', ascending=False)",
        }
        return ToolResult(
            artifact=artifact,
            summary={
                "status": "SUCCESS",
                "analysis_code": artifact["code_generated"],
                "result_summary": artifact["result_df_summary"],
            },
        )


class MockChartGenerator(Tool):
    name = "chart_generator"
    description = (
        "Generates charts from the final DataFrame. "
        "Requires ai_analytics output. Runs in parallel with insight_summarizer."
    )

    def run(self, query: str, upstream_artifacts: Dict[str, Any], **kwargs) -> ToolResult:
        print(f"  [ChartGenerator] query='{query}'")
        artifact = {
            "charts": [{"viz_type": "bar", "status": "SUCCESS", "raster": "<base64_png_placeholder>"}]
        }
        return ToolResult(
            artifact=artifact,
            summary={"status": "SUCCESS", "charts_generated": ["bar"]},
        )


class MockInsightSummarizer(Tool):
    name = "insight_summarizer"
    description = (
        "Generates a narrative insight summary. "
        "Requires ai_analytics output. Runs in parallel with chart_generator."
    )

    def run(self, query: str, upstream_artifacts: Dict[str, Any], **kwargs) -> ToolResult:
        print(f"  [InsightSummarizer] query='{query}'")
        commentary = (
            "Widget A leads in revenue at $15,000, followed by Gadget B ($12,000). "
            "Electronics dominate the top performers, accounting for 75% of total revenue."
        )
        artifact = {"commentary": commentary}
        return ToolResult(
            artifact=artifact,
            summary={"status": "SUCCESS", "commentary_snippet": commentary[:120]},
        )


# ---------------------------------------------------------------------------
# Mock Planner (deterministic – no LLM required)
# ---------------------------------------------------------------------------

class MockPlanner(Planner):
    """
    Deterministic planner for local testing.

    First query → starts from root (schema_retriever).
    Follow-up containing 'chart' → starts from chart_generator only.
    Follow-up containing 'change' or 'filter' → starts from ai_analytics.
    Everything else → starts from ai_analytics (reuse SQL subset).
    """

    def plan(self, query: str, session: SessionData, graph: SEAGraph) -> Plan:
        q = query.lower()
        has_state = bool(session.main_dag_state)

        if not has_state or any(kw in q for kw in ("new topic", "marketing", "shipment")):
            start = "schema_retriever"
        elif "chart" in q or "visuali" in q:
            start = "chart_generator"
        elif any(kw in q for kw in ("filter", "change", "only", "just")):
            start = "ai_analytics"
        else:
            start = "ai_analytics"

        subgraph = graph.subgraph_from(start)
        steps = [Step(node_id=nid, inputs="{}") for nid in subgraph]
        print(f"\n[MockPlanner] start_node='{start}', plan={subgraph}")

        return Plan(
            reasoning=f"Starting from '{start}' based on query context.",
            enriched_query=query,
            start_node=start,
            execution_plan=steps,
        )


# ---------------------------------------------------------------------------
# Wire the graph and run
# ---------------------------------------------------------------------------

def main():
    graph = SEAGraph()
    graph.add_node("schema_retriever",   tool=MockSchemaRetriever())
    graph.add_node("query_synthesizer",  tool=MockQuerySynthesizer(),  depends_on=["schema_retriever"])
    graph.add_node("ai_analytics",       tool=MockAIAnalytics(),       depends_on=["query_synthesizer"])
    graph.add_node("chart_generator",    tool=MockChartGenerator(),    depends_on=["ai_analytics"])
    graph.add_node("insight_summarizer", tool=MockInsightSummarizer(), depends_on=["ai_analytics"])

    sea = SEA(graph=graph, planner=MockPlanner())

    queries = [
        ("s1", "Show me the top 3 products by revenue last quarter."),
        ("s1", "Show the same as a chart."),           # reuses ai_analytics state
        ("s1", "Filter to only Electronics."),         # re-runs from ai_analytics
        ("s1", "Switch to shipment delays analysis."), # topic switch → full purge
    ]

    for session_id, query in queries:
        print(f"\n{'='*60}")
        print(f"USER: {query}")
        result = sea.chat(query=query, session_id=session_id)
        print(f"TOOL CALLS: {result['tool_calls']}")
        print(f"SUMMARY DAG KEYS: {list(result['final_dag_state'].keys())}")
        print(f"STATUS: {result['execution_summary']}")


if __name__ == "__main__":
    main()
