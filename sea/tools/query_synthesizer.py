# sea/tools/query_synthesizer.py
"""
QuerySynthesizer – Node 2 in the SEA analytics DAG.

Corresponds to QuerySynthesizer in the paper (Section 3.1):
  "Executes the Data Subsetting macro-stage. Receives the table schemas from
  the SchemaRetriever and the user's intent from the Planner. Generates and
  executes a complex SQL query … returning a single, unified pandas DataFrame."

Paper mapping: 2_sql_engine
"""

from __future__ import annotations

from typing import Any, Dict

from sea.core.schemas import ToolResult
from sea.core.tool import Tool
from sea.tools._lib.databricks import DatabricksSQLEngine


class QuerySynthesizer(Tool):
    """
    Generates a SQL query from the enriched user query + table schemas,
    executes it against the Databricks warehouse, and returns the raw
    DataFrame as a JSON string for downstream tools.

    Args:
        sql_engine: Initialised DatabricksSQLEngine (or any compatible
                    backend that exposes a `run(query, summaries)` method
                    returning {"sql_query": str, "sql_result": dict}).
    """

    name = "query_synthesizer"
    description = (
        "Generates and runs SQL to retrieve a broad data subset from the warehouse. "
        "Requires schema_retriever output. Use when the data to be analysed "
        "has changed (new topic, new filters)."
    )

    def __init__(self, sql_engine: DatabricksSQLEngine) -> None:
        self._engine = sql_engine

    # ------------------------------------------------------------------
    # Tool.run()
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        upstream_artifacts: Dict[str, Dict[str, Any]],
        **kwargs: Any,
    ) -> ToolResult:
        """
        Args:
            query:              Enriched user query.
            upstream_artifacts: Must contain 'schema_retriever' key with
                                {'table_summaries': dict} from SchemaRetriever.
        """
        schema_artifact = upstream_artifacts.get("schema_retriever", {})
        table_summaries = schema_artifact.get("table_summaries", {})
        if not table_summaries:
            raise ValueError(
                "QuerySynthesizer requires 'table_summaries' from schema_retriever. "
                "Ensure schema_retriever ran successfully in this session."
            )

        result = self._engine.run(query=query, table_summaries=table_summaries)

        artifact = result  # {"sql_query": str, "sql_result": {"status", "data", "columns", "row_count"}}
        summary = self.build_summary(artifact)
        return ToolResult(artifact=artifact, summary=summary)

    def build_summary(self, artifact: Dict[str, Any]) -> Dict[str, Any]:
        """Twin Summary: SQL query + row count (no raw data)."""
        sql_result = artifact.get("sql_result", {})
        return {
            "status": "SUCCESS" if sql_result.get("status") == "success" else "ERROR",
            "sql_query": artifact.get("sql_query"),
            "row_count": sql_result.get("row_count"),
            "columns": sql_result.get("columns", []),
        }
