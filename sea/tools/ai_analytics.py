# sea/tools/ai_analytics.py
"""
AIAnalytics – Node 3 in the SEA analytics DAG.

Corresponds to AIAnalytics in the paper (Section 3.1):
  "Begins the Localized Insight Generation macro-stage. Operates on the
  manageable DataFrame subset … generating and executing pandas code to
  produce the final, user-facing analytical DataFrame."

Paper mapping: 3_ai_playground
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from sea.core.schemas import ToolResult
from sea.core.tool import Tool
from sea.tools._lib.datamodel import Goal
from sea.tools._lib.df_summarizer import DataFrameSummarizer, convert_numpy_types
from sea.tools._lib.pandas_gen import PandasExecutor, PandasGenerator


class AIAnalytics(Tool):
    """
    Performs localised analysis on the DataFrame produced by QuerySynthesizer.

    Uses an LLM to generate pandas transformation code, executes it safely,
    and returns the final analytical DataFrame plus its schema summary.

    Args:
        llm:       LangChain-compatible chat model (Gemini Flash recommended
                   for low-latency code generation).
        summarizer: DataFrameSummarizer instance (optional; one is created
                    internally if not provided).
    """

    name = "ai_analytics"
    description = (
        "Performs localised data analysis on the SQL subset using Python/pandas. "
        "Requires query_synthesizer output. Use for follow-up analysis, new "
        "calculations, or reformatting that does not require re-fetching data."
    )

    def __init__(self, llm, summarizer: DataFrameSummarizer | None = None) -> None:
        self._llm = llm
        self._summarizer = summarizer or DataFrameSummarizer(llm=llm)
        self._generator = PandasGenerator(llm=llm)
        self._executor = PandasExecutor()

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
            upstream_artifacts: Must contain 'query_synthesizer' key with
                                {'sql_result': {'data': <json_str>}}.
        """
        sql_artifact = upstream_artifacts.get("query_synthesizer", {})
        sql_result = sql_artifact.get("sql_result", {})
        data_json = sql_result.get("data")
        if not data_json:
            raise ValueError(
                "AIAnalytics requires 'sql_result.data' from query_synthesizer. "
                "Ensure query_synthesizer completed successfully."
            )

        df = pd.read_json(data_json)

        # Build a base summary for code generation context
        base_summary = self._summarizer.summarize(df)

        # Generate and execute transformation code
        goal = Goal(question=query, visualization=query, rationale="")
        code = self._generator.generate(summary=base_summary, goal=goal)
        result_df = self._executor.execute(code=code, data=df)

        if result_df is None:
            raise RuntimeError("AIAnalytics: pandas transformation returned None.")

        # Clean non-JSON-serialisable values
        result_df.replace([np.inf, -np.inf], None, inplace=True)
        result_df = result_df.where(pd.notna(result_df), None)

        # Summarise the result DataFrame for ChartGenerator / InsightSummarizer
        result_summary = self._summarizer.summarize(result_df)
        result_summary = convert_numpy_types(result_summary)
        result_df_json = result_df.to_json(date_format="iso", orient="records")

        artifact = {
            "result_df": result_df_json,
            "result_df_summary": result_summary,
            "code_generated": code,
        }
        summary = self.build_summary(artifact)
        return ToolResult(artifact=artifact, summary=summary)

    def build_summary(self, artifact: Dict[str, Any]) -> Dict[str, Any]:
        """Twin Summary: generated code + result schema (no raw data)."""
        return {
            "status": "SUCCESS",
            "analysis_code": artifact.get("code_generated"),
            "result_summary": artifact.get("result_df_summary"),
        }
