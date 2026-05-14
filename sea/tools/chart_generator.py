# sea/tools/chart_generator.py
"""
ChartGenerator – Node 4 in the SEA analytics DAG (parallel leaf).

Corresponds to ChartGenerator in the paper (Section 3.1):
  "A code-generating agent that takes the final DataFrame's structured schema
  as input. It determines a suitable visualization type, then generates and
  executes Python code to render the chart."

Paper mapping: 4_visualization
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from sea.core.schemas import ToolResult
from sea.core.tool import Tool
from sea.tools._lib.chart_gen import ChartExecutor
from sea.tools._lib.chart_gen import ChartGenerator as _ChartGen
from sea.tools._lib.datamodel import Goal


class ChartGenerator(Tool):
    """
    Generates one or more charts from the AIAnalytics result DataFrame.

    Runs in parallel with InsightSummarizer (both depend only on AIAnalytics).

    Args:
        llm:     LangChain-compatible chat model.
        library: Plotting library ('matplotlib' or 'plotly'). Default: 'matplotlib'.
    """

    name = "chart_generator"
    description = (
        "Generates visualisations from the final analytical DataFrame. "
        "Requires ai_analytics output. Can run in parallel with insight_summarizer."
    )

    def __init__(self, llm, library: str = "matplotlib") -> None:
        self._generator = _ChartGen(llm=llm)
        self._executor = ChartExecutor()
        self._library = library

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
            query:              Enriched user query (used as chart title/question).
            upstream_artifacts: Must contain 'ai_analytics' with 'result_df'
                                and 'result_df_summary'.
            **kwargs:
                list_of_viz: List[str] – chart types to generate, e.g.
                             ['bar', 'pie'].  Defaults to ['bar'].
        """
        ai_artifact = upstream_artifacts.get("ai_analytics", {})
        df_json: str = ai_artifact.get("result_df")
        summary: dict = ai_artifact.get("result_df_summary", {})

        if not df_json:
            raise ValueError(
                "ChartGenerator requires 'result_df' from ai_analytics."
            )

        df = pd.read_json(df_json)
        viz_types: List[str] = kwargs.get("list_of_viz", ["bar"])

        charts = []
        for viz in viz_types:
            goal = Goal(question=query, visualization=viz, rationale="")
            code = self._generator.generate(summary=summary, goal=goal, library=self._library)
            resp = self._executor.execute(code=code, data=df, library=self._library, return_error=True)

            if resp is None or not resp.status:
                error = resp.error if resp else {"message": "Unknown error"}
                charts.append({"viz_type": viz, "status": "ERROR", "error": error})
            else:
                charts.append({
                    "viz_type": viz,
                    "status": "SUCCESS",
                    "raster": resp.raster,      # base64 PNG
                    "code": resp.code,
                })

        artifact = {"query": query, "charts": charts}
        summary_out = self.build_summary(artifact)
        return ToolResult(artifact=artifact, summary=summary_out)

    def build_summary(self, artifact: Dict[str, Any]) -> Dict[str, Any]:
        charts = artifact.get("charts", [])
        return {
            "status": "SUCCESS",
            "charts_generated": [c["viz_type"] for c in charts if c.get("status") == "SUCCESS"],
        }
