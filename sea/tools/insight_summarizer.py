# sea/tools/insight_summarizer.py
"""
InsightSummarizer – Node 5 in the SEA analytics DAG (parallel leaf).

Corresponds to InsightSummarizer in the paper (Section 3.1):
  "A data-to-text agent that ingests the final analyzed DataFrame and generates
  a concise, narrative summary of the key findings tailored to the user intent."

Paper mapping: 5_commentary
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from sea.core.schemas import ToolResult
from sea.core.tool import Tool

_COMMENTARY_TEMPLATE = """\
You are a skilled data analyst who excels at narrating insights.
Based on the user query and the provided DataFrame (as JSON), provide a
concise 2-3 line narrative of the key findings.
Do not suggest next steps or ask questions.

User Query: {query}
DataFrame JSON: {json}

Insights:"""


class InsightSummarizer(Tool):
    """
    Generates a natural-language narrative summary of the analytical results.

    Runs in parallel with ChartGenerator (both depend only on AIAnalytics).

    Args:
        llm: LangChain-compatible chat model.
    """

    name = "insight_summarizer"
    description = (
        "Generates a natural-language insight summary from the final DataFrame. "
        "Requires ai_analytics output. Can run in parallel with chart_generator."
    )

    def __init__(self, llm) -> None:
        prompt = PromptTemplate.from_template(_COMMENTARY_TEMPLATE)
        self._chain = prompt | llm | StrOutputParser()

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
            upstream_artifacts: Must contain 'ai_analytics' with 'result_df'.
        """
        ai_artifact = upstream_artifacts.get("ai_analytics", {})
        df_json: str = ai_artifact.get("result_df")

        if not df_json:
            raise ValueError(
                "InsightSummarizer requires 'result_df' from ai_analytics."
            )

        df = pd.read_json(df_json)
        commentary: str = self._chain.invoke({
            "query": query,
            "json": df.to_json(orient="records"),
        })

        artifact = {"query": query, "commentary": commentary}
        summary = self.build_summary(artifact)
        return ToolResult(artifact=artifact, summary=summary)

    def build_summary(self, artifact: Dict[str, Any]) -> Dict[str, Any]:
        commentary = artifact.get("commentary", "")
        snippet = commentary[:150] + "…" if len(commentary) > 150 else commentary
        return {"status": "SUCCESS", "commentary_snippet": snippet}
