# sea/tools/schema_retriever.py
"""
SchemaRetriever – Node 1 in the SEA analytics DAG (v0 / root).

Corresponds to SchemaRetriever in the paper (Section 3.1):
  "This tool initiates the workflow by mapping user intent to specific data
  tables.  It operates a two-stage process: an offline semantic indexing step
  … and an online retrieve-and-rerank mechanism."

Paper mapping: 1_data_discovery
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from sea.core.schemas import ToolResult
from sea.core.tool import Tool
from sea.tools._lib.chroma_store import TableReranker, VectorStore


class SchemaRetriever(Tool):
    """
    Retrieves the relevant table schemas for a user query via:
    1. Dense vector retrieval from a ChromaDB collection.
    2. LLM-based re-ranking to select the exact required tables.
    3. Loads full JSON summaries for the selected tables.

    Args:
        vector_store:    Initialised VectorStore instance.
        reranker:        Initialised TableReranker instance.
        json_summary_dir: Path to the directory containing per-table JSON
                          summary files (one file per table, named
                          `<table_name>.json`).
        k:               Number of candidates fetched from vector store
                         before re-ranking. Default 7 (paper default).
    """

    name = "schema_retriever"
    description = (
        "Finds the correct tables and their schemas for a user query. "
        "Use this for new topics or when the required tables change. "
        "No dependencies – always the first step."
    )

    def __init__(
        self,
        vector_store: VectorStore,
        reranker: TableReranker,
        json_summary_dir: str | Path,
        k: int = 7,
    ) -> None:
        self._vs = vector_store
        self._reranker = reranker
        self._summary_dir = Path(json_summary_dir)
        self._k = k

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
            upstream_artifacts: Empty dict – SchemaRetriever has no dependencies.
            **kwargs:           Optional override for `k` (number of candidates).
        """
        k = kwargs.get("k", self._k)

        # Step 1: dense retrieval
        candidates = self._vs.retrieve_top_k(query, k=k)

        # Step 2: LLM re-ranking
        selection = self._reranker.rerank(query, candidates)
        selected_tables: list = selection.get("selected_tables", [])
        reasoning: str = selection.get("reasoning", "")

        # Step 3: load full JSON summaries for selected tables
        table_summaries: Dict[str, Any] = {}
        for table in selected_tables:
            path = self._summary_dir / f"{table.lower()}.json"
            if path.exists():
                with path.open() as fh:
                    table_summaries[table] = json.load(fh)
            else:
                table_summaries[table] = None

        artifact = {
            "selected_tables": selected_tables,
            "reasoning": reasoning,
            "table_summaries": table_summaries,
        }
        summary = self.build_summary(artifact)
        return ToolResult(artifact=artifact, summary=summary)

    def build_summary(self, artifact: Dict[str, Any]) -> Dict[str, Any]:
        """Twin Summary: table names + reasoning (no raw data)."""
        return {
            "status": "SUCCESS",
            "selected_tables": artifact.get("selected_tables", []),
            "reasoning": artifact.get("reasoning", ""),
        }
