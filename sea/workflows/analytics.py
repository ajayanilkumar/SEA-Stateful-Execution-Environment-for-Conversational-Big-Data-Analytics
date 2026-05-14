# sea/workflows/analytics.py
"""
GloboMart Analytics Workflow – the 5-node DAG from the SEA paper.

This module provides `build_analytics_graph()`, a factory that wires up the
exact Planner–Executor graph described in Section 3.2:

    SchemaRetriever → QuerySynthesizer → AIAnalytics → ┬─ ChartGenerator
                                                        └─ InsightSummarizer

Usage
-----
    import os
    from sea.workflows.analytics import build_analytics_graph, build_analytics_sea

    # Option A: get just the SEAGraph (bring your own SEA instance)
    graph = build_analytics_graph(
        google_api_key=os.environ["GOOGLE_API_KEY"],
        databricks_hostname="dbc-xxx.cloud.databricks.com",
        databricks_http_path="/sql/1.0/warehouses/...",
        databricks_token=os.environ["DATABRICKS_TOKEN"],
        databricks_table_prefix="workspace.globomart_db_final",
        vector_db_path="./vector_db_globomart",
        vector_collection="globomart_sea",
        json_summary_dir="./json_summaries",
    )

    # Option B: get a fully configured SEA instance ready to chat
    sea = build_analytics_sea(google_api_key=..., ...)
    result = sea.chat("What were the top 5 products last quarter?", session_id="s1")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from sea.core.dag import SEAGraph
from sea.core.planner import GeminiPlanner
from sea.core.session import SessionManager
from sea.tools._lib.chroma_store import TableReranker, VectorStore
from sea.tools._lib.databricks import DatabricksConfig, DatabricksSQLEngine
from sea.tools.ai_analytics import AIAnalytics
from sea.tools.chart_generator import ChartGenerator
from sea.tools.insight_summarizer import InsightSummarizer
from sea.tools.query_synthesizer import QuerySynthesizer
from sea.tools.schema_retriever import SchemaRetriever


def _make_gemini_flash_llm(api_key: str):
    """Gemini Flash – used for latency-sensitive tool code-gen (paper default)."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0,
        max_retries=2,
    )


def build_analytics_graph(
    *,
    google_api_key: str,
    databricks_hostname: str,
    databricks_http_path: str,
    databricks_token: str,
    databricks_table_prefix: str = "workspace.globomart_db_final",
    vector_db_path: str = "./vector_db_globomart",
    vector_collection: str = "globomart_sea",
    json_summary_dir: str = "./json_summaries",
    chart_library: str = "matplotlib",
    retriever_k: int = 7,
) -> SEAGraph:
    """
    Build the 5-node SEAGraph for the GloboMart analytics workflow.

    Args:
        google_api_key:          Google API key for Gemini Flash (tool LLM).
        databricks_hostname:     Databricks server hostname (no https://).
        databricks_http_path:    SQL warehouse HTTP path.
        databricks_token:        Databricks personal access token.
        databricks_table_prefix: Fully-qualified schema prefix for SQL queries.
        vector_db_path:          Path to the ChromaDB persistent store.
        vector_collection:       ChromaDB collection name.
        json_summary_dir:        Directory containing per-table JSON summaries.
        chart_library:           Plotting library ('matplotlib' or 'plotly').
        retriever_k:             Top-k candidates fetched during dense retrieval.

    Returns:
        A configured SEAGraph ready to be passed to SEA.
    """
    flash_llm = _make_gemini_flash_llm(google_api_key)

    # --- Tool 1: SchemaRetriever (root / v0) ---
    vs = VectorStore(db_path=vector_db_path, collection_name=vector_collection)
    reranker = TableReranker(llm=flash_llm)
    schema_retriever = SchemaRetriever(
        vector_store=vs,
        reranker=reranker,
        json_summary_dir=json_summary_dir,
        k=retriever_k,
    )

    # --- Tool 2: QuerySynthesizer ---
    db_config = DatabricksConfig(
        server_hostname=databricks_hostname,
        http_path=databricks_http_path,
        access_token=databricks_token,
        table_prefix=databricks_table_prefix,
    )
    sql_engine = DatabricksSQLEngine(config=db_config, llm=flash_llm)
    query_synthesizer = QuerySynthesizer(sql_engine=sql_engine)

    # --- Tool 3: AIAnalytics ---
    ai_analytics = AIAnalytics(llm=flash_llm)

    # --- Tools 4 & 5: parallel leaves ---
    chart_generator = ChartGenerator(llm=flash_llm, library=chart_library)
    insight_summarizer = InsightSummarizer(llm=flash_llm)

    # --- Wire the DAG ---
    graph = SEAGraph()
    graph.add_node("schema_retriever",  tool=schema_retriever)
    graph.add_node("query_synthesizer", tool=query_synthesizer,  depends_on=["schema_retriever"])
    graph.add_node("ai_analytics",      tool=ai_analytics,       depends_on=["query_synthesizer"])
    graph.add_node("chart_generator",   tool=chart_generator,    depends_on=["ai_analytics"])
    graph.add_node("insight_summarizer",tool=insight_summarizer, depends_on=["ai_analytics"])

    return graph


def build_analytics_sea(
    *,
    google_api_key: str,
    planner_model: str = "gemini-2.5-pro",
    session_manager: Optional[SessionManager] = None,
    **graph_kwargs,
):
    """
    Convenience factory: returns a fully wired SEA instance.

    Args:
        google_api_key: Google API key (used for both Planner and tool LLM).
        planner_model:  Gemini model for the Planner (default: gemini-2.5-pro).
        session_manager: Optional custom SessionManager.
        **graph_kwargs: Forwarded to build_analytics_graph().

    Returns:
        A SEA instance with `.chat(query, session_id)` ready to use.
    """
    from sea import SEA  # avoid circular import at module load time

    graph = build_analytics_graph(google_api_key=google_api_key, **graph_kwargs)
    planner = GeminiPlanner(api_key=google_api_key, model=planner_model)
    return SEA(graph=graph, planner=planner, session_manager=session_manager)
