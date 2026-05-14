# SEA: Stateful Execution Environment for Conversational Big Data Analytics

<p align="center">
  <a href="https://neurips.cc/virtual/2025/loc/san-diego/124546"><img src="https://img.shields.io/badge/NeurIPS%202025-Workshop%20Paper-purple" /></a>
  <a href="https://huggingface.co/datasets/ajay-anil-kumar/Globo-Mart"><img src="https://img.shields.io/badge/🤗%20Dataset-GloboMart-yellow" /></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" />
  <img src="https://img.shields.io/badge/license-MIT-green" />
</p>

> **Paper**: *SEA: Stateful Execution Environment for Conversational Big Data Analytics*  
> Rohit Kumar · Ajay Anil Kumar — NeurIPS 2025 Workshop: Scaling Environments for Agents  
> [[Paper](https://neurips.cc/virtual/2025/loc/san-diego/124546)] · [[Dataset](https://huggingface.co/datasets/ajay-anil-kumar/Globo-Mart)] · [[Benchmark & Code](https://osf.io/buxma/files/osfstorage?view_only=f7de66430a7b42e0acbc6330ecedd255)]

---

## Overview

Existing LLM agents for data analytics operate **statelessly** — re-running expensive queries and re-fetching data on every conversational turn. SEA fixes this by encoding the analytical workflow as a **Directed Acyclic Graph (DAG)** and persisting tool outputs across turns.

The key insight: the planner's job is no longer open-ended workflow discovery. It becomes a tractable **node classification problem** — *"which node should I resume from to maximise state reuse?"*

<p align="center">
  <img width="700" src="https://raw.githubusercontent.com/langchain-ai/langgraph/main/docs/static/img/langgraph_logo.png" alt="placeholder — replace with Figure 1 from paper">
</p>

### Results on the GloboMart Benchmark (100 conversational queries)

| Metric | Score |
|---|---|
| Planner entry-point accuracy | **95.65%** |
| Path fidelity (1 − norm. Levenshtein) | **92.68%** |
| Schema retrieval accuracy | **84.06%** |
| End-to-end correctness (human eval) | **84.06%** |
| Avg. latency — fresh query | 53.4 s |
| Avg. latency — stateful follow-up | **34.1 s (↓ 36%)** |

---

## How it works

```
User query
    │
    ▼
┌─────────┐   Twin Summary DAG (S')   ┌─────────────────────────────────┐
│ Planner │ ◄────────────────────────► │  Dual-Representation State Model │
│(Gemini  │                            │                                   │
│  Pro)   │   Execution Plan (P_t)     │  Main State DAG (S)              │
└────┬────┘ ──────────────────────►   │  └─ full DataFrames, images, …   │
     │                                │                                   │
     ▼                                │  Twin Summary DAG (S')           │
┌──────────┐                          │  └─ SQL, schemas, code snippets  │
│ Executor │ ──── updates ──────────► └─────────────────────────────────┘
└──────────┘
     │
     ▼  (parallel where possible)
SchemaRetriever → QuerySynthesizer → AIAnalytics → ┬─ ChartGenerator
                                                    └─ InsightSummarizer
```

**Five tools** map to the two macro-stages of data analysis:

| Stage | Tool | Backed by |
|---|---|---|
| Data Subsetting | `SchemaRetriever` | ChromaDB + LLM re-ranker |
| Data Subsetting | `QuerySynthesizer` | Gemini Flash → Databricks SQL |
| Insight Generation | `AIAnalytics` | Gemini Flash → pandas code gen |
| Insight Generation | `ChartGenerator` | Gemini Flash → matplotlib |
| Insight Generation | `InsightSummarizer` | Gemini Flash → data-to-text |

---

## Installation

```bash
git clone https://github.com/your-org/SEA.git
cd SEA
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

---

## Quickstart

### 1 · Configure your `.env`

```bash
cp .env.example .env
# then fill in your values
```

```dotenv
# .env
GOOGLE_API_KEY=your_google_api_key

DATABRICKS_HOSTNAME=dbc-xxxxxxxx-xxxx.cloud.databricks.com
DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/your_warehouse_id
DATABRICKS_TOKEN=dapixxxxxxxxxxxxxxxxxxx
DATABRICKS_TABLE_PREFIX=workspace.globomart_db_final

VECTOR_DB_PATH=./data/vector_db_globomart
VECTOR_COLLECTION=globomart_sea
JSON_SUMMARY_DIR=./data/json_summaries
```

### 2 · Run the notebook

Open **`sea_quickstart.ipynb`** in Jupyter. It walks through a full 5-turn conversation showing fresh queries, stateful follow-ups, drill-downs, and topic switches with latency measurements at the end.

### 3 · Use the Python API directly

```python
import os
from dotenv import load_dotenv
from sea.workflows.analytics import build_analytics_sea

load_dotenv()

sea = build_analytics_sea(
    google_api_key          = os.environ["GOOGLE_API_KEY"],
    databricks_hostname     = os.environ["DATABRICKS_HOSTNAME"],
    databricks_http_path    = os.environ["DATABRICKS_HTTP_PATH"],
    databricks_token        = os.environ["DATABRICKS_TOKEN"],
    databricks_table_prefix = os.environ["DATABRICKS_TABLE_PREFIX"],
    vector_db_path          = os.environ["VECTOR_DB_PATH"],
    vector_collection       = os.environ["VECTOR_COLLECTION"],
    json_summary_dir        = os.environ["JSON_SUMMARY_DIR"],
)

# Turn 1 — fresh query, full pipeline (~53 s)
result = sea.chat("What were the top 5 products by revenue last quarter?", session_id="s1")

# Turn 2 — follow-up, enters at AIAnalytics, skips SQL (~20 s)
result = sea.chat("Break that down by product category.", session_id="s1")

# Turn 3 — topic switch → purge event → fresh pipeline
result = sea.chat("Now show me marketing spend by channel last year.", session_id="s1")
```

---

## Try it without any credentials

Run the mock demo — no API keys, no Databricks, no ChromaDB:

```bash
python examples/minimal_local.py
```

```
USER: Show me the top 3 products by revenue last quarter.
[MockPlanner] start_node='schema_retriever'
  [SchemaRetriever] → [QuerySynthesizer] → [AIAnalytics] → [ChartGenerator ‖ InsightSummarizer]
TOOL CALLS: ['schema_retriever', 'query_synthesizer', 'ai_analytics', 'insight_summarizer', 'chart_generator']

USER: Show the same as a chart.
[MockPlanner] start_node='chart_generator'   ← state reused
TOOL CALLS: ['chart_generator']
```

---

## Bring your own DAG

SEA is not limited to the GloboMart analytics workflow. Any DAG shape is supported:

```python
from sea import SEA, SEAGraph, Tool, ToolResult
from sea.core.planner import GeminiPlanner

class FetchTool(Tool):
    name = "fetch"
    description = "Fetches raw data. No dependencies."

    def run(self, query, upstream_artifacts, **kwargs):
        data = {"rows": [1, 2, 3]}
        return ToolResult(artifact=data, summary={"status": "SUCCESS", "row_count": 3})

class AnalyseTool(Tool):
    name = "analyse"
    description = "Analyses fetched data. Requires fetch output."

    def run(self, query, upstream_artifacts, **kwargs):
        rows = upstream_artifacts["fetch"]["rows"]
        result = {"total": sum(rows)}
        return ToolResult(artifact=result, summary={"status": "SUCCESS", "total": result["total"]})

graph = SEAGraph()
graph.add_node("fetch",   tool=FetchTool())
graph.add_node("analyse", tool=AnalyseTool(), depends_on=["fetch"])

sea = SEA(graph=graph, planner=GeminiPlanner(api_key="..."))
result = sea.chat("Analyse the data", session_id="demo")
```

---

## Repository structure

```
SEA/
├── sea/
│   ├── __init__.py            # SEA class (Algorithm 1 operational loop)
│   ├── core/
│   │   ├── tool.py            # Tool ABC → implement run() → ToolResult
│   │   ├── dag.py             # SEAGraph: add_node / add_edge builder
│   │   ├── state.py           # DualStateStore: Main DAG (S) + Twin Summary DAG (S')
│   │   ├── session.py         # SessionManager: history window + purge event
│   │   ├── planner.py         # Planner ABC + GeminiPlanner
│   │   ├── executor.py        # Deterministic executor with auto-parallel groups
│   │   └── schemas.py         # Plan, Step, ToolResult, SessionData
│   ├── tools/
│   │   ├── schema_retriever.py
│   │   ├── query_synthesizer.py
│   │   ├── ai_analytics.py
│   │   ├── chart_generator.py
│   │   ├── insight_summarizer.py
│   │   └── _lib/              # ChromaDB, Databricks, pandas/chart helpers
│   ├── workflows/
│   │   └── analytics.py       # build_analytics_graph() / build_analytics_sea()
│   └── api/
│       └── server.py          # Optional FastAPI /chat endpoint
├── data/
│   ├── vector_db_globomart/   # ChromaDB with GloboMart table embeddings
│   └── json_summaries/        # Per-table JSON schema summaries (8 tables)
├── examples/
│   └── minimal_local.py       # End-to-end demo, no credentials required
├── sea_quickstart.ipynb        # Full walkthrough notebook
├── .env.example                # Environment variable template
└── pyproject.toml
```

---

## The GloboMart Benchmark

A synthetic but realistic e-commerce data warehouse with **8 tables** (up to 6 M rows each) in a star schema, and **100 multi-turn conversational queries** covering:

- Single-table and multi-table JOIN queries
- Drill-downs, topic switches, and user corrections
- Diverse output types: KPI cards, data tables, charts, narrative summaries

| Table | Description |
|---|---|
| `Fact_Sales` | Line-item sales transactions |
| `Fact_Web_Analytics` | Session-level website events |
| `Dim_Customers` | Customer segments and geography |
| `Dim_Products` | Product category and brand |
| `Dim_Stores` | Store country and region |
| `Dim_Date` | Calendar dimension |
| `Dim_Marketing` | Campaign spend by channel |
| `Dim_Shipments` | Fulfilment and delivery data |

Dataset: [huggingface.co/datasets/ajay-anil-kumar/Globo-Mart](https://huggingface.co/datasets/ajay-anil-kumar/Globo-Mart)  
Benchmark & code: [osf.io/buxma](https://osf.io/buxma/files/osfstorage?view_only=f7de66430a7b42e0acbc6330ecedd255)

---

## Citation

```bibtex
@inproceedings{kumar2025sea,
  title     = {SEA: Stateful Execution Environment for Conversational Big Data Analytics},
  author    = {Kumar, Rohit and Kumar, Ajay Anil},
  booktitle = {NeurIPS 2025 Workshop: Scaling Environments for Agents},
  year      = {2025}
}
```
