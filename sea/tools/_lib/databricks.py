# sea/tools/_lib/databricks.py
"""
Databricks SQL and Unity Catalog helpers.
Migrated from ADAPT2/tools/data-access/src/databricks_fetcher.py
and ADAPT2/tools/playground/sqlengine.py.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from databricks import sql as dbsql
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connection config (populated from env or passed directly)
# ---------------------------------------------------------------------------

class DatabricksConfig(BaseModel):
    server_hostname: str
    http_path: str
    access_token: str
    catalog: Optional[str] = None
    schema_name: Optional[str] = None
    table_prefix: Optional[str] = None  # e.g. "workspace.globomart_db_final"


# ---------------------------------------------------------------------------
# Data fetcher (schema discovery)
# ---------------------------------------------------------------------------

class DatabricksDataFetcher:
    """Fetches table metadata from Databricks Unity Catalog."""

    def __init__(self, config: DatabricksConfig) -> None:
        import requests
        self._cfg = config
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {config.access_token}",
            "Content-Type": "application/json",
        })
        self._host = config.server_hostname.rstrip("/")

    def list_tables(self, catalog: str, schema: str) -> List[str]:
        endpoint = f"https://{self._host}/api/2.1/unity-catalog/tables"
        resp = self._session.get(endpoint, params={
            "catalog_name": catalog, "schema_name": schema, "max_results": 100
        })
        resp.raise_for_status()
        return [t.get("name") for t in resp.json().get("tables", [])]

    def sample_rows(self, catalog: str, schema: str, table: str, num_rows: int = 10) -> pd.DataFrame:
        query = f"SELECT * FROM {catalog}.{schema}.{table} TABLESAMPLE ({num_rows} ROWS)"
        return self._run_query(query)

    def _run_query(self, query: str) -> pd.DataFrame:
        with dbsql.connect(
            server_hostname=self._host,
            http_path=self._cfg.http_path,
            access_token=self._cfg.access_token,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description]
        return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# SQL engine (QuerySynthesizer backend)
# ---------------------------------------------------------------------------

class SQLResponse(BaseModel):
    reasoning: str = Field(description="Brief explanation of the generated SQL.")
    sql_query: str = Field(description="The SQL query to execute.")


_SQL_PROMPT = """\
You are an expert SQL data analyst.
Your task is to write a SQL query that retrieves a relevant *raw* slice of data
to help answer the user's question. A downstream Python/pandas step will do the
final calculations.

CRITICAL RULES:
- Do NOT use GROUP BY, ORDER BY, LIMIT, or aggregate functions (COUNT, SUM, …).
- Focus on an accurate WHERE clause.
- SELECT * to preserve all columns for downstream analysis.
- All table names MUST be lower-case.
- Prefix table names with `{table_prefix}` (e.g. `{table_prefix}.dim_customers`).
- Avoid duplicate column names if joins are needed; use aliases.

{format_instructions}

TABLE SCHEMA:
```json
{schema_str}
```

USER QUERY: {query}
"""


class DatabricksSQLEngine:
    """Generates and executes SQL queries against a Databricks warehouse."""

    def __init__(self, config: DatabricksConfig, llm) -> None:
        self._cfg = config
        self._llm = llm
        self._prefix = config.table_prefix or ""

    def run(self, query: str, table_summaries: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate SQL from `query` + `table_summaries`, execute it, and return
        a result dict compatible with the SEA Main State DAG format.
        """
        sql_response = self._generate_sql(query, table_summaries)
        result = self._execute_sql(sql_response.sql_query)
        return {
            "sql_query": sql_response.sql_query,
            "sql_result": result,
        }

    def _generate_sql(self, query: str, summaries: Dict[str, Any]) -> SQLResponse:
        from langchain.output_parsers import PydanticOutputParser
        from langchain_core.prompts import PromptTemplate

        parser = PydanticOutputParser(pydantic_object=SQLResponse)
        prompt = PromptTemplate(
            template=_SQL_PROMPT,
            input_variables=["schema_str", "query"],
            partial_variables={
                "format_instructions": parser.get_format_instructions(),
                "table_prefix": self._prefix,
            },
        )
        chain = prompt | self._llm | parser
        return chain.invoke({"schema_str": json.dumps(summaries, indent=2), "query": query})

    def _execute_sql(self, query: str) -> Dict[str, Any]:
        try:
            with dbsql.connect(
                server_hostname=self._cfg.server_hostname,
                http_path=self._cfg.http_path,
                access_token=self._cfg.access_token,
            ) as conn:
                with conn.cursor() as cur:
                    logger.info("Executing SQL: %s", query)
                    cur.execute(query)
                    rows = cur.fetchall()
                    cols = [d[0] for d in cur.description]
            df = pd.DataFrame(rows, columns=cols)
            return {
                "status": "success",
                "data": df.to_json(orient="records"),
                "columns": cols,
                "row_count": len(df),
            }
        except Exception as exc:
            logger.exception("SQL execution failed: %s", exc)
            return {"status": "error", "message": str(exc), "data": None}
