# sea/tools/_lib/df_summarizer.py
"""
DataFrame summariser used by multiple analytics tools.

Produces a structured JSON summary of a DataFrame's schema and sample values,
with optional LLM enrichment for semantic descriptions.

Consolidated from:
  - ADAPT2/tools/data-access/src/summarizer.py
  - ADAPT2/tools/playground/utils.py (Summarizer class)
"""

from __future__ import annotations

import datetime
import json
import logging
import re
import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JSON serialisation helpers
# ---------------------------------------------------------------------------

def _serialize_value(value: Any) -> Any:
    """Convert numpy / pandas types to JSON-safe Python scalars."""
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (datetime.date, pd.Timestamp)):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types in nested dicts/lists."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(el) for el in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# Core summariser
# ---------------------------------------------------------------------------

_ENRICH_SYSTEM_PROMPT = """\
You are an expert Business Intelligence (BI) analyst.
Annotate the JSON summary below by:
1. Adding a concise `dataset_description` for the table.
2. For each field, filling in `description` (business meaning) and
   `semantic_type` (one word, e.g. product_id, revenue, timestamp).
Return only the updated JSON object. No extra text."""


class DataFrameSummarizer:
    """
    Generates a structured summary of a pandas DataFrame.

    Args:
        llm: Optional LangChain-compatible chat model. When provided,
             `summarize(use_llm=True)` enriches the summary with semantic
             descriptions. When None, a plain schema summary is returned.
    """

    def __init__(self, llm=None) -> None:
        self._llm = llm

    # ------------------------------------------------------------------
    # Column property extraction
    # ------------------------------------------------------------------

    def _column_properties(self, df: pd.DataFrame, n_samples: int = 3) -> List[Dict]:
        props_list = []
        for col in df.columns:
            dtype = df[col].dtype
            props: Dict[str, Any] = {}

            if dtype == bool or pd.api.types.is_bool_dtype(dtype):
                props["dtype"] = "boolean"
            elif pd.api.types.is_numeric_dtype(dtype):
                props["dtype"] = "number"
                props["std"] = _serialize_value(df[col].std())
                props["min"] = _serialize_value(df[col].min())
                props["max"] = _serialize_value(df[col].max())
            elif dtype == object:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pd.to_datetime(df[col], errors="raise")
                    props["dtype"] = "date"
                except (ValueError, TypeError):
                    ratio = df[col].nunique() / max(len(df[col]), 1)
                    props["dtype"] = "category" if ratio < 0.5 else "string"
            elif pd.api.types.is_categorical_dtype(df[col]):
                props["dtype"] = "category"
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                props["dtype"] = "date"
            else:
                props["dtype"] = str(dtype)

            if props["dtype"] == "date":
                try:
                    props["min"] = _serialize_value(df[col].min())
                    props["max"] = _serialize_value(df[col].max())
                except Exception:
                    pass

            non_null = df[col].dropna().unique()
            sample_count = min(n_samples, len(non_null))
            samples = (
                [_serialize_value(v) for v in pd.Series(non_null).sample(sample_count, random_state=42).tolist()]
                if sample_count > 0 else []
            )
            props["samples"] = samples
            props["num_unique_values"] = int(df[col].nunique())
            props["semantic_type"] = ""
            props["description"] = ""
            props_list.append({"column": col, "properties": props})

        return props_list

    # ------------------------------------------------------------------
    # LLM enrichment
    # ------------------------------------------------------------------

    def _enrich(self, base_summary: dict) -> dict:
        if self._llm is None:
            raise RuntimeError("No LLM provided; cannot enrich summary.")
        from langchain_core.messages import HumanMessage, SystemMessage

        def _default(obj):
            if isinstance(obj, (pd.Timestamp, datetime.datetime, datetime.date)):
                return obj.isoformat()
            raise TypeError(f"Not serialisable: {type(obj)}")

        messages = [
            SystemMessage(content=_ENRICH_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(base_summary, default=_default)),
        ]
        response = self._llm.invoke(messages)
        content = response.content.strip()
        # Strip markdown fences if present
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        return json.loads(content)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summarize(
        self,
        df: pd.DataFrame,
        table_name: str = "dataframe",
        n_samples: int = 3,
        use_llm: bool = False,
    ) -> dict:
        """
        Produce a structured summary dict for a DataFrame.

        Args:
            df:         The DataFrame to summarise.
            table_name: Logical name (used as `name` and `file_name` in output).
            n_samples:  Number of sample values per column.
            use_llm:    If True and an LLM is configured, enrich with semantic
                        descriptions. Requires `llm` to be set on the instance.

        Returns:
            A dict with keys: name, file_name, dataset_description, fields,
            field_names.
        """
        props = self._column_properties(df, n_samples)
        base = {
            "name": table_name,
            "file_name": table_name,
            "dataset_description": "",
            "fields": props,
            "field_names": list(df.columns),
        }
        if use_llm:
            enriched = self._enrich(base)
            enriched.setdefault("field_names", list(df.columns))
            return enriched
        return base
