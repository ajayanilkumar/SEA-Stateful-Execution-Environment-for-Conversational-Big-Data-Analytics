# sea/tools/_lib/pandas_gen.py
"""
Pandas code generation and execution helpers (AIAnalytics tool).
Migrated from ADAPT2/tools/playground/utils.py.
"""

from __future__ import annotations

import ast
import importlib
import logging
import re
import traceback
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from sea.tools._lib.datamodel import Goal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Code helpers
# ---------------------------------------------------------------------------

def clean_code_snippet(code: str) -> str:
    """Strip markdown code fences from an LLM response."""
    match = re.search(r"```(?:python)?\s*([\s\S]*?)\s*```", code)
    return match.group(1).strip() if match else code.strip()


def get_exec_globals(code: str, data: pd.DataFrame) -> dict:
    """Parse import statements in generated code and build exec() globals."""
    tree = ast.parse(code)
    imports: Dict[str, Any] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = importlib.import_module(alias.name)
                imports[alias.asname or alias.name] = mod
        elif isinstance(node, ast.ImportFrom) and node.module:
            mod = importlib.import_module(node.module)
            for alias in node.names:
                imports[alias.asname or alias.name] = getattr(mod, alias.name)
    imports.update({"pd": pd, "np": np, "data": data})
    return imports


# ---------------------------------------------------------------------------
# Pandas code generation
# ---------------------------------------------------------------------------

_PANDAS_SYSTEM_PROMPT = """\
You are an expert Python programmer specialising in pandas.
Write a single function `transform_data(data: pd.DataFrame) -> pd.DataFrame`.
Rules:
- The function MUST be named `transform_data` and accept a DataFrame as its only argument.
- The function MUST return the transformed DataFrame.
- Do NOT include plotting code.
- Handle NaN and Inf values (they are not JSON serialisable).
- Handle datetime columns with pd.to_datetime(..., errors='coerce').
- Return ONLY the complete Python code in one block."""


class PandasGenerator:
    """Generates pandas transformation code via an LLM for a given goal."""

    def __init__(self, llm) -> None:
        self._llm = llm

    def generate(self, summary: dict, goal: Goal) -> str:
        import json
        template = (
            "import pandas as pd\nimport numpy as np\n\n"
            "# plan:\n# 1.\n# 2.\n\n"
            "def transform_data(data: pd.DataFrame) -> pd.DataFrame:\n"
            "    transformed_df = data.copy()\n"
            "    # TODO\n"
            "    return transformed_df\n"
        )
        user_prompt = (
            f"Dataset Summary:\n{json.dumps(summary, indent=2)}\n\n"
            f"Goal: {goal.question}\n\n"
            f"Complete the template:\n```python\n{template}\n```"
        )
        messages = [
            {"role": "system", "content": _PANDAS_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        response = self._llm.invoke(messages).content
        return clean_code_snippet(response)


class PandasExecutor:
    """Executes pandas transformation code produced by PandasGenerator."""

    def execute(
        self, code: str, data: pd.DataFrame, return_error: bool = False
    ) -> Optional[pd.DataFrame]:
        try:
            ex_locals = get_exec_globals(code, data)
            exec(code, ex_locals)  # noqa: S102
            result = ex_locals["transform_data"](data)
            return result
        except Exception as exc:
            logger.error("PandasExecutor error: %s\n%s", exc, traceback.format_exc())
            if return_error:
                return {"error": str(exc), "traceback": traceback.format_exc()}
            return None
