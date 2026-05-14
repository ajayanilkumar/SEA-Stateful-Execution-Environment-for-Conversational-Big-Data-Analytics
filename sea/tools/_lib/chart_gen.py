# sea/tools/_lib/chart_gen.py
"""
Chart code generation and execution helpers (ChartGenerator tool).
Migrated from ADAPT2/tools/playground/utils.py.
"""

from __future__ import annotations

import base64
import io
import logging
import traceback
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from sea.tools._lib.datamodel import ChartExecutorResponse, Goal
from sea.tools._lib.pandas_gen import clean_code_snippet, get_exec_globals

logger = logging.getLogger(__name__)

_VIZ_SYSTEM_PROMPT = """\
You are an assistant highly skilled in writing PERFECT visualisation code.
The data is ALREADY TRANSFORMED and ready for plotting.
Rules:
- Match the user's goal and use visualisation best practices.
- All labels, titles, and legends must be WHITE (dark background).
- Ensure axis labels are legible (rotate if necessary).
- Return a FULL PYTHON PROGRAM in a single markdown block.
"""


class ChartGenerator:
    """Generates matplotlib visualisation code via an LLM."""

    def __init__(self, llm) -> None:
        self._llm = llm

    def _template(self, goal: Goal, library: str) -> Tuple[str, str]:
        if library in ("matplotlib", "seaborn"):
            instructions = "The plot function must return a matplotlib plt object."
            template = (
                "import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n"
                "from sea.tools._lib.colours import create_colored_plot\n\n"
                "def plot(data: pd.DataFrame):\n"
                "    fig, ax, colors = create_colored_plot()\n"
                "    # TODO – draw the chart here\n"
                f"    plt.title('{goal.question}', wrap=True)\n"
                "    return plt\n\n"
                "chart = plot(data)\n"
            )
        elif library == "plotly":
            instructions = "The plot function must return a plotly Figure."
            template = (
                "import pandas as pd\nimport plotly.express as px\n"
                "from sea.tools._lib.colours import create_colored_plotly\n\n"
                "def plot(data: pd.DataFrame):\n"
                "    fig, colors = create_colored_plotly()\n"
                "    # TODO\n"
                "    return fig\n\n"
                "chart = plot(data)\n"
            )
        else:
            raise ValueError(f"Unsupported library: {library}")
        return template, instructions

    def generate(self, summary: dict, goal: Goal, library: str = "matplotlib") -> str:
        import json
        template, instructions = self._template(goal, library)
        user_prompt = (
            f"Data Summary:\n{json.dumps(summary, indent=2)}\n\n"
            f"Goal: create a `{goal.visualization}` for '{goal.question}'.\n"
            f"`data` is a pandas DataFrame ready for plotting.\n"
            f"Complete the {library} template below. {instructions}"
        )
        messages = [
            {"role": "system", "content": _VIZ_SYSTEM_PROMPT},
            {"role": "user", "content": f"{user_prompt}\n\nTemplate:\n```python\n{template}\n```"},
        ]
        response = self._llm.invoke(messages).content
        return clean_code_snippet(response)


class ChartExecutor:
    """Executes chart code and returns a ChartExecutorResponse."""

    def execute(
        self,
        code: str,
        data: pd.DataFrame,
        library: str = "matplotlib",
        return_error: bool = False,
    ) -> Optional[ChartExecutorResponse]:
        try:
            ex_locals = get_exec_globals(code, data)

            if library in ("matplotlib", "seaborn"):
                plt.figure(figsize=(12, 12))
                exec(code, ex_locals)  # noqa: S102
                plt.box(False)
                plt.grid(color="lightgray", linestyle="dashed", zorder=-10)
                buf = io.BytesIO()
                plt.savefig(buf, format="png", dpi=100, pad_inches=0.2)
                buf.seek(0)
                raster = base64.b64encode(buf.read()).decode("ascii")
                plt.close()
                return ChartExecutorResponse(
                    spec=None, status=True, raster=raster, code=code, library=library
                )

            elif library == "plotly":
                import plotly.io as pio
                exec(code, ex_locals)  # noqa: S102
                chart = ex_locals["chart"]
                raster = base64.b64encode(
                    pio.to_image(chart, format="png", scale=3)
                ).decode("utf-8")
                return ChartExecutorResponse(
                    spec=chart.to_dict(), status=True, raster=raster,
                    code=code, library=library,
                )

        except Exception as exc:
            logger.error("ChartExecutor error: %s\n%s", exc, traceback.format_exc())
            plt.close()
            if return_error:
                return ChartExecutorResponse(
                    spec=None, status=False, raster=None, code=code,
                    library=library,
                    error={"message": str(exc), "traceback": traceback.format_exc()},
                )
        return None
