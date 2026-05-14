# sea/tools/_lib/datamodel.py
"""
Data model types shared across the built-in analytics tools.
Migrated from ADAPT2/tools/playground/datamodel.py.
"""

from __future__ import annotations

import base64
from dataclasses import field
from typing import Any, Dict, List, Optional, Union

from pydantic.dataclasses import dataclass


@dataclass
class Goal:
    """A single analytical/visualisation goal derived from a user query."""
    question: str
    visualization: str
    rationale: str
    index: Optional[int] = 0


@dataclass
class Persona:
    """Persona used to steer goal generation."""
    persona: str
    rationale: str


@dataclass
class ChartExecutorResponse:
    """Return value from ChartExecutor.execute()."""
    spec: Optional[Union[str, Dict]]  # Plotly dict or None for matplotlib
    status: bool
    raster: Optional[str]             # base64-encoded PNG
    code: str
    library: str
    error: Optional[Dict] = None

    def savefig(self, path: str) -> None:
        """Save the raster image to disk."""
        if self.raster:
            with open(path, "wb") as fh:
                fh.write(base64.b64decode(self.raster))
        else:
            raise FileNotFoundError("No raster image to save.")

    def _repr_mimebundle_(self, include=None, exclude=None):  # pragma: no cover
        bundle = {"text/plain": self.code}
        if self.raster is not None:
            bundle["image/png"] = self.raster
        if self.spec is not None:
            bundle["application/vnd.vegalite.v5+json"] = self.spec
        return bundle
