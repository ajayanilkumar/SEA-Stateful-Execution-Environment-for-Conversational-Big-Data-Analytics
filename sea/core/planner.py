# sea/core/planner.py
"""
Planner Agent (P) – Section 3.3 of the paper.

The Planner's task is reframed from open-ended workflow discovery to a
strategic node-classification problem: given the current query and the
Twin Summary DAG S', select the optimal entry-point v_entry and generate
an ordered execution plan P_t.

Design
------
- `Planner`        – abstract base class; swap in any LLM or rules engine.
- `GeminiPlanner`  – concrete implementation using Gemini Pro (paper default).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from sea.core.schemas import Plan, SessionData

if TYPE_CHECKING:
    from sea.core.dag import SEAGraph


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_PLANNER_PROMPT = """\
# ROLE & GOAL
You are an expert AI orchestrator for a big data analytics platform.
Your goal is to create an efficient, multi-step execution plan.
Your most important task is to determine the correct entry-point (start_node)
to avoid re-doing unnecessary work while respecting all node dependencies.

# AVAILABLE NODES / TOOLS
{node_descriptions}

# NODE DEPENDENCIES (Required Inputs)
{dependency_description}

# CONTEXT
## Conversation History (raw_user_query → enriched_interpretation)
{conversation_history}

## Last Run's Twin Summary DAG (S') – lightweight symbolic state
{dag_state_summary}

# CURRENT TASK
## User's New Request
{user_query}

# INSTRUCTIONS
1. **Analyse Context**: Compare the new request with the conversation history and
   the Twin Summary DAG.
2. **Enrich Query**: Rewrite the user's request into a self-contained
   `enriched_query`.
3. **Determine Start Node**: Decide the first node that needs to be re-run.
   - CRITICAL: Verify that required inputs for the chosen `start_node` are
     present in the Twin Summary DAG.  If a required predecessor is missing,
     set `start_node` to that earliest missing dependency.
4. **Generate Plan**: List all steps from `start_node` through to the leaf
   nodes.
5. **Define Inputs**: For each step, encode any extra parameters as a JSON
   string in `inputs`.

# OUTPUT (must conform to the provided JSON schema)
"""


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class Planner(ABC):
    """
    Abstract Planner.  Implement `plan()` to integrate any LLM or
    deterministic strategy.
    """

    @abstractmethod
    def plan(self, query: str, session: SessionData, graph: "SEAGraph") -> Plan:
        """
        Produce an execution Plan for the current conversational turn.

        Args:
            query:   Raw user query q_t.
            session: Current SessionData (includes Twin Summary DAG S' and
                     conversation history M).
            graph:   The SEAGraph (for node descriptions and dependency info).

        Returns:
            A Plan with `start_node` and ordered `execution_plan`.
        """


# ---------------------------------------------------------------------------
# Gemini implementation (default – matches the paper)
# ---------------------------------------------------------------------------

class GeminiPlanner(Planner):
    """
    Planner backed by Google Gemini Pro (used in the paper).

    Uses structured JSON output via the Gemini `response_schema` feature so
    the Plan is always well-formed without regex post-processing.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.5-pro") -> None:
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError as exc:
            raise ImportError(
                "google-genai is required for GeminiPlanner. "
                "Install it with: pip install google-genai"
            ) from exc

        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._types = genai_types

    def plan(self, query: str, session: SessionData, graph: "SEAGraph") -> Plan:
        node_desc_lines = "\n".join(
            f"- `{nid}`: {desc}"
            for nid, desc in graph.node_descriptions.items()
        )
        prompt = _PLANNER_PROMPT.format(
            node_descriptions=node_desc_lines,
            dependency_description=graph.dependency_description(),
            conversation_history=json.dumps(
                [h.dict() for h in session.conversation_history], indent=2
            ),
            dag_state_summary=json.dumps(session.summary_dag_state, indent=2),
            user_query=query,
        )

        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=self._types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=Plan,
            ),
        )
        response_json = json.loads(response.text)
        return Plan(**response_json)
