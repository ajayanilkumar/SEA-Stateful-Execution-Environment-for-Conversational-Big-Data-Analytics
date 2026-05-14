# sea/core/schemas.py
"""
Pydantic schemas for the SEA framework – plans, steps, session data.
These match the data model described in the paper (Algorithm 1).
"""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# LLM Planner output
# ---------------------------------------------------------------------------

class Step(BaseModel):
    """One tool call inside an execution plan."""
    node_id: str = Field(..., description="Node identifier in the SEAGraph.")
    inputs: str = Field(
        default="{}",
        description="JSON-encoded dict of extra inputs the planner wants to pass to this tool.",
    )


class Plan(BaseModel):
    """Structured plan produced by the Planner Agent (P) at each turn t."""
    reasoning: str = Field(..., description="Planner's chain-of-thought explanation.")
    enriched_query: str = Field(..., description="Self-contained, context-enriched rewrite of the user query.")
    start_node: str = Field(..., description="Entry-point node id (v_entry) that the executor starts from.")
    execution_plan: List[Step] = Field(..., description="Ordered list of tool calls from start_node to leaves.")


# ---------------------------------------------------------------------------
# Session / conversation history
# ---------------------------------------------------------------------------

class HistoryItem(BaseModel):
    """One turn in the Planner's session memory M."""
    raw: str
    enriched: str


class SessionData(BaseModel):
    """
    Implements the Dual-Representation State Model (Section 3.2).

    main_dag_state  – The Main State DAG (S):
                      maps node_id -> full computational artifact (heavy).
    summary_dag_state – The Twin Summary DAG (S'):
                      maps node_id -> lightweight symbolic metadata for the Planner.
    conversation_history – The Planner's memory M (sliding window of raw/enriched queries).
    """
    main_dag_state: Dict[str, Any] = {}
    summary_dag_state: Dict[str, Any] = {}
    conversation_history: List[HistoryItem] = []


# ---------------------------------------------------------------------------
# Tool result
# ---------------------------------------------------------------------------

class ToolResult(BaseModel):
    """
    Return value of Tool.run().

    artifact – stored in Main State DAG (may be large; not sent to LLM).
    summary  – stored in Twin Summary DAG (compact; used in planner prompt).
    """
    artifact: Dict[str, Any]
    summary: Dict[str, Any]


# ---------------------------------------------------------------------------
# API layer (optional FastAPI server)
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str
    query: str


class ChatResponse(BaseModel):
    session_id: str
    tool_calls: List[str]
    response: Dict[str, Any]
