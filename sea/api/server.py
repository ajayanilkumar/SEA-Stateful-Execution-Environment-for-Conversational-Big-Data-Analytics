# sea/api/server.py
"""
Optional FastAPI server that wraps a configured SEA instance.

Usage
-----
    # In your entrypoint (e.g. main.py):
    import uvicorn
    from sea.api.server import create_app
    from sea.workflows.analytics import build_analytics_sea

    sea_instance = build_analytics_sea(
        google_api_key="...",
        databricks_hostname="...",
        ...
    )
    app = create_app(sea_instance)
    uvicorn.run(app, host="0.0.0.0", port=8080)
"""

from __future__ import annotations

import traceback
import uuid
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from sea.core.schemas import ChatRequest, ChatResponse

if TYPE_CHECKING:
    from sea import SEA


def create_app(sea_instance: "SEA") -> FastAPI:
    """
    Create a FastAPI application backed by the given SEA instance.

    The single `/chat` endpoint accepts a query + session_id and returns the
    Twin Summary DAG state and execution summary for the turn.
    """
    app = FastAPI(
        title="SEA – Stateful Execution Environment",
        description="Conversational Big Data Analytics API (NeurIPS 2025 SEA paper).",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {"status": "ok", "graph_nodes": list(sea_instance.graph.nodes.keys())}

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest):
        session_id = request.session_id or str(uuid.uuid4())
        try:
            result = sea_instance.chat(query=request.query, session_id=session_id)
            return ChatResponse(
                session_id=session_id,
                tool_calls=result["tool_calls"],
                response={
                    "final_dag_state": result["final_dag_state"],
                    "execution_summary": result["execution_summary"],
                },
            )
        except Exception:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail="Internal orchestrator error.")

    return app
