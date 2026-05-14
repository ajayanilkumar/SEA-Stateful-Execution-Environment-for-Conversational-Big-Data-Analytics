# sea/tools/_lib/chroma_store.py
"""
ChromaDB vector store helpers for SchemaRetriever.
Migrated from ADAPT2/tools/data-access/src/vector_store.py and retriever.py.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import List

import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_RERANK_PROMPT = """\
You are an expert E-commerce BI Analyst specialising in SQL and star schema data warehouses.
Identify the tables needed to answer the user's analytical query from the candidates below.

## Star Schema Thinking
1. Deconstruct the query: find metrics (Fact tables) and dimensions (Dimension tables).
2. Identify the Fact table containing the requested metrics.
3. Identify Dimension tables for filtering/grouping.
4. Confirm the join keys exist.

## User Query
{query}

## Candidate Schemas
{schemas}

{format_instructions}
"""


class VectorStore:
    """
    Wraps a ChromaDB persistent collection for table-schema embeddings.
    Used by SchemaRetriever for dense retrieval.
    """

    def __init__(self, db_path: str = "./vector_db", collection_name: str = "schema_embeddings") -> None:
        self._client = chromadb.PersistentClient(path=db_path)
        self._collection = self._client.get_or_create_collection(collection_name)
        self._embedder = SentenceTransformer(_EMBEDDING_MODEL)

    def _embed(self, text: str):
        return self._embedder.encode(text, convert_to_numpy=True)

    def store(self, json_summary: dict, enriched_text: str) -> None:
        """Store a table summary with its enriched-text embedding."""
        table_name = json_summary.get("name") or str(uuid.uuid4())
        embedding = self._embed(enriched_text)
        self._collection.add(
            ids=[table_name],
            embeddings=[embedding.tolist()],
            metadatas=[{
                "enriched_text": enriched_text,
                "json_summary": json.dumps(json_summary),
            }],
        )
        logger.info("Stored table '%s' in vector store.", table_name)

    def retrieve_top_k(self, query: str, k: int = 7) -> list:
        """Dense retrieval: return top-k metadata dicts."""
        embedding = self._embed(query)
        results = self._collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=k,
        )
        metadatas = results.get("metadatas", [[]])[0]
        return metadatas


class TableReranker:
    """
    LLM-based re-ranker that selects the exact set of tables needed to answer
    a query from the top-k dense-retrieval candidates.

    Uses a LangChain-compatible chat model + PydanticOutputParser.
    """

    def __init__(self, llm) -> None:
        self._llm = llm

    def rerank(self, query: str, candidates: list) -> dict:
        """
        Args:
            query:      User query.
            candidates: List of metadata dicts from VectorStore.retrieve_top_k().

        Returns:
            dict with keys:
                selected_tables: List[str]
                reasoning:       str
        """
        from langchain.output_parsers import PydanticOutputParser
        from langchain_core.prompts import PromptTemplate
        from pydantic import BaseModel, Field

        class _TableSelection(BaseModel):
            reasoning: str = Field(description="Short explanation for the table selection.")
            selected_tables: List[str] = Field(description="List of table names required.")

        parser = PydanticOutputParser(pydantic_object=_TableSelection)

        snippets = []
        for i, meta in enumerate(candidates):
            js = meta.get("json_summary", "").strip()
            if js:
                try:
                    obj = json.loads(js)
                    snippets.append(f"Candidate #{i+1} – {obj.get('name','?')}\n{json.dumps(obj, indent=2)}")
                except Exception:
                    snippets.append(f"Candidate #{i+1} – {meta.get('enriched_text', '')}")
            else:
                snippets.append(f"Candidate #{i+1} – {meta.get('enriched_text', '')}")

        prompt = PromptTemplate(
            template=_RERANK_PROMPT,
            input_variables=["query", "schemas"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt | self._llm | parser
        selection = chain.invoke({"query": query, "schemas": "\n\n".join(snippets)})
        return selection.model_dump()
