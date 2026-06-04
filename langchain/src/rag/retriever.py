from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.observability import get_logger, metrics
from src.rag.embedder import embed_query
from src.security.prompt_guard import scan as prompt_guard_scan
from src.tools.supabase_mcp_client import supabase_mcp_tool

log = get_logger(__name__)


class RAGRetrieverArgs(BaseModel):
    user_id: str = Field(..., description="Tenant id — hard filter")
    query: str = Field(..., description="Natural language query")
    k: int = Field(8, ge=1, le=25)


def _k_bucket(k: int) -> str:
    if k <= 5:
        return "1-5"
    if k <= 10:
        return "6-10"
    return "11-25"


@tool("rag_retriever_tool", args_schema=RAGRetrieverArgs)
def rag_retriever_tool(user_id: str, query: str, k: int = 8) -> list[dict[str, Any]]:
    """Retrieve top-k experience chunks for this user."""
    embedding = embed_query(query)
    raw: list[dict[str, Any]] = supabase_mcp_tool.invoke(
        {
            "action": "vector_search",
            "user_id": user_id,
            "payload": {"embedding": embedding, "k": k},
        }
    )

    metrics.rag_retrievals_total.labels(k_bucket=_k_bucket(k)).inc()
    if raw:
        top1 = max((r.get("similarity", 0.0) for r in raw), default=0.0)
        metrics.rag_retrieval_score.observe(top1)

    safe: list[dict[str, Any]] = []
    dropped = 0
    for row in raw:
        chunk = row.get("chunk", "")
        block, reason = prompt_guard_scan(chunk, source="retrieve")
        if block:
            dropped += 1
            log.warning("rag.chunk_dropped", reason=reason)
            continue
        safe.append(row)

    if dropped:
        log.warning("rag.dropped_chunks", dropped=dropped, returned=len(safe))

    if raw and not safe:
        log.error("rag.empty_corpus_after_screening", original=len(raw))
        raise RuntimeError(
            "rag_retriever_tool: all retrieved chunks were blocked by Prompt "
            "Guard 2 — refusing to proceed with empty evidence"
        )
    return safe
