from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field


ALLOWED_ACTIONS = {
    "upsert_profile",
    "embed_experiences",
    "fetch_profile",
    "vector_search",
}


class SupabaseMCPArgs(BaseModel):
    action: Literal[
        "upsert_profile",
        "embed_experiences",
        "fetch_profile",
        "vector_search",
    ] = Field(..., description="Allow-listed action")
    user_id: str = Field(..., description="UUID of the tenant — hard-filters every query")
    payload: dict[str, Any] = Field(default_factory=dict)


@lru_cache(maxsize=1)
def _sb():
    """Lazy Supabase client — built once, reused across requests."""
    from supabase import create_client  

    return create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_ROLE_KEY"],
    )

def _upsert_profile(user_id: str, payload: dict) -> dict:
    sb = _sb()

    sb.table("profiles").upsert({
        "user_id":        user_id,
        "full_name":      payload.get("full_name", ""),
        "email":          payload.get("email", ""),
        "phone":          payload.get("phone"),
        "location":       payload.get("location"),
        "hard_skills":    payload.get("hard_skills", []),
        "soft_skills":    payload.get("soft_skills", []),
        "certifications": payload.get("certifications", []),
        "leadership":     payload.get("leadership", []),
        "extracurricular_activities": payload.get("extracurricular_activities", []),
    }).execute()

    sb.table("experiences").delete().eq("user_id", user_id).execute()
    for exp in payload.get("experiences", []):
        sb.table("experiences").insert({
            "user_id":    user_id,
            "company":    exp.get("company", ""),
            "title":      exp.get("title", ""),
            "start_date": exp.get("start_date"),
            "end_date":   exp.get("end_date"),
            "bullets":    exp.get("bullets", []),
        }).execute()

    sb.table("education").delete().eq("user_id", user_id).execute()
    for edu in payload.get("education", []):
        sb.table("education").insert({
            "user_id":     user_id,
            "institution": edu.get("institution", ""),
            "degree":      edu.get("degree", ""),
            "field":       edu.get("field"),
            "start_date":  edu.get("start_date", ""),
            "end_date":    edu.get("end_date"),
        }).execute()

    return {"ok": True}


def _embed_experiences(user_id: str, payload: dict) -> dict:
    from src.rag.embedder import embed_documents 

    sb = _sb()

    bullets: list[str] = []
    for exp in payload.get("experiences", []):
        bullets.extend(b for b in exp.get("bullets", []) if b.strip())

    if not bullets:
        return {"ok": True, "embedded": 0}

    embeddings = embed_documents(bullets)

    sb.table("experience_embeddings").delete().eq("user_id", user_id).execute()
    for chunk, vec in zip(bullets, embeddings):
        sb.table("experience_embeddings").insert({
            "user_id":   user_id,
            "chunk":     chunk,
            "embedding": vec,   
        }).execute()

    return {"ok": True, "embedded": len(bullets)}


def _fetch_profile(user_id: str) -> dict | None:
    sb = _sb()

    r = sb.table("profiles").select("*").eq("user_id", user_id).execute()
    if not r.data:
        return None
    profile: dict = r.data[0]

    exps = sb.table("experiences").select("*").eq("user_id", user_id).execute()
    profile["experiences"] = [
        {
            "company":    e["company"],
            "title":      e["title"],
            "start_date": e.get("start_date", ""),
            "end_date":   e.get("end_date"),
            "bullets":    e.get("bullets", []),
        }
        for e in (exps.data or [])
    ]

    edus = sb.table("education").select("*").eq("user_id", user_id).execute()
    profile["education"] = [
        {
            "institution": e["institution"],
            "degree":      e["degree"],
            "field":       e.get("field"),
            "start_date":  e.get("start_date", ""),
            "end_date":    e.get("end_date"),
        }
        for e in (edus.data or [])
    ]

    return profile


def _vector_search(user_id: str, payload: dict) -> list[dict]:
    sb = _sb()
    embedding: list[float] = payload.get("embedding", [])
    k: int = payload.get("k", 8)
    r = sb.rpc("match_experiences", {
        "p_user_id": user_id,
        "p_query":   embedding,
        "p_k":       k,
    }).execute()
    return r.data or []

@tool("supabase_mcp_tool", args_schema=SupabaseMCPArgs)
def supabase_mcp_tool(
    action: str, user_id: str, payload: dict[str, Any]
) -> Any:
    """
    Gateway to Supabase. Always passes user_id so every query is
    tenant-scoped. Only allow-listed actions are accepted.
    """
    if action not in ALLOWED_ACTIONS:
        raise PermissionError(f"action '{action}' is not allow-listed")

    if action == "upsert_profile":
        return _upsert_profile(user_id, payload)
    if action == "embed_experiences":
        return _embed_experiences(user_id, payload)
    if action == "fetch_profile":
        return _fetch_profile(user_id)
    if action == "vector_search":
        return _vector_search(user_id, payload)
