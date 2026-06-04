from __future__ import annotations

import os
from functools import lru_cache

import httpx


@lru_cache(maxsize=1)
def _cfg() -> tuple[str, str, int]:
    """Return (api_key, model_id, output_dims) — cached after first call."""
    raw_model = os.environ.get("EMBED_MODEL", "gemini-embedding-001")
    model_id = raw_model.removeprefix("models/")  
    output_dims = int(os.environ.get("EMBED_DIMS", "768"))
    return os.environ["GEMINI_API_KEY"], model_id, output_dims


def _batch_embed(texts: list[str]) -> list[list[float]]:
    """POST to Gemini v1beta batchEmbedContents — one round-trip for all texts."""
    api_key, model_id, output_dims = _cfg()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:batchEmbedContents"
    payload = {
        "requests": [
            {
                "model": f"models/{model_id}",
                "content": {"parts": [{"text": t}]},
                "outputDimensionality": output_dims,
            }
            for t in texts
        ]
    }
    resp = httpx.post(url, params={"key": api_key}, json=payload, timeout=60.0)
    resp.raise_for_status()
    return [item["values"] for item in resp.json()["embeddings"]]


def embed_query(text: str) -> list[float]:
    return _batch_embed([text])[0]


def embed_documents(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return _batch_embed(texts)
