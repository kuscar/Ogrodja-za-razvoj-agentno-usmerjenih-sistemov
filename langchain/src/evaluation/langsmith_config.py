from __future__ import annotations

import os

from langsmith import Client


def configure_langsmith() -> Client | None:
    if not os.environ.get("LANGCHAIN_API_KEY"):
        return None
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ.setdefault("LANGCHAIN_PROJECT", "cv-builder")
    return Client()
