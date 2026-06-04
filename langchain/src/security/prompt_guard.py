from __future__ import annotations

import os
from typing import Literal

import httpx

from src.observability import get_logger, metrics

log = get_logger(__name__)

PromptGuardSource = Literal["input", "ingest", "embed", "retrieve"]

CHUNK_CHARS = 1500


class PromptGuardClient:
    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 10.0,
    ):
        self.base_url = base_url or os.environ.get(
            "PROMPT_GUARD_URL", "http://prompt-guard:8088"
        )
        self.timeout = timeout

    def classify(self, text: str, source: PromptGuardSource) -> dict:
        try:
            resp = httpx.post(
                f"{self.base_url}/classify",
                json={"text": text, "source": source},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            # Fail closed.
            log.error("prompt_guard.unreachable", error=str(exc), source=source)
            metrics.prompt_guard_inferences_total.labels(
                source=source, label="UNREACHABLE"
            ).inc()
            return {"label": "UNREACHABLE", "score": 1.0, "block": True}

        metrics.prompt_guard_inferences_total.labels(
            source=source, label=data["label"]
        ).inc()
        return data


_client: PromptGuardClient | None = None


def _get_client() -> PromptGuardClient:
    global _client
    if _client is None:
        _client = PromptGuardClient()
    return _client


def _chunks(text: str, size: int = CHUNK_CHARS) -> list[str]:
    text = text or ""
    return [text[i:i + size] for i in range(0, len(text), size)] or [""]


def scan(text: str, *, source: PromptGuardSource) -> tuple[bool, str | None]:
    """
    Returns (block, reason). Splits long text into 512-token-safe chunks
    and blocks if ANY chunk is flagged.

    Skipped (allow) when PROMPT_GUARD_URL is not set — the sidecar is not
    running (e.g. local dev without Docker). Set the env var to enable.
    """
    if not text or not text.strip():
        return False, None
    if not os.environ.get("PROMPT_GUARD_URL"):
        return False, None

    client = _get_client()
    for idx, chunk in enumerate(_chunks(text)):
        res = client.classify(chunk, source=source)
        if res["block"]:
            return True, f"prompt_guard_2:{res['label']}:chunk={idx}:score={res['score']:.2f}"
    return False, None
