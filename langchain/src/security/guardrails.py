from __future__ import annotations

import re
from typing import Any

from src.graph.state import AgentState
from src.observability import get_logger, metrics
from src.security.prompt_guard import scan as prompt_guard_scan

log = get_logger(__name__)

JAILBREAK_PATTERNS = [
    re.compile(r"ignore (all |the )?(previous|prior) (instructions?|prompts?)", re.I),
    re.compile(r"disregard (all |the )?(previous|prior)", re.I),
    re.compile(r"you (are|act as) (now |) ?(an? )?(developer|admin|root)", re.I),
    re.compile(r"reveal (your )?(system|hidden) prompt", re.I),
    re.compile(r"<\|im_start\|>system", re.I),
    re.compile(r"do anything now|dan mode", re.I),
]

LEAKAGE_PATTERNS = [
    re.compile(r"system prompt is", re.I),
    re.compile(r"BEGIN PRIVATE KEY", re.I),
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    re.compile(r"eyJ[a-zA-Z0-9_\-]{15,}\."),
]


def _lexical_block(text: str, patterns: list[re.Pattern]) -> str | None:
    for pat in patterns:
        if pat.search(text):
            return pat.pattern
    return None


def _collect_user_text(state: AgentState) -> str:
    return "\n".join(
        filter(
            None,
            [
                state.get("enrichment_text") or "",
                state.get("job_description") or "",
            ],
        )
    )


def input_guardrail_node(state: AgentState) -> dict[str, Any]:
    """Block prompt injection at the door."""
    text = _collect_user_text(state)
    if not text.strip():
        metrics.guardrail_decisions_total.labels(
            layer="lexical", verdict="allow", reason="empty"
        ).inc()
        return {"guardrail_verdict": "allow"}

    if hit := _lexical_block(text, JAILBREAK_PATTERNS):
        metrics.guardrail_decisions_total.labels(
            layer="lexical", verdict="block", reason="jailbreak_pattern"
        ).inc()
        log.warning("guardrail.block", layer="lexical", reason=hit)
        return {
            "guardrail_verdict": "block",
            "guardrail_reason": f"jailbreak pattern: {hit}",
            "errors": ["input_guardrail: lexical block"],
        }

    block, reason = prompt_guard_scan(text, source="input")
    if block:
        metrics.guardrail_decisions_total.labels(
            layer="prompt_guard_2", verdict="block", reason=reason or "unknown"
        ).inc()
        log.warning("guardrail.block", layer="prompt_guard_2", reason=reason)
        return {
            "guardrail_verdict": "block",
            "guardrail_reason": reason,
            "errors": ["input_guardrail: prompt_guard_2 block"],
        }

    metrics.guardrail_decisions_total.labels(
        layer="prompt_guard_2", verdict="allow", reason="benign"
    ).inc()
    return {"guardrail_verdict": "allow"}


def output_guardrail_node(state: AgentState) -> dict[str, Any]:
    """Scrub final artifacts for leakage / dangerous content (LLM07)."""
    pieces = [
        state.get("latex_source") or "",
        state.get("cover_letter") or "",
    ]
    blob = "\n".join(pieces)

    if hit := _lexical_block(blob, LEAKAGE_PATTERNS):
        metrics.guardrail_decisions_total.labels(
            layer="output", verdict="block", reason="leakage_pattern"
        ).inc()
        log.warning("guardrail.output_block", reason=hit)
        return {
            "errors": [f"output_guardrail: leakage pattern '{hit}' redacted"],
            "latex_source": None,
            "cover_letter": None,
        }
    metrics.guardrail_decisions_total.labels(
        layer="output", verdict="allow", reason="clean"
    ).inc()
    return {}
