from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


REGISTRY = CollectorRegistry()


http_requests_total = Counter(
    "cvb_http_requests_total",
    "HTTP requests grouped by endpoint and outcome.",
    ["endpoint", "method", "status"],
    registry=REGISTRY,
)

http_request_duration_seconds = Histogram(
    "cvb_http_request_duration_seconds",
    "HTTP request latency.",
    ["endpoint", "method"],
    registry=REGISTRY,
)


rate_limited_total = Counter(
    "cvb_rate_limited_total",
    "Requests blocked by rate limiter.",
    ["endpoint"],
    registry=REGISTRY,
)


agent_invocations_total = Counter(
    "cvb_agent_invocations_total",
    "How many times each agent was invoked.",
    ["agent", "outcome"],
    registry=REGISTRY,
)

agent_latency_seconds = Histogram(
    "cvb_agent_latency_seconds",
    "End-to-end latency of one agent invocation.",
    ["agent"],
    registry=REGISTRY,
)

llm_tokens_total = Counter(
    "cvb_llm_tokens_total",
    "Token usage per agent and direction.",
    ["agent", "direction"],
    registry=REGISTRY,
)


guardrail_decisions_total = Counter(
    "cvb_guardrail_decisions_total",
    "Verdicts emitted by each guardrail layer.",
    ["layer", "verdict", "reason"],
    registry=REGISTRY,
)

prompt_guard_inferences_total = Counter(
    "cvb_prompt_guard_inferences_total",
    "Calls to the Prompt Guard 2 service.",
    ["source", "label"],
    registry=REGISTRY,
)


rag_retrievals_total = Counter(
    "cvb_rag_retrievals_total",
    "RAG retrieval calls.",
    ["k_bucket"],
    registry=REGISTRY,
)

rag_retrieval_score = Histogram(
    "cvb_rag_top1_similarity",
    "Top-1 cosine similarity of retrieved chunks.",
    registry=REGISTRY,
    buckets=(0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0),
)


llm_judge_score = Histogram(
    "cvb_llm_judge_score",
    "Score returned by each LLM judge (0..1).",
    ["judge"],
    buckets=(0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0),
    registry=REGISTRY,
)

llm_judge_decisions_total = Counter(
    "cvb_llm_judge_decisions_total",
    "Pass / fail decisions emitted by an LLM judge.",
    ["judge", "verdict"],
    registry=REGISTRY,
)


in_flight_requests = Gauge(
    "cvb_in_flight_requests",
    "Requests currently being processed.",
    registry=REGISTRY,
)


def metrics_response() -> tuple[bytes, str]:
    """Return (body, content_type) for /metrics."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
