from __future__ import annotations

import time
from functools import lru_cache

from langchain_core.messages import HumanMessage

from src.agents.base_agent import BaseAgent
from src.config.settings import get_llm
from src.graph.state import AgentState
from src.observability import get_logger, metrics
from src.security.prompt_guard import scan as prompt_guard_scan
from src.tools.supabase_mcp_client import supabase_mcp_tool

log = get_logger(__name__)


STORAGE_SYSTEM_PROMPT = """\
You are the Storage Agent.

GOAL
Persist the user's structured profile to Supabase using the `supabase_mcp_tool`.

HARD RULES
1. Always pass `user_id` exactly as supplied in state — never modify it.
2. Use ONLY the allow-listed RPC actions: `upsert_profile`, `embed_experiences`.
3. Never call destructive actions (delete, drop, truncate).
4. Output a short confirmation string.
"""


def _scrub_bullets_for_rag(bullets: list[str]) -> tuple[list[str], list[str]]:
    """Run each bullet through Prompt Guard 2. Returns (clean, dropped)."""
    clean, dropped = [], []
    for b in bullets:
        block, reason = prompt_guard_scan(b, source="embed")
        if block:
            dropped.append(reason or b[:30])
        else:
            clean.append(b)
    return clean, dropped


class StorageAgent(BaseAgent):
    name = "storage"
    system_prompt = STORAGE_SYSTEM_PROMPT

    def __init__(self):
        super().__init__(llm=get_llm("storage"), tool=supabase_mcp_tool)

    def __call__(self, state: AgentState) -> dict:
        start = time.perf_counter()
        profile = state.get("structured_profile")
        user_id = state.get("user_id")
        if not (profile and user_id):
            metrics.agent_invocations_total.labels(agent=self.name, outcome="error").inc()
            return {"errors": ["storage: profile or user_id missing"]}

        cleansed_experiences = []
        all_dropped: list[str] = []
        for exp in profile.experiences:
            clean, dropped = _scrub_bullets_for_rag(exp.bullets)
            cleansed_experiences.append(exp.model_copy(update={"bullets": clean}))
            all_dropped.extend(dropped)

        if all_dropped:
            log.warning(
                "storage.bullets_dropped",
                count=len(all_dropped),
                reasons=all_dropped[:3],
            )

        safe_profile = profile.model_copy(update={"experiences": cleansed_experiences})

        supabase_mcp_tool.invoke(
            {
                "action": "upsert_profile",
                "user_id": user_id,
                "payload": safe_profile.model_dump(),
            }
        )
        supabase_mcp_tool.invoke(
            {
                "action": "embed_experiences",
                "user_id": user_id,
                "payload": {
                    "experiences": [
                        e.model_dump() for e in safe_profile.experiences
                    ]
                },
            }
        )

        metrics.agent_invocations_total.labels(agent=self.name, outcome="success").inc()
        metrics.agent_latency_seconds.labels(agent=self.name).observe(
            time.perf_counter() - start
        )
        log.info("storage.persisted", dropped=len(all_dropped))
        return {
            "phase": "base_cv",
            "structured_profile": safe_profile,
            "messages": [HumanMessage(content="Profile persisted to Supabase")],
        }


@lru_cache(maxsize=1)
def get_storage_node() -> StorageAgent:
    return StorageAgent()
