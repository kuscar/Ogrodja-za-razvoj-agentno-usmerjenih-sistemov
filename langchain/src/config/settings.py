from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from langchain_core.language_models import BaseChatModel


ROOT_DIR = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = ROOT_DIR / "templates"
PROMPT_DIR = ROOT_DIR / "prompts"

AGENT_LLM_PROFILES: dict[str, dict] = {
    "ingestion":    {"temperature": 0.0, "max_tokens": 8192, "thinking_budget":   0},
    "enrichment":   {"temperature": 0.1, "max_tokens": 8192, "thinking_budget":   0},
    "storage":      {"temperature": 0.0, "max_tokens":  512, "thinking_budget":   0},
    "latex":        {"temperature": 0.2, "max_tokens": 8192, "thinking_budget":   0},
    "validation":   {"temperature": 0.0, "max_tokens": 8192, "thinking_budget":   0},
    "job_analysis": {"temperature": 0.0, "max_tokens": 8192, "thinking_budget":   0},
    "ats":          {"temperature": 0.1, "max_tokens": 8192, "thinking_budget":   0},
    "judge":        {"temperature": 0.0, "max_tokens": 8192, "thinking_budget":   0},
}


def _profile_for(role: str) -> dict:
    return AGENT_LLM_PROFILES.get(role, {"temperature": 0.0})


@lru_cache(maxsize=16)
def get_llm(role: str) -> BaseChatModel:
    """Return a freshly configured chat model for a given agent role."""
    profile = _profile_for(role)
    provider = os.environ.get("LLM_PROVIDER", "gemini").lower()

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs: dict = {
            "model": os.environ.get("GEMINI_MODEL", "gemini-2.5-pro"),
            "temperature": profile["temperature"],
            "max_output_tokens": profile.get("max_tokens"),
            "google_api_key": os.environ["GEMINI_API_KEY"],
        }
        budget = profile.get("thinking_budget")
        if budget is not None:
            kwargs["model_kwargs"] = {"thinking_config": {"thinking_budget": budget}}
        return ChatGoogleGenerativeAI(**kwargs)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=profile["temperature"],
            max_tokens=profile.get("max_tokens"),
            api_key=os.environ["OPENAI_API_KEY"],
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            temperature=profile["temperature"],
            max_tokens=profile.get("max_tokens"),
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )

    raise ValueError(f"Unknown LLM_PROVIDER={provider!r}")
