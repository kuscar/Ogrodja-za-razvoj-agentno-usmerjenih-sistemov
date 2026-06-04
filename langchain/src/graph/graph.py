from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from langgraph.graph import StateGraph, START, END

from src.graph.state import AgentState, ProfileSchema
from src.agents.ingestion_agent import get_ingestion_node
from src.agents.skills_enrichment_agent import get_skills_enrichment_node
from src.agents.storage_agent import get_storage_node
from src.agents.latex_compilation_agent import get_latex_compilation_node
from src.agents.validation_agent import get_validation_node
from src.agents.job_analysis_agent import get_job_analysis_node
from src.agents.ats_alignment_agent import get_ats_alignment_node
from src.security.guardrails import (
    input_guardrail_node,
    output_guardrail_node,
)

def load_profile_node(state: AgentState) -> dict:
    """Load the user's stored profile via the Supabase MCP tool."""
    from src.tools.supabase_mcp_client import supabase_mcp_tool  

    user_id = state.get("user_id")
    if not user_id:
        return {"errors": ["load_profile: user_id missing"], "phase": "done"}

    try:
        result = supabase_mcp_tool.invoke(
            {"action": "fetch_profile", "user_id": user_id, "payload": {}}
        )
        if result:
            return {"structured_profile": ProfileSchema(**result)}
    except Exception as exc:
        return {
            "errors": [
                f"No stored profile found — upload your CV first. ({exc})"
            ],
            "phase": "done",
        }

    return {
        "errors": ["No stored profile found — upload your CV first."],
        "phase": "done",
    }

def supervisor_router(state: AgentState) -> Literal[
    "ingestion",
    "skills_enrichment",
    "storage",
    "load_profile",
    "latex_compile",
    "validation",
    "job_analyzer",
    "ats_alignment",
    "output_guardrail",
]:
    """Decide which specialist runs next based on state contents."""
    phase = state.get("phase", "onboarding")

    if phase == "onboarding":
        if state.get("raw_cv_path") and not state.get("structured_profile"):
            return "ingestion"
        if state.get("enrichment_text") and not state.get("errors", []):
            return "skills_enrichment"
        if state.get("structured_profile"):
            return "storage"

    if phase == "base_cv":
        if not state.get("structured_profile"):
            return "load_profile"
        if not state.get("latex_source"):
            return "latex_compile"
        if not state.get("compiled_pdf_path"):
            return "validation"

    if phase == "targeted":
        if not state.get("structured_profile"):
            return "load_profile"
        if not state.get("job_analysis"):
            return "job_analyzer"
        if not state.get("cover_letter"):
            return "ats_alignment"
        if not state.get("compiled_pdf_path"):
            return "validation"

    return "output_guardrail"


def input_router(state: AgentState) -> Literal["supervisor", "__end__"]:
    """Gate every request through the input guardrail first."""
    return "supervisor" if state.get("guardrail_verdict") == "allow" else END


def build_graph(checkpointer: Any | None = None):
    """Compile the LangGraph state machine.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("input_guardrail", input_guardrail_node)
    workflow.add_node("output_guardrail", output_guardrail_node)

    workflow.add_node("supervisor", lambda state: {})

    workflow.add_node("load_profile", load_profile_node)

    workflow.add_node("ingestion", get_ingestion_node())
    workflow.add_node("skills_enrichment", get_skills_enrichment_node())
    workflow.add_node("storage", get_storage_node())
    workflow.add_node("latex_compile", get_latex_compilation_node())
    workflow.add_node("validation", get_validation_node())
    workflow.add_node("job_analyzer", get_job_analysis_node())
    workflow.add_node("ats_alignment", get_ats_alignment_node())

    workflow.add_edge(START, "input_guardrail")
    workflow.add_conditional_edges(
        "input_guardrail",
        input_router,
        {"supervisor": "supervisor", END: END},
    )

    workflow.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "ingestion": "ingestion",
            "skills_enrichment": "skills_enrichment",
            "storage": "storage",
            "load_profile": "load_profile",
            "latex_compile": "latex_compile",
            "validation": "validation",
            "job_analyzer": "job_analyzer",
            "ats_alignment": "ats_alignment",
            "output_guardrail": "output_guardrail",
        },
    )

    for node in (
        "ingestion",
        "skills_enrichment",
        "storage",
        "load_profile",
        "latex_compile",
        "validation",
        "job_analyzer",
        "ats_alignment",
    ):
        workflow.add_edge(node, "supervisor")

    workflow.add_edge("output_guardrail", END)

    return workflow.compile(checkpointer=checkpointer)

@lru_cache(maxsize=1)
def get_graph():
    """Return the process-wide compiled LangGraph (built once)."""
    return build_graph()
