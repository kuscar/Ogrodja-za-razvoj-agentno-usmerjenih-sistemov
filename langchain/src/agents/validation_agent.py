from __future__ import annotations

import re
import time
from datetime import datetime
from functools import lru_cache

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.agents.base_agent import BaseAgent
from src.config.settings import get_llm
from src.graph.state import AgentState
from src.observability import get_logger, metrics
from src.tools.latex_validator import latex_validator_tool

log = get_logger(__name__)


VALIDATION_SYSTEM_PROMPT = """\
You are the Validation Agent — repair pass.

GOAL
Given a LaTeX source and the validator's error output, propose the SMALLEST
edit that fixes the error without removing user content.

PROCEDURE
Step 1 — Identify the specific token(s) that triggered the error.
Step 2 — Decide whether to remove or escape them.
Step 3 — Produce the corrected LaTeX with ONLY that change.

HARD RULES
1. Preserve every factual claim and every formatting decision unrelated to
   the error.
2. Never introduce new commands; only escape, remove, or replace.

OUTPUT (JSON object IN THIS ORDER):
  * "reasoning"     — which token(s) you decided to fix, why (≤ 60 words)
  * "latex_source"  — the corrected LaTeX
"""


class RepairOutput(BaseModel):
    reasoning: str = Field(..., description="Which tokens were fixed and why.")
    latex_source: str


class ValidationAgent(BaseAgent):
    name = "validation"
    system_prompt = VALIDATION_SYSTEM_PROMPT

    def __init__(self):
        super().__init__(llm=get_llm("validation"), tool=latex_validator_tool)

    @staticmethod
    def _build_friendly_name(state: AgentState) -> str:
        profile = state.get("structured_profile")
        job = state.get("job_analysis")
        name_part = re.sub(r"[^a-z0-9]+", "_", (profile.full_name if profile else "cv").lower()).strip("_")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if job and job.title:
            title_part = re.sub(r"[^a-z0-9]+", "_", job.title.lower()).strip("_")
            return f"cv_{name_part}_{title_part}_{ts}"
        return f"cv_{name_part}_{ts}"

    def __call__(self, state: AgentState) -> dict:
        start = time.perf_counter()
        latex_src = state.get("latex_source")
        user_id = state.get("user_id")
        if not (latex_src and user_id):
            metrics.agent_invocations_total.labels(
                agent=self.name, outcome="error"
            ).inc()
            return {"errors": ["validation: latex_source or user_id missing"], "phase": "done"}

        friendly_name = self._build_friendly_name(state)
        result = latex_validator_tool.invoke(
            {"latex_source": latex_src, "user_id": user_id, "friendly_name": friendly_name}
        )

        if result["ok"]:
            metrics.agent_invocations_total.labels(
                agent=self.name, outcome="success"
            ).inc()
            metrics.agent_latency_seconds.labels(agent=self.name).observe(
                time.perf_counter() - start
            )
            return {
                "compiled_pdf_path": result["pdf_path"],
                "compiled_pdf_name": result["pdf_name"],
                "phase": "done",
                "messages": [HumanMessage(content="CV compiled successfully")],
            }

        log.warning("validation.repair_attempt")
        metrics.agent_invocations_total.labels(
            agent=self.name, outcome="retry"
        ).inc()
        structured = self.llm.with_structured_output(RepairOutput)
        repair: RepairOutput = structured.invoke(
            [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Validator errors:\n{result['errors']}\n\n"
                        f"Original LaTeX:\n{latex_src}"
                    ),
                },
            ]
        )

        import re as _re
        repaired_src = _re.sub(r'\\\\(?![\\[\] ])', r'\\', repair.latex_source)
        retry = latex_validator_tool.invoke(
            {"latex_source": repaired_src, "user_id": user_id, "friendly_name": friendly_name}
        )
        if retry["ok"]:
            metrics.agent_invocations_total.labels(
                agent=self.name, outcome="success"
            ).inc()
            metrics.agent_latency_seconds.labels(agent=self.name).observe(
                time.perf_counter() - start
            )
            return {
                "latex_source": repaired_src,
                "compiled_pdf_path": retry["pdf_path"],
                "compiled_pdf_name": retry["pdf_name"],
                "phase": "done",
                "messages": [HumanMessage(content="CV repaired & compiled")],
            }

        metrics.agent_invocations_total.labels(
            agent=self.name, outcome="error"
        ).inc()
        metrics.agent_latency_seconds.labels(agent=self.name).observe(
            time.perf_counter() - start
        )
        return {
            "errors": [f"validation: compile failed after repair — {retry['errors']}"],
            "phase": "done",
            "messages": [HumanMessage(content="CV failed to compile")],
        }


@lru_cache(maxsize=1)
def get_validation_node() -> ValidationAgent:
    return ValidationAgent()
