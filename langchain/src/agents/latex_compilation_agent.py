from __future__ import annotations

import time
from functools import lru_cache

from jinja2 import Environment, FileSystemLoader, select_autoescape
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.agents.base_agent import BaseAgent
from src.config.settings import TEMPLATE_DIR, get_llm
from src.graph.state import AgentState
from src.observability import get_logger, metrics

log = get_logger(__name__)


LATEX_SYSTEM_PROMPT = """\
You are the LaTeX Compilation Agent.

GOAL
Rewrite each experience bullet in a strong, quantified, FAANG-style voice
using the STAR pattern. You may NOT invent metrics or technologies — only
rephrase what is already present.

HARD RULES
1. Preserve every factual claim. No new dates, no new companies.
2. Each bullet <= 24 words, starts with a strong action verb.
3. Output a JSON object IN THIS ORDER:
   * "reasoning" — short step-by-step rationale (<= 80 words)
   * "bullets"   — JSON array in the same order as input
"""


_LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def tex_escape(value: object) -> str:
    """Escape user-supplied content before LaTeX rendering."""
    if value is None:
        return ""
    s = str(value)
    s = "".join(ch for ch in s if ord(ch) >= 0x20 or ch in "\t\n\r")
    return "".join(_LATEX_SPECIALS.get(ch, ch) for ch in s)


@lru_cache(maxsize=1)
def _jinja_env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(disabled_extensions=("tex",)),
        block_start_string="((*",
        block_end_string="*))",
        variable_start_string="(((",
        variable_end_string=")))",
        comment_start_string="((#",
        comment_end_string="#))",
        finalize=lambda v: "" if v is None else v,
    )
    env.filters["tex_escape"] = tex_escape
    return env


class BulletRewriteOutput(BaseModel):
    reasoning: str = Field(
        ...,
        description="Short STAR/verb rationale before the bullets.",
    )
    bullets: list[str] = Field(default_factory=list)


class LatexCompilationAgent(BaseAgent):
    name = "latex_compile"
    system_prompt = LATEX_SYSTEM_PROMPT

    def __init__(self):
        super().__init__(llm=get_llm("latex"), tool=None)

    def _rewrite_bullets(self, bullets: list[str]) -> list[str]:
        if not bullets:
            return []
        structured = self.llm.with_structured_output(BulletRewriteOutput)
        result: BulletRewriteOutput = structured.invoke(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": "\n".join(f"- {b}" for b in bullets)},
            ]
        )
        log.info("latex.bullets_rewritten", n=len(result.bullets))
        return result.bullets

    def __call__(self, state: AgentState) -> dict:
        start = time.perf_counter()
        profile = state.get("structured_profile")
        if not profile:
            metrics.agent_invocations_total.labels(
                agent=self.name, outcome="error"
            ).inc()
            return {"errors": ["latex: structured_profile missing"], "phase": "done"}

        rewritten_exps = []
        for exp in profile.experiences:
            new_bullets = self._rewrite_bullets(exp.bullets)
            rewritten_exps.append(exp.model_copy(update={"bullets": new_bullets}))

        template = _jinja_env().get_template("faang_cv_template.tex")
        latex_src = template.render(
            profile=profile.model_copy(update={"experiences": rewritten_exps})
        )

        metrics.agent_invocations_total.labels(
            agent=self.name, outcome="success"
        ).inc()
        metrics.agent_latency_seconds.labels(agent=self.name).observe(
            time.perf_counter() - start
        )
        return {
            "latex_source": latex_src,
            "messages": [HumanMessage(content="LaTeX source generated")],
        }


@lru_cache(maxsize=1)
def get_latex_compilation_node() -> LatexCompilationAgent:
    return LatexCompilationAgent()
