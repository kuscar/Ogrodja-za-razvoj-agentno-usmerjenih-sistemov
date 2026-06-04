from __future__ import annotations

import json
import time
from functools import lru_cache

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.agents.base_agent import BaseAgent
from src.agents.latex_compilation_agent import _jinja_env
from src.config.settings import get_llm
from src.evaluation.llm_judge import get_groundedness_judge
from src.graph.state import AgentState, ProfileSchema
from src.observability import get_logger, metrics
from src.rag.retriever import rag_retriever_tool

log = get_logger(__name__)


ATS_SYSTEM_PROMPT = """\
You are the ATS Alignment Agent.

GOAL
Produce a MEANINGFULLY DIFFERENT tailored CV and cover letter that maximises
ATS overlap with the target job — using ONLY facts, skills, and experiences
explicitly present in the user's profile. NEVER invent skills, technologies,
metrics, certifications, employers, or dates.
JD stands for job description.

═══════════════════════════════════════════════════════
STEP 0 — SKILL INTERSECTION ANALYSIS  (do this first, in your reasoning)
═══════════════════════════════════════════════════════
  a. List every hard skill the JD explicitly requires.
  b. For each JD skill, check if it appears (exactly or approximately) in
     profile.hard_skills or profile.soft_skills.
  c. Build three groups:
       MATCHED   — skills in BOTH the JD and the profile  → emphasise everywhere
       ABSENT    — skills the JD needs but profile lacks  → do NOT add; note as gap
       DEMOTED   — skills in the profile NOT mentioned in the JD → push to end or omit

═══════════════════════════════════════════════════════
STEP 1 — REORDER SKILLS
═══════════════════════════════════════════════════════
  • hard_skills: MATCHED skills come FIRST in JD priority order.
    DEMOTED skills move to the very end or are removed entirely.
  • soft_skills: same principle.

═══════════════════════════════════════════════════════
STEP 2 — REWRITE EXPERIENCE BULLETS
═══════════════════════════════════════════════════════
  • For every bullet that relates to a MATCHED skill:
      - Rewrite it to use the JD's EXACT terminology where factually correct.
        e.g. profile says "built APIs" + JD says "REST API development"
             → rewrite as "Developed RESTful APIs …"
      - The underlying fact must remain unchanged; only the framing shifts.
  • Move the most JD-relevant bullet to the TOP of each role's list.
  • Bullets unrelated to any JD requirement go LAST or are removed.

═══════════════════════════════════════════════════════
STEP 3 — COVER LETTER  (<= 350 words, three paragraphs)
═══════════════════════════════════════════════════════
  Paragraph 1 — HOOK
    Name the role, company (if known), and the 2-3 most critical MATCHED skills
    from the intersection analysis. Be specific.

  Paragraph 2 — EVIDENCE
    Pick 2-3 concrete experiences that directly match JD requirements.
    Use the JD's OWN language for technologies/tools (only if they appear in
    the profile). Include at least one quantified achievement if the profile
    contains one.

  Paragraph 3 — FIT + CTA
    Why this company/role specifically. Confident, direct call to action.

  COVER LETTER MUST reference the specific technologies from the MATCHED list —
  not generic phrases like "strong technical background".

═══════════════════════════════════════════════════════
ABSOLUTE HARD RULES
═══════════════════════════════════════════════════════
1. Only use skills the profile explicitly lists. If a skill is absent from the
   profile, do NOT add it — even if the JD demands it.
2. Rewriting bullets to use JD terminology is allowed; inventing new facts is not.
3. Do not add employers, roles, dates, certifications, or metrics that are not
   already in the profile.

OUTPUT (JSON object IN THIS ORDER):
  * "reasoning"        — STEP 0 intersection analysis (MATCHED / ABSENT / DEMOTED
                         lists) + reordering + rewrite plan (<= 300 words)
  * "tailored_profile" — tailored version of the profile
  * "cover_letter"     — final cover letter Markdown
"""


class ATSOutput(BaseModel):
    reasoning: str = Field(
        ...,
        description="Plan: keyword->bullet mapping, demotions, ordering.",
    )
    tailored_profile: ProfileSchema
    cover_letter: str


class ATSAlignmentAgent(BaseAgent):
    name = "ats_alignment"
    system_prompt = ATS_SYSTEM_PROMPT

    def __init__(self):
        super().__init__(llm=get_llm("ats"), tool=rag_retriever_tool)

    def _generate(self, payload: dict, extra_system: str = "") -> ATSOutput:
        structured = self.llm.with_structured_output(ATSOutput)
        system = self.system_prompt + (
            f"\n\nADDITIONAL CONSTRAINTS\n{extra_system}" if extra_system else ""
        )
        return structured.invoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, default=str)},
            ]
        )

    def __call__(self, state: AgentState) -> dict:
        start = time.perf_counter()
        profile = state.get("structured_profile")
        job = state.get("job_analysis")
        user_id = state.get("user_id")
        if not profile or not job or not user_id:
            metrics.agent_invocations_total.labels(
                agent=self.name, outcome="error"
            ).inc()
            return {"errors": ["ats: profile, job_analysis, or user_id missing"]}

        try:
            retrieved = rag_retriever_tool.invoke(
                {
                    "user_id": user_id,
                    "query": " ".join(job.ats_keywords[:25]),
                    "k": 12,
                }
            )
        except RuntimeError as exc:
            metrics.agent_invocations_total.labels(
                agent=self.name, outcome="error"
            ).inc()
            return {"errors": [f"ats: {exc}"], "phase": "done"}

        prompt_payload = {
            "profile": profile.model_dump(),
            "job_analysis": job.model_dump(),
            "retrieved_evidence": retrieved,
        }

        out = self._generate(prompt_payload)
        if out is None:
            metrics.agent_invocations_total.labels(
                agent=self.name, outcome="error"
            ).inc()
            return {
                "errors": ["ats: LLM returned no structured output — try again"],
                "phase": "done",
            }
        log.info("ats.generated", reasoning_chars=len(out.reasoning))

        template = _jinja_env().get_template("faang_cv_template.tex")
        rendered_latex = template.render(profile=out.tailored_profile)

        judge = get_groundedness_judge()
        verdict = judge.judge(
            profile=profile.model_dump(),
            tailored_cv=rendered_latex,
            cover_letter=out.cover_letter,
        )

        if not verdict.pass_ and verdict.flagged_claims:
            log.warning(
                "ats.regenerating_after_judge",
                flagged_count=len(verdict.flagged_claims),
                score=verdict.score,
            )
            metrics.agent_invocations_total.labels(
                agent=self.name, outcome="retry"
            ).inc()
            forbidden = "\n".join(f"- {c}" for c in verdict.flagged_claims[:20])
            retry_constraint = (
                "The previous draft contained the following UNSUPPORTED claims. "
                "DO NOT include or rephrase any of them in this draft:\n"
                + forbidden
            )
            out = self._generate(prompt_payload, extra_system=retry_constraint)
            if out is None:
                metrics.agent_invocations_total.labels(
                    agent=self.name, outcome="error"
                ).inc()
                return {
                    "errors": [
                        "ats: LLM returned no structured output on retry — try again"
                    ],
                    "phase": "done",
                }
            rendered_latex = template.render(profile=out.tailored_profile)
            verdict = judge.judge(
                profile=profile.model_dump(),
                tailored_cv=rendered_latex,
                cover_letter=out.cover_letter,
            )

        metrics.agent_invocations_total.labels(
            agent=self.name,
            outcome="success" if verdict.pass_ else "error",
        ).inc()
        metrics.agent_latency_seconds.labels(agent=self.name).observe(
            time.perf_counter() - start
        )

        errors = []
        if not verdict.pass_:
            errors.append(
                f"ats: groundedness judge fail after retry (score={verdict.score:.2f})"
            )

        return {
            "latex_source": rendered_latex,
            "cover_letter": out.cover_letter,
            "structured_profile": out.tailored_profile,
            "retrieved_evidence": retrieved,
            "errors": errors,
            "messages": [HumanMessage(content="ATS alignment complete")],
        }


@lru_cache(maxsize=1)
def get_ats_alignment_node() -> ATSAlignmentAgent:
    return ATSAlignmentAgent()
