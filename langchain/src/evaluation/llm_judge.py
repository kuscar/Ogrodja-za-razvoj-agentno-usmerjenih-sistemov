from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from pydantic import BaseModel, Field

from src.config.settings import get_llm
from src.observability import get_logger, metrics

log = get_logger(__name__)


def _post_langsmith_feedback(judge_name: str, score: float, verdict: str) -> None:
    """Attach judge score as LangSmith feedback on the current trace."""
    try:
        run = get_current_run_tree()
        if run is None:
            return
        from langsmith import Client
        Client().create_feedback(
            run_id=str(run.id),
            key=f"{judge_name}_score",
            score=score,
            comment=f"verdict={verdict}",
        )
    except Exception:
        pass  # never let feedback posting crash the agent


class JudgeVerdict(BaseModel):
    reasoning: str = Field(
        ...,
        description="Step-by-step analysis performed before scoring.",
    )
    score: float = Field(..., ge=0.0, le=1.0)
    pass_: bool = Field(..., alias="pass")
    reasons: list[str] = Field(default_factory=list)
    flagged_claims: list[str] = Field(default_factory=list)

    class Config:
        populate_by_name = True


GROUNDEDNESS_SYSTEM_PROMPT = """\
You are a strict factual auditor. You will receive:
  * a JSON "profile" — the SOURCE OF TRUTH about a job applicant
  * a "tailored_cv" string (LaTeX)
  * a "cover_letter" string

PROCEDURE
Step 1 — Enumerate every factual claim in tailored_cv and cover_letter.
Step 2 — For each claim, check whether the profile supports it.
Step 3 — Compute score = supported_claims / total_claims.
Step 4 — Set pass = (score >= 0.9 AND no fabrications).

FABRICATIONS INCLUDE
  * new employers, titles, or dates not in the profile
  * new metrics, percentages, dollar amounts, or team sizes
  * skills, tools, or certifications absent from the profile

OUTPUT (JSON object IN THIS ORDER):
  * "reasoning"      — your stepwise analysis
  * "score"          — float 0..1
  * "pass"           — boolean
  * "reasons"        — list of short bullets
  * "flagged_claims" — verbatim fabricated sentences
"""


@dataclass
class GroundednessJudge:
    """Inline judge invoked by the ATS agent after generation."""

    llm: BaseChatModel | None = None
    pass_threshold: float = 0.9

    def __post_init__(self):
        if not self.llm:
            self.llm = get_llm("judge")

    @traceable(name="groundedness_judge", run_type="chain", tags=["evaluation", "judge"])
    def judge(self, profile: dict, tailored_cv: str, cover_letter: str) -> JudgeVerdict:
        payload = {
            "profile": profile,
            "tailored_cv": tailored_cv,
            "cover_letter": cover_letter,
        }
        structured = self.llm.with_structured_output(JudgeVerdict)
        verdict = structured.invoke(
            [
                {"role": "system", "content": GROUNDEDNESS_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, default=str)},
            ]
        )
        metrics.llm_judge_score.labels(judge="groundedness").observe(verdict.score)
        verdict_label = "pass" if verdict.pass_ else "fail"
        metrics.llm_judge_decisions_total.labels(
            judge="groundedness", verdict=verdict_label
        ).inc()
        log.info(
            "llm_judge.groundedness",
            score=verdict.score,
            verdict=verdict_label,
            flagged_count=len(verdict.flagged_claims),
        )
        _post_langsmith_feedback("groundedness", verdict.score, verdict_label)
        return verdict


COVER_LETTER_QUALITY_SYSTEM_PROMPT = """\
You are an experienced technical recruiter. Score the cover letter on a
1-5 scale across four axes, then map the average to 0..1.

AXES
* HOOK         — does paragraph 1 land a clear hook tied to the role?
* SPECIFICITY  — concrete wins, not generics?
* STRUCTURE    — three paragraphs, <=350 words, professional tone?
* CTA          — clear, confident call to action?

OUTPUT (JSON object IN THIS ORDER):
  * "reasoning"      — per-axis scores and justifications
  * "score"          — float 0..1
  * "pass"           — boolean (score >= 0.7)
  * "reasons"        — list of improvement bullets
  * "flagged_claims" — empty list
"""


@dataclass
class CoverLetterQualityJudge:
    llm: BaseChatModel | None = None
    pass_threshold: float = 0.7

    def __post_init__(self):
        if not self.llm:
            self.llm = get_llm("judge")

    @traceable(name="cover_letter_quality_judge", run_type="chain", tags=["evaluation", "judge"])
    def judge(self, cover_letter: str, job_analysis: dict) -> JudgeVerdict:
        payload = {"cover_letter": cover_letter, "job_analysis": job_analysis}
        structured = self.llm.with_structured_output(JudgeVerdict)
        verdict = structured.invoke(
            [
                {"role": "system", "content": COVER_LETTER_QUALITY_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, default=str)},
            ]
        )
        verdict_label = "pass" if verdict.pass_ else "fail"
        metrics.llm_judge_score.labels(judge="cover_letter_quality").observe(verdict.score)
        metrics.llm_judge_decisions_total.labels(
            judge="cover_letter_quality", verdict=verdict_label
        ).inc()
        _post_langsmith_feedback("cover_letter_quality", verdict.score, verdict_label)
        return verdict


@lru_cache(maxsize=1)
def get_groundedness_judge() -> GroundednessJudge:
    return GroundednessJudge()


@lru_cache(maxsize=1)
def get_cover_letter_quality_judge() -> CoverLetterQualityJudge:
    return CoverLetterQualityJudge()
