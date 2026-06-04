from __future__ import annotations

from typing import Annotated, Literal, Optional
from operator import add

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

class ExperienceItem(BaseModel):
    company: str
    title: str
    start_date: str
    end_date: Optional[str] = None
    bullets: list[str] = Field(default_factory=list)


class EducationItem(BaseModel):
    institution: str
    degree: str
    field: Optional[str] = None
    start_date: str
    end_date: Optional[str] = None


class ProfileSchema(BaseModel):
    full_name: str
    email: str
    phone: Optional[str] = None
    location: Optional[str] = None
    hard_skills: list[str] = Field(default_factory=list)
    soft_skills: list[str] = Field(default_factory=list)
    experiences: list[ExperienceItem] = Field(default_factory=list)
    education: list[EducationItem] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    leadership: list[str] = Field(default_factory=list)
    extracurricular_activities: list[str] = Field(default_factory=list)


class JobAnalysisSchema(BaseModel):
    title: str
    company: Optional[str] = None
    hard_skills: list[str] = Field(default_factory=list)
    soft_skills: list[str] = Field(default_factory=list)
    ats_keywords: list[str] = Field(default_factory=list)
    seniority: Optional[str] = None
    requirements: list[str] = Field(default_factory=list)



class AgentState(TypedDict, total=False):
    user_id: str

    phase: Literal["onboarding", "base_cv", "targeted", "done"]

    raw_cv_path: Optional[str]
    enrichment_text: Optional[str]
    job_description: Optional[str]

    structured_profile: Optional[ProfileSchema]
    job_analysis: Optional[JobAnalysisSchema]
    latex_source: Optional[str]
    cover_letter: Optional[str]
    compiled_pdf_path: Optional[str]
    compiled_pdf_name: Optional[str]

    messages: Annotated[list[BaseMessage], add_messages]

    guardrail_verdict: Literal["allow", "block"]
    guardrail_reason: Optional[str]

    retrieved_evidence: Optional[list]

    errors: Annotated[list[str], add]

    next: Optional[str]
