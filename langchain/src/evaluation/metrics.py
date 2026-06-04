from __future__ import annotations

from dataclasses import dataclass

from langsmith.evaluation import EvaluationResult, RunEvaluator

from src.security.output_validator import grounding_report


@dataclass
class GroundednessEvaluator(RunEvaluator):
    threshold: float = 0.85

    def evaluate_run(self, run, example=None) -> EvaluationResult:
        tailored = run.outputs.get("latex_source", "") or ""
        grounding = example.inputs.get("grounding_corpus", []) if example else []
        report = grounding_report(tailored, grounding, threshold=self.threshold)
        return EvaluationResult(
            key="groundedness",
            score=report.score,
            comment=f"ungrounded={report.ungrounded_claims[:3]}",
        )


@dataclass
class LatexCompileEvaluator(RunEvaluator):
    def evaluate_run(self, run, example=None) -> EvaluationResult:
        ok = bool(run.outputs.get("compiled_pdf_path"))
        return EvaluationResult(key="latex_compile_rate", score=1.0 if ok else 0.0)


@dataclass
class KeywordRecallEvaluator(RunEvaluator):
    def evaluate_run(self, run, example=None) -> EvaluationResult:
        keywords = (
            example.inputs.get("job_analysis", {}).get("ats_keywords", [])
            if example
            else []
        )
        text = (run.outputs.get("latex_source") or "").lower()
        if not keywords:
            return EvaluationResult(key="keyword_recall", score=0.0)
        hits = sum(1 for kw in keywords if kw.lower() in text)
        return EvaluationResult(
            key="keyword_recall",
            score=hits / len(keywords),
            comment=f"{hits}/{len(keywords)} keywords hit",
        )


EVALUATORS = [
    GroundednessEvaluator(),
    LatexCompileEvaluator(),
    KeywordRecallEvaluator(),
]
