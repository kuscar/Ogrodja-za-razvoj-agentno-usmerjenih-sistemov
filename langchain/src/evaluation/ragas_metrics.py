from __future__ import annotations

import os
from dataclasses import dataclass

from langsmith.evaluation import EvaluationResult, RunEvaluator


def _ragas_llm():
    from ragas.llms import LangchainLLMWrapper
    from src.config.settings import get_llm
    return LangchainLLMWrapper(get_llm("judge"))


def _ragas_embeddings():
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    return LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model=os.environ.get("EMBED_MODEL", "models/gemini-embedding-001"),
            google_api_key=os.environ["GEMINI_API_KEY"],
        )
    )


@dataclass
class RagasFaithfulnessEvaluator(RunEvaluator):
    """Are the claims in the tailored CV supported by the retrieved profile chunks?"""

    def evaluate_run(self, run, example=None) -> EvaluationResult:
        try:
            from ragas import EvaluationDataset, SingleTurnSample
            from ragas import evaluate as ragas_eval
            from ragas.metrics import Faithfulness

            answer = (
                (run.outputs.get("latex_source") or "")
                + "\n"
                + (run.outputs.get("cover_letter") or "")
            )
            contexts = run.outputs.get("retrieved_evidence") or []
            question = (example.inputs.get("job_description") or "") if example else ""

            if not contexts or not answer.strip():
                return EvaluationResult(
                    key="ragas_faithfulness", score=None, comment="missing contexts or answer"
                )

            sample = SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=[str(c) for c in contexts],
            )
            result = ragas_eval(
                EvaluationDataset(samples=[sample]),
                metrics=[Faithfulness(llm=_ragas_llm())],
            )
            score = float(result.to_pandas()["faithfulness"].iloc[0])
            return EvaluationResult(key="ragas_faithfulness", score=score)
        except Exception as exc:
            return EvaluationResult(key="ragas_faithfulness", score=None, comment=str(exc))


@dataclass
class RagasResponseRelevancyEvaluator(RunEvaluator):
    """Is the tailored CV/cover letter relevant to the job description?"""

    def evaluate_run(self, run, example=None) -> EvaluationResult:
        try:
            from ragas import EvaluationDataset, SingleTurnSample
            from ragas import evaluate as ragas_eval
            from ragas.metrics import ResponseRelevancy

            answer = (
                (run.outputs.get("latex_source") or "")
                + "\n"
                + (run.outputs.get("cover_letter") or "")
            )
            question = (example.inputs.get("job_description") or "") if example else ""

            if not answer.strip() or not question:
                return EvaluationResult(
                    key="ragas_response_relevancy", score=None, comment="missing answer or question"
                )

            sample = SingleTurnSample(user_input=question, response=answer)
            result = ragas_eval(
                EvaluationDataset(samples=[sample]),
                metrics=[ResponseRelevancy(llm=_ragas_llm(), embeddings=_ragas_embeddings())],
            )
            score = float(result.to_pandas()["answer_relevancy"].iloc[0])
            return EvaluationResult(key="ragas_response_relevancy", score=score)
        except Exception as exc:
            return EvaluationResult(
                key="ragas_response_relevancy", score=None, comment=str(exc)
            )


RAGAS_EVALUATORS = [
    RagasFaithfulnessEvaluator(),
    RagasResponseRelevancyEvaluator(),
]
