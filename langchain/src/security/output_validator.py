from __future__ import annotations

import re
from dataclasses import dataclass

from src.rag.embedder import embed_documents


@dataclass
class GroundingReport:
    score: float                    
    ungrounded_claims: list[str]
    threshold: float


_BULLET_SPLIT = re.compile(r"(?:^|\n)\s*[-*•]\s+|\\item\s+")


def _split_claims(text: str) -> list[str]:
    chunks = [c.strip() for c in _BULLET_SPLIT.split(text) if c.strip()]
    return [c for c in chunks if len(c.split()) >= 4]


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if (na and nb) else 0.0


def grounding_report(
    tailored_text: str,
    grounding_corpus: list[str],
    threshold: float = 0.85,
) -> GroundingReport:
    claims = _split_claims(tailored_text)
    if not claims or not grounding_corpus:
        return GroundingReport(score=0.0, ungrounded_claims=claims, threshold=threshold)

    claim_emb = embed_documents(claims)
    corpus_emb = embed_documents(grounding_corpus)

    ungrounded = []
    for claim_text, ce in zip(claims, claim_emb):
        best = max(_cos(ce, ke) for ke in corpus_emb)
        if best < threshold:
            ungrounded.append(claim_text)

    return GroundingReport(
        score=1.0 - (len(ungrounded) / len(claims)),
        ungrounded_claims=ungrounded,
        threshold=threshold,
    )
