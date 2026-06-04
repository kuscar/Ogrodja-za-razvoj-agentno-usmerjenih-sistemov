from __future__ import annotations

import os
from functools import lru_cache

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_ID = os.environ.get("PROMPT_GUARD_MODEL", "meta-llama/Llama-Prompt-Guard-2-86M")
DEVICE = os.environ.get("PROMPT_GUARD_DEVICE", "cpu")
THRESH = float(os.environ.get("PROMPT_GUARD_THRESH", "0.5"))
HF_TOKEN = os.environ.get("HF_TOKEN")

app = FastAPI(title="prompt-guard-2", version="1.0.0")


class ClassifyIn(BaseModel):
    text: str = Field(..., max_length=20_000)
    source: str | None = None


class ClassifyOut(BaseModel):
    label: str          
    score: float          
    block: bool           
    truncated: bool


@lru_cache(maxsize=1)
def _model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, token=HF_TOKEN
    ).to(DEVICE)
    mdl.eval()
    return tok, mdl


@app.on_event("startup")
def _warm():
    _model()


@app.post("/classify", response_model=ClassifyOut)
def classify(req: ClassifyIn) -> ClassifyOut:
    tok, mdl = _model()
    enc = tok(
        req.text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    truncated = enc["input_ids"].shape[1] >= 512

    with torch.no_grad():
        logits = mdl(**{k: v.to(DEVICE) for k, v in enc.items()}).logits
    probs = torch.softmax(logits, dim=-1)[0].tolist()
    id2label = mdl.config.id2label

    safe_names = {"BENIGN", "LABEL_0"}
    unsafe_idx = [
        i for i, name in id2label.items()
        if name.upper() not in safe_names
    ]
    unsafe_max = max((probs[i] for i in unsafe_idx), default=0.0)

    _readable = {"LABEL_0": "BENIGN", "LABEL_1": "INJECTION", "LABEL_2": "JAILBREAK"}

    pred_idx = int(max(range(len(probs)), key=probs.__getitem__))
    raw_label = id2label[pred_idx].upper()
    label = _readable.get(raw_label, raw_label)
    return ClassifyOut(
        label=label,
        score=probs[pred_idx],
        block=unsafe_max >= THRESH,
        truncated=truncated,
    )


@app.get("/healthz")
def healthz():
    _model()
    return {"ok": True, "model": MODEL_ID, "device": DEVICE, "threshold": THRESH}
