from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from src.services.diagnostics_service import DiagnosticsService


app = FastAPI(title="MedDiagnost API", version="1.0.0")
service = DiagnosticsService()


class AnalyzeRequest(BaseModel):
    text: str


class RecommendRequest(BaseModel):
    query: str
    top_k: int = 3


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze")
def analyze(payload: AnalyzeRequest) -> dict:
    return service.analyze_text(payload.text)


@app.post("/recommend")
def recommend(payload: RecommendRequest) -> dict:
    return service.recommend_diseases(payload.query, top_k=payload.top_k)
