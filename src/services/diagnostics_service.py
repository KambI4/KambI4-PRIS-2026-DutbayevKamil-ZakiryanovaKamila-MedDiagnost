from __future__ import annotations

from src.data_loader import load_diseases_from_json
from src.knowledge_graph import create_graph
from src.logic import process_text_message
from src.recommendation import recommend_by_fuzzy, recommend_labeled_by_cosine


class DiagnosticsService:
    def __init__(self, graph=None):
        self.graph = graph or create_graph()
        self.diseases = load_diseases_from_json()

    def analyze_text(self, text: str) -> dict:
        answer = process_text_message(text, self.graph)
        return {"query": text, "answer": answer}

    def recommend_diseases(self, query: str, top_k: int = 3) -> dict:
        disease_profiles = []
        disease_names = []

        for disease in self.diseases:
            disease_names.append(disease.name)
            profile = " ".join(
                [disease.name, *disease.symptoms, *disease.medicines]
            )
            disease_profiles.append((disease.name, profile))

        cosine = recommend_labeled_by_cosine(query, disease_profiles, top_k=top_k)
        fuzzy = recommend_by_fuzzy(query, disease_names, top_k=top_k)
        return {
            "query": query,
            "cosine": [{"item": item, "score": score} for item, score in cosine],
            "fuzzy": [{"item": item, "score": score} for item, score in fuzzy],
        }
