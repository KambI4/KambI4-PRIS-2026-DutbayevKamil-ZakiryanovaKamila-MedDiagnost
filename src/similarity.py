from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable


IMAGE_PROTOTYPES = [
    {
        "id": "xray_chest_infiltrate",
        "label": "Рентген грудной клетки с описанием затемнения",
        "category": "xray_image",
        "description": "рентген грудная клетка легкие инфильтрация затемнение пневмония снимок",
        "keywords": ["рентген", "легкие", "инфильтрация", "пневмония", "снимок"],
        "visual_profile": {
            "mean_intensity": 82.0,
            "std_intensity": 52.0,
            "edge_density": 0.19,
            "white_ratio": 0.11,
            "ocr_length": 8.0,
        },
    },
    {
        "id": "xray_fracture",
        "label": "Рентген конечности при подозрении на перелом",
        "category": "xray_image",
        "description": "рентген кость перелом травма конечность снимок",
        "keywords": ["рентген", "кость", "перелом", "травма", "снимок"],
        "visual_profile": {
            "mean_intensity": 88.0,
            "std_intensity": 48.0,
            "edge_density": 0.17,
            "white_ratio": 0.15,
            "ocr_length": 6.0,
        },
    },
    {
        "id": "prescription_antibiotic",
        "label": "Рецепт с антибиотиком и дозировкой",
        "category": "prescription_document",
        "description": "рецепт амоксициллин антибиотик дозировка принимать два раза в день",
        "keywords": ["рецепт", "амоксициллин", "дозировка", "принимать", "антибиотик"],
        "visual_profile": {
            "mean_intensity": 211.0,
            "std_intensity": 34.0,
            "edge_density": 0.07,
            "white_ratio": 0.74,
            "ocr_length": 95.0,
        },
    },
    {
        "id": "prescription_painkiller",
        "label": "Рецепт с обезболивающим и схемой приема",
        "category": "prescription_document",
        "description": "рецепт ибупрофен таблетки после еды три раза в день",
        "keywords": ["рецепт", "ибупрофен", "таблетки", "после еды", "схема"],
        "visual_profile": {
            "mean_intensity": 206.0,
            "std_intensity": 31.0,
            "edge_density": 0.08,
            "white_ratio": 0.7,
            "ocr_length": 88.0,
        },
    },
    {
        "id": "diagnostic_report_pneumonia",
        "label": "Диагностическое заключение по пневмонии",
        "category": "diagnostic_report",
        "description": "заключение диагноз пневмония жалобы температура кашель пациент",
        "keywords": ["заключение", "диагноз", "пневмония", "кашель", "пациент"],
        "visual_profile": {
            "mean_intensity": 219.0,
            "std_intensity": 26.0,
            "edge_density": 0.05,
            "white_ratio": 0.82,
            "ocr_length": 180.0,
        },
    },
    {
        "id": "diagnostic_report_virus",
        "label": "Диагностическое заключение по вирусной инфекции",
        "category": "diagnostic_report",
        "description": "диагноз вирусная инфекция осмотр жалобы слабость температура",
        "keywords": ["диагноз", "вирусная", "инфекция", "осмотр", "жалобы"],
        "visual_profile": {
            "mean_intensity": 217.0,
            "std_intensity": 24.0,
            "edge_density": 0.05,
            "white_ratio": 0.8,
            "ocr_length": 170.0,
        },
    },
]


TOKEN_PATTERN = re.compile(r"[a-zA-Zа-яА-Я0-9]+")
TEXT_VOCABULARY = sorted(
    {token for item in IMAGE_PROTOTYPES for token in TOKEN_PATTERN.findall(item["description"].lower())}
)
VISUAL_KEYS = ("mean_intensity", "std_intensity", "edge_density", "white_ratio", "ocr_length")


def _cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    left_list = list(left)
    right_list = list(right)
    numerator = sum(a * b for a, b in zip(left_list, right_list))
    left_norm = math.sqrt(sum(a * a for a in left_list))
    right_norm = math.sqrt(sum(b * b for b in right_list))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _text_to_vector(text: str) -> list[float]:
    counts = Counter(_tokenize(text))
    return [float(counts.get(token, 0.0)) for token in TEXT_VOCABULARY]


def _visual_to_vector(features: dict, ocr_text: str) -> list[float]:
    enriched = dict(features)
    enriched["ocr_length"] = float(len(ocr_text.strip()))
    return [float(enriched.get(key, 0.0)) for key in VISUAL_KEYS]


def recommend_similar_medical_objects(classification: dict, ocr_text: str, limit: int = 3) -> list[dict]:
    class_id = classification["class_id"]
    features = classification["features"]

    query_visual_vector = _visual_to_vector(features, ocr_text)
    query_text_vector = _text_to_vector(ocr_text)
    has_text = bool(ocr_text.strip())

    recommendations = []
    for prototype in IMAGE_PROTOTYPES:
        prototype_visual_vector = [prototype["visual_profile"][key] for key in VISUAL_KEYS]
        prototype_text_vector = _text_to_vector(prototype["description"])

        visual_score = _cosine_similarity(query_visual_vector, prototype_visual_vector)
        text_score = _cosine_similarity(query_text_vector, prototype_text_vector) if has_text else 0.0

        class_bonus = 0.08 if prototype["category"] == class_id else 0.0
        if class_id == "xray_image":
            hybrid_score = 0.85 * visual_score + 0.15 * text_score + class_bonus
        else:
            hybrid_score = 0.45 * visual_score + 0.55 * text_score + class_bonus

        matched_keywords = [keyword for keyword in prototype["keywords"] if keyword in ocr_text.lower()]
        rationale = (
            "Совпали ключевые термины: " + ", ".join(matched_keywords)
            if matched_keywords
            else "Рекомендация построена по косинусной близости визуальных и текстовых признаков"
        )

        recommendations.append(
            {
                "id": prototype["id"],
                "label": prototype["label"],
                "category": prototype["category"],
                "score": round(min(hybrid_score, 1.0), 4),
                "visual_score": round(visual_score, 4),
                "text_score": round(text_score, 4),
                "rationale": rationale,
            }
        )

    recommendations.sort(key=lambda item: item["score"], reverse=True)
    return recommendations[:limit]


def recommend_similar_diseases(query: str, graph, limit: int = 3) -> list[dict]:
    disease_profiles = []
    for node, attrs in graph.nodes(data=True):
        if attrs.get("type") != "disease":
            continue

        neighbors = list(graph.neighbors(node))
        profile_text = " ".join([str(node), *map(str, neighbors)])
        disease_profiles.append(
            {
                "name": str(node),
                "profile_text": profile_text,
                "neighbors": neighbors,
            }
        )

    query_vector = _text_to_vector(query)
    recommendations = []
    for profile in disease_profiles:
        disease_vector = _text_to_vector(profile["profile_text"])
        score = _cosine_similarity(query_vector, disease_vector)
        recommendations.append(
            {
                "name": profile["name"],
                "score": round(score, 4),
                "neighbors": profile["neighbors"],
            }
        )

    recommendations.sort(key=lambda item: item["score"], reverse=True)
    return [item for item in recommendations if item["score"] > 0][:limit]
