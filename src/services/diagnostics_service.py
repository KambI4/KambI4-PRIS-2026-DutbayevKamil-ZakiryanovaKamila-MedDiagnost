from __future__ import annotations

from src.data_loader import load_diseases_from_json
from src.logic import ALIASES_BY_CANONICAL, _normalize_text, process_text_message
from src.pipeline import DiagnosisPipeline, DiagnosisRequest
from src.recommendation import recommend_by_fuzzy, recommend_labeled_by_cosine

CLARIFICATION_RULES = {
    "Кашель": {
        "question": "Какой у вас кашель: сухой или с мокротой?",
        "keywords": ["сух", "мокрот", "влажн", "приступ", "ночью", "лающ"],
    },
    "Температура": {
        "question": "Какая у вас температура в градусах и сколько дней она держится?",
        "keywords": ["37", "38", "39", "градус", "день", "ночь", "утро"],
    },
    "Головная боль": {
        "question": "Головная боль сильная, пульсирующая или сопровождается тошнотой?",
        "keywords": ["сильн", "пульс", "тошн", "свет", "шум"],
    },
    "Боль в животе": {
        "question": "Где именно болит живот: сверху, снизу, справа или слева?",
        "keywords": ["сверху", "снизу", "справа", "слева", "после еды", "натощак"],
    },
    "Боль в горле": {
        "question": "Боль в горле усиливается при глотании и есть ли температура?",
        "keywords": ["глот", "температ", "налет", "миндал"],
    },
}


class DiagnosticsService:
    def __init__(self, graph=None):
        self.pipeline = DiagnosisPipeline()
        if graph is not None:
            self.pipeline.graph = graph

        self.graph = self.pipeline.graph
        self.diseases = load_diseases_from_json()
        self.diseases_by_name = {disease.name: disease for disease in self.diseases}

    def _format_urgency(self, urgency: str) -> str:
        labels = {
            "watch": "наблюдение",
            "plan": "плановая консультация",
            "soon": "желательно обратиться в ближайшее время",
            "urgent": "требуется срочная очная оценка",
        }
        return labels.get(urgency, urgency)

    def _build_disease_profiles(self) -> list[tuple[str, str]]:
        profiles = []
        for disease in self.diseases:
            aliases = ALIASES_BY_CANONICAL.get(disease.name, [])
            profile_parts = [disease.name, *aliases, *disease.symptoms, *disease.medicines]
            profiles.append((disease.name, " ".join(profile_parts)))
        return profiles

    def _rank_conditions(self, query: str, top_k: int = 3) -> list[dict]:
        query_norm = _normalize_text(query)
        scores = {disease.name: 0.0 for disease in self.diseases}
        profiles = self._build_disease_profiles()
        matched_symptoms = set(self._extract_known_symptoms(query))

        for disease_name, score in recommend_labeled_by_cosine(query, profiles, top_k=len(profiles)):
            scores[disease_name] = max(scores[disease_name], float(score))

        alias_choices = []
        alias_to_disease = {}
        for disease in self.diseases:
            for alias in [disease.name, *ALIASES_BY_CANONICAL.get(disease.name, [])]:
                alias_choices.append(alias)
                alias_to_disease[alias] = disease.name
                alias_norm = _normalize_text(alias)
                if len(alias_norm) >= 4 and alias_norm in query_norm:
                    scores[disease.name] = max(scores[disease.name], 0.95)

        for alias, score in recommend_by_fuzzy(query, alias_choices, top_k=min(6, len(alias_choices))):
            disease_name = alias_to_disease[alias]
            boosted = float(score) * 0.6
            scores[disease_name] = max(scores[disease_name], boosted)

        for disease in self.diseases:
            overlap = matched_symptoms.intersection(set(disease.symptoms))
            if overlap:
                symptom_boost = 0.35 + 0.2 * (len(overlap) - 1)
                scores[disease.name] = max(scores[disease.name], symptom_boost)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [
            {"name": disease_name, "score": round(score, 4)}
            for disease_name, score in ranked[:top_k]
            if score > 0
        ]

    def _collect_related_medicines(self, top_conditions: list[dict]) -> list[str]:
        medicines = []
        for condition in top_conditions[:2]:
            disease = self.diseases_by_name.get(condition["name"])
            if not disease:
                continue
            for medicine in disease.medicines:
                if medicine not in medicines:
                    medicines.append(medicine)
        return medicines

    def _extract_known_symptoms(self, query: str) -> list[str]:
        query_norm = _normalize_text(query)
        found = []
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get("type") != "symptom":
                continue

            aliases = [str(node), *ALIASES_BY_CANONICAL.get(str(node), [])]
            for alias in aliases:
                alias_norm = _normalize_text(alias)
                if len(alias_norm) >= 3 and alias_norm in query_norm:
                    found.append(str(node))
                    break

        return sorted(set(found))

    def _build_lead_profile(self, top_conditions: list[dict]) -> dict | None:
        if not top_conditions:
            return None
        lead = self.diseases_by_name.get(top_conditions[0]["name"])
        if lead is None:
            return None
        return {
            "name": lead.name,
            "severity": lead.severity,
            "urgency": lead.urgency,
            "urgency_label": self._format_urgency(lead.urgency),
            "advice": lead.advice,
            "when_to_seek_help": lead.when_to_seek_help,
        }

    def _build_answer(
        self,
        query: str,
        top_conditions: list[dict],
        recognized_symptoms: list[str],
        related_medicines: list[str],
        rules_check: str,
        confidence: str,
        lead_profile: dict | None,
    ) -> str:
        if not top_conditions and not recognized_symptoms:
            return (
                "Я не нашел уверенного совпадения в учебной базе знаний. "
                "Попробуйте описать симптомы подробнее, например: температура, кашель, слабость."
            )

        lines = []
        if top_conditions:
            lead = top_conditions[0]
            lines.append(
                f"Предварительный разбор: по описанию это больше всего похоже на '{lead['name']}'."
            )
            ranked = ", ".join([f"{item['name']} ({item['score']:.2f})" for item in top_conditions])
            lines.append(f"Наиболее близкие состояния из базы знаний: {ranked}.")
        else:
            lines.append("Я распознал симптомы, но не смог уверенно ранжировать заболевания из базы.")

        if recognized_symptoms:
            lines.append(f"Распознаны симптомы: {', '.join(recognized_symptoms)}.")

        if related_medicines:
            lines.append(
                "Связанные препараты из учебной базы знаний: "
                + ", ".join(related_medicines)
                + "."
            )

        if lead_profile is not None:
            lines.append(
                f"Оценка тяжести состояния в учебной базе: {lead_profile['severity']}. "
                f"Рекомендованный уровень реакции: {lead_profile['urgency_label']}."
            )
            if lead_profile["advice"]:
                lines.append("Базовые рекомендации: " + "; ".join(lead_profile["advice"]) + ".")
            if lead_profile["when_to_seek_help"]:
                lines.append(
                    "Когда стоит обратиться за помощью: "
                    + "; ".join(lead_profile["when_to_seek_help"])
                    + "."
                )

        lines.append(f"Проверка правил безопасности: {rules_check}.")
        lines.append(f"Уровень уверенности системы: {confidence}.")
        lines.append("Важно: это учебный ИИ-ассистент и он не заменяет консультацию врача.")
        return "\n".join(lines)

    def _detect_follow_up(self, query: str, recognized_symptoms: list[str], context_query: str | None) -> dict | None:
        if context_query:
            return None
        if len(recognized_symptoms) != 1:
            return None

        symptom = recognized_symptoms[0]
        rule = CLARIFICATION_RULES.get(symptom)
        if rule is None:
            return None

        query_norm = _normalize_text(query)
        if any(keyword in query_norm for keyword in rule["keywords"]):
            return None

        token_count = len(query_norm.split())
        if token_count > 5:
            return None

        return {
            "symptom": symptom,
            "question": rule["question"],
        }

    def analyze_text(self, text: str, context_query: str | None = None) -> dict:
        effective_text = text.strip()
        if context_query:
            effective_text = f"{context_query.strip()} {text.strip()}".strip()

        diagnosis = self.pipeline.diagnose_from_request(
            DiagnosisRequest(patient_symptoms=[effective_text], is_registered=True)
        )
        top_conditions = self._rank_conditions(effective_text)
        recognized_symptoms = sorted(
            set(diagnosis.related_symptoms + self._extract_known_symptoms(effective_text))
        )
        related_medicines = self._collect_related_medicines(top_conditions)
        lead_profile = self._build_lead_profile(top_conditions)
        follow_up = self._detect_follow_up(effective_text, recognized_symptoms, context_query)
        answer = self._build_answer(
            query=effective_text,
            top_conditions=top_conditions,
            recognized_symptoms=recognized_symptoms,
            related_medicines=related_medicines,
            rules_check=diagnosis.rules_check,
            confidence=diagnosis.confidence.value,
            lead_profile=lead_profile,
        )
        if follow_up is not None:
            answer = (
                f"Я уже вижу основной симптом: {follow_up['symptom']}.\n"
                f"Чтобы точнее оценить состояние, уточню: {follow_up['question']}"
            )

        return {
            "query": text,
            "effective_query": effective_text,
            "answer": answer,
            "confidence": diagnosis.confidence.value,
            "rules_check": diagnosis.rules_check,
            "recognized_symptoms": recognized_symptoms,
            "top_conditions": top_conditions,
            "related_medicines": related_medicines,
            "lead_profile": lead_profile,
            "needs_clarification": follow_up is not None,
            "follow_up_question": follow_up["question"] if follow_up is not None else "",
            "clarification_symptom": follow_up["symptom"] if follow_up is not None else "",
            "entities_found": diagnosis.entities_found,
            "reasoning": diagnosis.reasoning,
        }

    def analyze_text_legacy(self, text: str) -> dict:
        answer = process_text_message(text, self.graph)
        return {"query": text, "answer": answer}

    def analyze_image(self, image_bytes: bytes) -> dict:
        image_result, diagnosis_result = self.pipeline.diagnose_from_image(image_bytes)
        answer = "Текст на изображении не найден, поэтому диагноз не сформирован."
        if diagnosis_result is not None:
            ranked = self._rank_conditions(image_result.ocr_text)
            answer = self._build_answer(
                query=image_result.ocr_text,
                top_conditions=ranked,
                recognized_symptoms=diagnosis_result.related_symptoms,
                related_medicines=self._collect_related_medicines(ranked),
                rules_check=diagnosis_result.rules_check,
                confidence=diagnosis_result.confidence.value,
                lead_profile=self._build_lead_profile(ranked),
            )

        return {
            "classification": image_result.classification,
            "ocr_text": image_result.ocr_text,
            "recommendations": image_result.recommendations,
            "diagnosis_suggestions": image_result.diagnosis_suggestions,
            "answer": answer,
        }

    def recommend_diseases(self, query: str, top_k: int = 3) -> dict:
        ranked = self._rank_conditions(query, top_k=top_k)
        fuzzy = recommend_by_fuzzy(query, [disease.name for disease in self.diseases], top_k=top_k)
        return {
            "query": query,
            "cosine": [{"item": item["name"], "score": item["score"]} for item in ranked],
            "fuzzy": [{"item": item, "score": score} for item, score in fuzzy],
        }
