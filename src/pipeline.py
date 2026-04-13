"""
Unified Medical Diagnosis Pipeline
Интегрирует символьный (правила) и нейросетевой подходы
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from enum import Enum

from src.knowledge_graph import load_graph
from src.logic import (
    process_text_message,
    check_rules,
    _extract_named_entities,
    _extract_dates,
    _extract_medical_entities,
)
from src.vision import analyze_image_bytes
from src.models import Disease


class DiagnosisConfidence(str, Enum):
    """Уровень уверенности диагноза"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class DiagnosisResult:
    """Результат диагностики"""
    primary_candidates: List[Dict[str, Any]]  # Основные кандидаты на диагноз
    related_symptoms: List[str]  # Связанные симптомы
    recommended_medicines: List[str]  # Рекомендуемые лекарства
    rules_check: str  # Результат проверки правил
    confidence: DiagnosisConfidence  # Уровень уверенности
    reasoning: str  # Объяснение процесса
    entities_found: List[tuple]  # Найденные сущности (NER)


@dataclass
class ImageAnalysisResult:
    """Результат анализа изображения"""
    classification: Dict[str, Any]  # Классификация изображения
    ocr_text: str  # Извлеченный текст
    recommendations: List[Dict[str, Any]]  # Медицинские рекомендации
    diagnosis_suggestions: List[str]  # Предложенные диагнозы


@dataclass
class DiagnosisRequest:
    """Запрос на диагностику"""
    patient_symptoms: List[str]
    medical_history: Optional[str] = None
    temperature: Optional[float] = None
    is_registered: bool = True


class DiagnosisPipeline:
    """Unified pipeline для диагностики"""

    def __init__(self):
        self.graph = load_graph()
        self.rules_cache = {}

    def diagnose_from_text(self, text: str) -> DiagnosisResult:
        """
        Диагностика на основе текстового описания симптомов
        Комбинирует: правила + NLP + граф знаний
        """
        # Шаг 1: Обработка текста
        reasoning_parts = ["=== Анализ текста ==="]
        reasoning_parts.append(f"Входной текст: {text}")

        text_response = process_text_message(text, self.graph)
        reasoning_parts.append(f"NLP обработка: {text_response}")

        # Шаг 2: Извлечение сущностей
        ner_entities = _extract_named_entities(text)
        dates = _extract_dates(text)
        if dates:
            ner_entities.extend((dt, "DATE") for dt in dates)

        medical_entities = _extract_medical_entities(text, self.graph)
        reasoning_parts.append(f"Найденные сущности: {ner_entities}")
        reasoning_parts.append(f"Медицинские объекты: {medical_entities}")

        # Шаг 3: Сбор информации о болезнях
        primary_candidates = []
        all_symptoms = set()
        all_medicines = set()

        for node in medical_entities:
            neighbors = list(self.graph.neighbors(node))
            node_type = self.graph.nodes[node].get("type", "unknown")

            if node_type == "disease":
                primary_candidates.append({
                    "name": node,
                    "score": 1.0,  # Точное совпадение
                    "related": neighbors,
                })
            elif node_type == "symptom":
                all_symptoms.add(node)
            elif node_type == "medicine":
                all_medicines.add(node)

        if not primary_candidates and medical_entities:
            for entity in medical_entities:
                neighbors = list(self.graph.neighbors(entity))
                primary_candidates.append({
                    "name": entity,
                    "score": 0.8,
                    "related": neighbors,
                })

        # Шаг 4: Определение уровня уверенности
        confidence = self._calculate_confidence(primary_candidates, len(medical_entities))
        reasoning_parts.append(f"Уровень уверенности: {confidence}")

        # Шаг 5: Формирование результата
        result = DiagnosisResult(
            primary_candidates=primary_candidates,
            related_symptoms=sorted(list(all_symptoms)),
            recommended_medicines=sorted(list(all_medicines)),
            rules_check="",  # Будет обновлено ниже
            confidence=confidence,
            entities_found=ner_entities,
            reasoning="\n".join(reasoning_parts),
        )

        # Шаг 6: Проверка правил безопасности
        result.rules_check = self._check_safety_rules(
            symptoms=result.related_symptoms,
            primary_disease=primary_candidates[0]["name"] if primary_candidates else None,
        )

        return result

    def diagnose_from_image(self, image_bytes: bytes) -> tuple[ImageAnalysisResult, Optional[DiagnosisResult]]:
        """
        Диагностика на основе медицинского изображения
        Комбинирует: зрение + OCR + правила
        """
        # Анализ изображения
        analysis = analyze_image_bytes(image_bytes)

        # Извлечение текста из OCR
        ocr_text = analysis["ocr"]["text"]
        diagnosis_suggestions = []

        # Если есть текст, запускаем диагностику
        diagnosis_result = None
        if ocr_text.strip():
            diagnosis_result = self.diagnose_from_text(ocr_text)
            diagnosis_suggestions = [
                cand["name"] for cand in diagnosis_result.primary_candidates
            ]

        image_result = ImageAnalysisResult(
            classification=analysis["classification"],
            ocr_text=ocr_text,
            recommendations=analysis["recommendations"],
            diagnosis_suggestions=diagnosis_suggestions,
        )

        return image_result, diagnosis_result

    def diagnose_from_request(self, request: DiagnosisRequest) -> DiagnosisResult:
        """
        Полная диагностика на основе структурированного запроса
        Использует все доступные данные пациента
        """
        # Конвертируем в текст
        text_parts = request.patient_symptoms.copy()
        if request.medical_history:
            text_parts.append(request.medical_history)
        if request.temperature:
            text_parts.append(f"температура {request.temperature}")

        combined_text = " ".join(text_parts)

        # Запускаем основную диагностику
        result = self.diagnose_from_text(combined_text)

        # Проверяем правила пациента
        safety_check = check_rules({
            "symptoms": result.related_symptoms,
            "temperature": request.temperature,
            "is_registered": request.is_registered,
        })
        result.rules_check = safety_check

        return result

    def _calculate_confidence(self, candidates: List[Dict], entities_count: int) -> DiagnosisConfidence:
        """Вычисляет уровень уверенности на основе числа совпадений"""
        if not candidates:
            return DiagnosisConfidence.LOW

        avg_score = sum(c.get("score", 0.5) for c in candidates) / len(candidates)
        entities_factor = min(entities_count / 3.0, 1.0)  # Нормализуем к 0-1

        combined_score = (avg_score + entities_factor) / 2.0

        if combined_score >= 0.85:
            return DiagnosisConfidence.HIGH
        elif combined_score >= 0.5:
            return DiagnosisConfidence.MEDIUM
        else:
            return DiagnosisConfidence.LOW

    def _check_safety_rules(self, symptoms: List[str] = None, primary_disease: str = None) -> str:
        """Проверяет критические правила безопасности"""
        if not symptoms:
            symptoms = []

        # Критические симптомы
        critical_symptoms = {"потеря сознания", "сильная боль", "кровотечение"}
        if any(s.lower() in critical_symptoms for s in symptoms):
            return "⚠️ КРИТИЧНО: Обнаружены признаки серьезного состояния. Требуется немедленная консультация врача!"

        return "✓ Пациент прошел первичный осмотр. Рекомендуется консультация специалиста."

    def get_graph_info(self) -> Dict[str, Any]:
        """Получить информацию о графе знаний"""
        diseases = [n for n, attrs in self.graph.nodes(data=True) if attrs.get("type") == "disease"]
        symptoms = [n for n, attrs in self.graph.nodes(data=True) if attrs.get("type") == "symptom"]
        medicines = [n for n, attrs in self.graph.nodes(data=True) if attrs.get("type") == "medicine"]

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "diseases": sorted(diseases),
            "symptoms": sorted(symptoms),
            "medicines": sorted(medicines),
        }
