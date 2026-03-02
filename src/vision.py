from __future__ import annotations

from io import BytesIO
import re
from typing import Any

import numpy as np
from PIL import Image

try:
    import cv2
except Exception:
    cv2 = None

try:
    import easyocr
except Exception:
    easyocr = None

_READER = None


CLASS_LABELS = {
    "prescription_document": "Медицинский рецепт",
    "diagnostic_report": "Диагностическое заключение",
    "medical_document": "Медицинский документ",
    "xray_image": "Рентген",
}

MEDICATION_KEYWORDS = (
    "парацетамол",
    "ибупрофен",
    "амоксициллин",
    "азитромицин",
    "аспирин",
    "омепразол",
    "нурофен",
    "анальгин",
    "paracetamol",
    "ibuprofen",
    "amoxicillin",
    "aspirin",
    "omeprazole",
)

DIAGNOSIS_TRIGGERS = (
    "диагноз",
    "заключение",
    "клинический диагноз",
    "основной диагноз",
    "сопутствующий диагноз",
    "diagnosis",
    "impression",
)


def _get_reader():
    global _READER
    if easyocr is None:
        return None
    if _READER is None:
        _READER = easyocr.Reader(["ru", "en"], gpu=False)
    return _READER


def _decode_image(image_bytes: bytes):
    if cv2 is None:
        raise RuntimeError("OpenCV не установлен. Добавьте opencv-python-headless в зависимости.")

    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Не удалось декодировать изображение")
    return image


def _preprocess_for_ocr(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    return thr


def _extract_ocr(image_bgr):
    reader = _get_reader()
    if reader is None:
        return {
            "text": "",
            "lines": [],
            "available": False,
            "message": "EasyOCR не установлен. Добавьте easyocr в зависимости.",
        }

    prep = _preprocess_for_ocr(image_bgr)
    results = reader.readtext(prep)

    lines = []
    for item in results:
        if len(item) != 3:
            continue
        _, text, confidence = item
        cleaned = str(text).strip()
        if not cleaned:
            continue
        lines.append({"text": cleaned, "confidence": float(confidence)})

    full_text = "\n".join(line["text"] for line in lines)
    return {
        "text": full_text,
        "lines": lines,
        "available": True,
        "message": "",
    }


def _classify_image(image_bgr, ocr_text: str):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    mean_intensity = float(np.mean(gray))
    std_intensity = float(np.std(gray))

    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)

    white_ratio = float(np.count_nonzero(gray > 210)) / float(gray.size)
    ocr_norm = ocr_text.lower().replace("ё", "е")
    ocr_len = len(ocr_norm.strip())

    rx_keywords = (
        "рецепт",
        "назначение",
        "дозировка",
        "таб",
        "капс",
        "принимать",
        "rx",
        "prescription",
    )
    diagnosis_keywords = (
        "диагноз",
        "заключение",
        "анамнез",
        "жалобы",
        "осмотр",
        "пациент",
        "diagnosis",
        "impression",
    )

    cls = "medical_document"
    confidence = 0.6

    if mean_intensity < 110 and std_intensity > 35 and white_ratio < 0.22:
        cls = "xray_image"
        confidence = 0.72
    elif any(keyword in ocr_norm for keyword in rx_keywords):
        cls = "prescription_document"
        confidence = 0.83 if ocr_len > 25 else 0.74
    elif any(keyword in ocr_norm for keyword in diagnosis_keywords):
        cls = "diagnostic_report"
        confidence = 0.81 if ocr_len > 40 else 0.73
    elif (ocr_len > 50 and white_ratio > 0.35) or (white_ratio > 0.55 and edge_density < 0.18):
        cls = "medical_document"
        confidence = 0.75

    return {
        "class_id": cls,
        "class_label": CLASS_LABELS[cls],
        "confidence": confidence,
        "features": {
            "width": int(w),
            "height": int(h),
            "mean_intensity": round(mean_intensity, 2),
            "std_intensity": round(std_intensity, 2),
            "edge_density": round(edge_density, 4),
            "white_ratio": round(white_ratio, 4),
        },
    }


def _extract_medical_entities_from_ocr(ocr_text: str):
    text = ocr_text.replace("\r", "\n")
    text_norm = text.lower().replace("ё", "е")
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    diagnosis_candidates = []
    prescriptions = []
    icd_codes = sorted(set(re.findall(r"\b[A-ZА-Я]\d{2}(?:\.\d)?\b", text.upper())))

    dosage_pattern = re.compile(
        r"\b\d+(?:[.,]\d+)?\s*(?:мг|mg|мл|ml|г|гр|таб|табл|капс|капсул|drops?)\b",
        re.IGNORECASE,
    )
    frequency_pattern = re.compile(
        r"(?:\b\d+\s*раз[а]?\s*в\s*день\b|\bежедневно\b|\bутром\b|\bвечером\b)",
        re.IGNORECASE,
    )

    for line in lines:
        line_norm = line.lower().replace("ё", "е")

        if any(trigger in line_norm for trigger in DIAGNOSIS_TRIGGERS):
            diagnosis_candidates.append(line)
            continue

        has_med_keyword = any(keyword in line_norm for keyword in MEDICATION_KEYWORDS)
        has_dosage = bool(dosage_pattern.search(line_norm))
        has_frequency = bool(frequency_pattern.search(line_norm))

        if has_med_keyword or has_dosage or has_frequency:
            medication_match = None
            for keyword in MEDICATION_KEYWORDS:
                if keyword in line_norm:
                    medication_match = keyword
                    break

            dosage_match = dosage_pattern.search(line)
            frequency_match = frequency_pattern.search(line)

            prescriptions.append(
                {
                    "line": line,
                    "medicine_hint": medication_match,
                    "dosage": dosage_match.group(0) if dosage_match else "",
                    "frequency": frequency_match.group(0) if frequency_match else "",
                }
            )

    diagnosis_candidates = list(dict.fromkeys(diagnosis_candidates))[:5]
    prescriptions = prescriptions[:10]

    return {
        "diagnosis_candidates": diagnosis_candidates,
        "icd_codes": icd_codes,
        "prescriptions": prescriptions,
    }


def analyze_image_bytes(image_bytes: bytes) -> dict[str, Any]:
    image_bgr = _decode_image(image_bytes)
    ocr = _extract_ocr(image_bgr)
    classification = _classify_image(image_bgr, ocr["text"])
    medical_extraction = _extract_medical_entities_from_ocr(ocr["text"]) if ocr["available"] else {
        "diagnosis_candidates": [],
        "icd_codes": [],
        "prescriptions": [],
    }

    return {
        "classification": classification,
        "ocr": ocr,
        "medical_extraction": medical_extraction,
    }


def image_bytes_to_pil(image_bytes: bytes) -> Image.Image:
    return Image.open(BytesIO(image_bytes)).convert("RGB")
