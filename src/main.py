from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_medicines_catalog, load_symptoms_catalog
from src.services.diagnostics_service import DiagnosticsService
from src.vision import analyze_image_bytes, image_bytes_to_pil

st.set_page_config(page_title="MedDiagnost Chat", page_icon="🩺", layout="wide")
st.title("MedDiagnost: медицинский ассистент")
st.caption("Учебный прототип. Не является медицинским заключением.")

if "service" not in st.session_state:
    st.session_state.service = DiagnosticsService()
if "messages" not in st.session_state:
    st.session_state.messages = []

service = st.session_state.service
graph = service.graph

disease_nodes = sorted([n for n, attrs in graph.nodes(data=True) if attrs.get("type") == "disease"])
symptom_nodes = sorted([n for n, attrs in graph.nodes(data=True) if attrs.get("type") == "symptom"])
medicine_nodes = sorted([n for n, attrs in graph.nodes(data=True) if attrs.get("type") == "medicine"])

tab_chat, tab_dashboard, tab_cv = st.tabs(["Чат", "Справочники и графики", "CV/OCR"])

with tab_chat:
    with st.form("diagnostic_form"):
        user_text = st.text_area("Опишите симптомы", placeholder="Например: кашель, температура 38.5, слабость")
        submit = st.form_submit_button("Проанализировать")

    if submit and user_text.strip():
        analysis = service.analyze_text(user_text)
        rec = service.recommend_diseases(user_text)

        st.session_state.messages.append({"role": "user", "content": user_text})
        st.session_state.messages.append({"role": "assistant", "content": analysis["answer"]})

        st.subheader("Рекомендации")
        rec_df = pd.DataFrame(rec["cosine"])
        if not rec_df.empty:
            st.dataframe(rec_df, use_container_width=True)

    for message in st.session_state.messages[-10:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

with tab_dashboard:
    st.subheader("Предметные справочники")
    symptoms_df = pd.DataFrame(load_symptoms_catalog())
    medicines_df = pd.DataFrame(load_medicines_catalog())

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Симптомы (CSV)**")
        st.dataframe(symptoms_df, use_container_width=True)
    with col2:
        st.markdown("**Препараты (CSV)**")
        st.dataframe(medicines_df, use_container_width=True)

    st.subheader("Визуализация графа знаний")
    stats = pd.DataFrame(
        [
            {"type": "disease", "count": len(disease_nodes)},
            {"type": "symptom", "count": len(symptom_nodes)},
            {"type": "medicine", "count": len(medicine_nodes)},
        ]
    )
    st.bar_chart(stats.set_index("type"))

with tab_cv:
    st.subheader("Компьютерное зрение: OCR и классификация")
    uploaded_file = st.file_uploader(
        "Загрузите медицинское изображение",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
    )

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        st.image(image_bytes_to_pil(image_bytes), caption=uploaded_file.name, use_container_width=True)

        try:
            result = analyze_image_bytes(image_bytes)
            cls = result["classification"]
            ocr = result["ocr"]
            medical = result["medical_extraction"]

            st.write(f"Тип изображения: **{cls['class_label']}**")
            st.write(f"Уверенность: **{cls['confidence']:.2f}**")
            st.json(cls["features"])

            if ocr["available"] and ocr["text"].strip():
                st.text_area("OCR текст", ocr["text"], height=180)
                st.write("Выделенные МКБ-коды:", ", ".join(medical["icd_codes"]) or "не найдены")
            elif not ocr["available"]:
                st.warning(ocr["message"])
            else:
                st.info("Текст не обнаружен")
        except Exception as exc:
            st.error(f"Ошибка анализа изображения: {exc}")
