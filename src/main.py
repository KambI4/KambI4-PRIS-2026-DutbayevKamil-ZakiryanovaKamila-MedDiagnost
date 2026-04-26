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
from src.vision import image_bytes_to_pil


st.set_page_config(page_title="MedDiagnost", page_icon="MD", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(41, 98, 255, 0.12), transparent 24%),
            radial-gradient(circle at top right, rgba(0, 188, 212, 0.10), transparent 22%),
            linear-gradient(180deg, #0b1220 0%, #111827 100%);
    }
    .hero {
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 24px;
        padding: 1.4rem 1.6rem;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(30, 41, 59, 0.88));
        box-shadow: 0 18px 50px rgba(15, 23, 42, 0.28);
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.4rem;
        letter-spacing: -0.03em;
    }
    .hero p {
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
        color: #cbd5e1;
    }
    .info-card {
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        background: rgba(15, 23, 42, 0.68);
        min-height: 122px;
    }
    .info-card h3 {
        margin: 0;
        font-size: 0.95rem;
        color: #93c5fd;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .info-card p {
        margin: 0.6rem 0 0 0;
        color: #e5e7eb;
        line-height: 1.45;
    }
    .assistant-box {
        border-left: 4px solid #38bdf8;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        background: rgba(15, 23, 42, 0.72);
        margin: 0.75rem 0 1rem 0;
    }
    .assistant-box strong {
        color: #7dd3fc;
    }
    .badge-row {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin: 0.5rem 0 0.9rem 0;
    }
    .badge {
        display: inline-block;
        padding: 0.42rem 0.8rem;
        border-radius: 999px;
        font-size: 0.9rem;
        font-weight: 600;
        border: 1px solid rgba(255,255,255,0.12);
    }
    .severity-low, .urgency-watch {
        background: rgba(74, 222, 128, 0.12);
        color: #86efac;
    }
    .severity-medium, .urgency-plan {
        background: rgba(250, 204, 21, 0.12);
        color: #fde68a;
    }
    .severity-high, .urgency-soon {
        background: rgba(251, 146, 60, 0.12);
        color: #fdba74;
    }
    .urgency-urgent {
        background: rgba(248, 113, 113, 0.14);
        color: #fca5a5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


if "service" not in st.session_state:
    st.session_state.service = DiagnosticsService()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None
if "pending_clarification" not in st.session_state:
    st.session_state.pending_clarification = None

service = st.session_state.service
graph = service.graph

disease_nodes = sorted([n for n, attrs in graph.nodes(data=True) if attrs.get("type") == "disease"])
symptom_nodes = sorted([n for n, attrs in graph.nodes(data=True) if attrs.get("type") == "symptom"])
medicine_nodes = sorted([n for n, attrs in graph.nodes(data=True) if attrs.get("type") == "medicine"])

st.markdown(
    """
    <div class="hero">
        <h1>MedDiagnost</h1>
        <p>
            Гибридный учебный AI-ассистент для анализа симптомов, рекомендаций из базы знаний
            и обработки медицинских изображений.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown(
        """
        <div class="info-card">
            <h3>Rule-Based Core</h3>
            <p>Критические правила безопасности, граф знаний и учебная база симптомов и заболеваний.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_b:
    st.markdown(
        """
        <div class="info-card">
            <h3>NLP + Similarity</h3>
            <p>spaCy, извлечение сущностей, ранжирование похожих состояний и объяснение результата.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_c:
    st.markdown(
        """
        <div class="info-card">
            <h3>CV / OCR</h3>
            <p>Классификация медицинских изображений, OCR и извлечение структурированной информации.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

tab_chat, tab_knowledge, tab_cv = st.tabs(
    ["AI-консультант", "База знаний", "CV/OCR"]
)

with tab_chat:
    left, right = st.columns([1.35, 0.85], gap="large")

    with left:
        st.subheader("Консультация по симптомам")
        st.caption("Пример: температура 38.5, кашель, слабость")

        with st.form("diagnostic_form"):
            user_text = st.text_area(
                "Опишите состояние пациента",
                placeholder="Например: высокая температура, кашель и боль в горле уже 3 дня",
                height=140,
            )
            submit = st.form_submit_button("Запустить анализ")

        if st.session_state.pending_clarification:
            st.warning(
                "Ассистент ждёт уточнение: "
                + st.session_state.pending_clarification["question"]
            )

        if submit and user_text.strip():
            context_query = None
            if st.session_state.pending_clarification:
                context_query = st.session_state.pending_clarification["base_query"]

            analysis = service.analyze_text(user_text, context_query=context_query)
            st.session_state.last_analysis = analysis
            st.session_state.messages.append({"role": "user", "content": user_text})
            st.session_state.messages.append(
                {"role": "assistant", "content": analysis["answer"], "analysis": analysis}
            )
            if analysis["needs_clarification"]:
                st.session_state.pending_clarification = {
                    "base_query": analysis["effective_query"],
                    "question": analysis["follow_up_question"],
                    "symptom": analysis["clarification_symptom"],
                }
            else:
                st.session_state.pending_clarification = None

        for message in st.session_state.messages[-8:]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if st.session_state.last_analysis:
            analysis = st.session_state.last_analysis
            st.markdown(
                f"""
                <div class="assistant-box">
                    <strong>Краткий вывод ассистента</strong><br>
                    {analysis["answer"].replace(chr(10), "<br>")}
                </div>
                """,
                unsafe_allow_html=True,
            )

            top_conditions_df = pd.DataFrame(analysis["top_conditions"])
            if not top_conditions_df.empty:
                pretty_df = top_conditions_df.rename(
                    columns={"name": "Состояние", "score": "Оценка близости"}
                )
                st.markdown("**Наиболее вероятные состояния**")
                st.dataframe(pretty_df, use_container_width=True, hide_index=True)

            if analysis["related_medicines"]:
                st.markdown("**Связанные препараты из учебной базы**")
                st.write(", ".join(analysis["related_medicines"]))

            with st.expander("Показать reasoning pipeline"):
                st.code(analysis["reasoning"])

    with right:
        st.subheader("Сводка анализа")
        analysis = st.session_state.last_analysis
        if analysis:
            top_name = analysis["top_conditions"][0]["name"] if analysis["top_conditions"] else "нет данных"
            st.metric("Главная гипотеза", top_name)
            st.metric("Уверенность", analysis["confidence"])
            st.metric("Распознано симптомов", len(analysis["recognized_symptoms"]))

            if analysis.get("lead_profile"):
                st.markdown(
                    f"""
                    <div class="badge-row">
                        <span class="badge severity-{analysis['lead_profile']['severity']}">
                            Тяжесть: {analysis['lead_profile']['severity']}
                        </span>
                        <span class="badge urgency-{analysis['lead_profile']['urgency']}">
                            Реакция: {analysis['lead_profile']['urgency_label']}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("**Проверка правил безопасности**")
            st.info(analysis["rules_check"])

            if analysis["recognized_symptoms"]:
                st.markdown("**Что поняла система**")
                for symptom in analysis["recognized_symptoms"]:
                    st.write(f"- {symptom}")

            if analysis.get("lead_profile"):
                st.markdown("**Базовые рекомендации**")
                for item in analysis["lead_profile"]["advice"]:
                    st.write(f"- {item}")

                st.markdown("**Когда стоит обратиться за помощью**")
                for item in analysis["lead_profile"]["when_to_seek_help"]:
                    st.write(f"- {item}")
        else:
            st.info("После первого запроса здесь появятся ключевые результаты анализа.")

with tab_knowledge:
    st.subheader("Структура базы знаний")
    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Болезни", len(disease_nodes))
    metric_2.metric("Симптомы", len(symptom_nodes))
    metric_3.metric("Препараты", len(medicine_nodes))

    explorer_col, relation_col = st.columns([0.8, 1.2], gap="large")
    with explorer_col:
        all_nodes = sorted(graph.nodes())
        selected_node = st.selectbox("Выберите сущность графа", all_nodes)
        neighbors = sorted(map(str, graph.neighbors(selected_node)))
        st.markdown("**Связанные объекты**")
        if neighbors:
            for neighbor in neighbors:
                st.write(f"- {neighbor}")
        else:
            st.write("Связей пока нет.")

    with relation_col:
        st.markdown("**Связи болезнь → симптомы / препараты**")
        rows = []
        for disease in disease_nodes:
            for neighbor in sorted(map(str, graph.neighbors(disease))):
                rows.append({"Болезнь": disease, "Связанный объект": neighbor})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    disease_rows = []
    for disease in service.diseases:
        disease_rows.append(
            {
                "Болезнь": disease.name,
                "Тяжесть": disease.severity,
                "Срочность": service._format_urgency(disease.urgency),
                "Симптомы": ", ".join(disease.symptoms),
                "Препараты": ", ".join(disease.medicines),
            }
        )
    st.markdown("**Паспорт заболеваний в базе знаний**")
    st.dataframe(pd.DataFrame(disease_rows), use_container_width=True, hide_index=True)

    with st.expander("Показать справочники CSV"):
        symptoms_df = pd.DataFrame(load_symptoms_catalog())
        medicines_df = pd.DataFrame(load_medicines_catalog())
        left_df, right_df = st.columns(2)
        with left_df:
            st.markdown("**Симптомы**")
            st.dataframe(symptoms_df, use_container_width=True, hide_index=True)
        with right_df:
            st.markdown("**Препараты**")
            st.dataframe(medicines_df, use_container_width=True, hide_index=True)

with tab_cv:
    st.subheader("Обработка медицинского изображения")
    st.caption("Поддерживаются рецепты, медицинские документы и рентгеновские изображения.")
    uploaded_file = st.file_uploader(
        "Загрузите изображение",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
    )

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        st.image(image_bytes_to_pil(image_bytes), caption=uploaded_file.name, use_container_width=True)

        try:
            image_result = service.analyze_image(image_bytes)
            cls = image_result["classification"]

            info_1, info_2 = st.columns(2)
            info_1.metric("Тип изображения", cls["class_label"])
            info_2.metric("Уверенность классификации", f"{cls['confidence']:.2f}")

            st.markdown("**AI-вывод по изображению**")
            st.success(image_result["answer"])

            if image_result["ocr_text"].strip():
                st.text_area("OCR текст", image_result["ocr_text"], height=180)

            if image_result["recommendations"]:
                st.markdown("**Похожие медицинские объекты**")
                st.dataframe(pd.DataFrame(image_result["recommendations"]), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Ошибка анализа изображения: {exc}")
