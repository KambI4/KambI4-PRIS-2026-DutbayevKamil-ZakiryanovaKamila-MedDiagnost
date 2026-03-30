import streamlit as st

from knowledge_graph import load_graph
from pipeline import DiagnosisPipeline
from vision import image_bytes_to_pil

st.set_page_config(page_title="MedDiagnost Chat", page_icon="🩺", layout="wide")
st.title("Медицинский чат-ассистент")
st.caption(
    "Поддерживаются русские и английские названия (например: 'кашель', 'cough', 'flu')."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "graph" not in st.session_state:
    st.session_state.graph = load_graph()

if "pipeline" not in st.session_state:
    st.session_state.pipeline = DiagnosisPipeline()

graph = st.session_state.graph
pipeline = st.session_state.pipeline

disease_nodes = sorted([n for n, attrs in graph.nodes(data=True) if attrs.get("type") == "disease"])
symptom_nodes = sorted([n for n, attrs in graph.nodes(data=True) if attrs.get("type") == "symptom"])
medicine_nodes = sorted([n for n, attrs in graph.nodes(data=True) if attrs.get("type") == "medicine"])

with st.expander("Подсказки по вводу"):
    st.markdown("Примеры: `кашель`, `cough`, `грипп`, `flu`, `covid`, `болит голова`")
    st.markdown(f"Болезни: {', '.join(disease_nodes)}")
    st.markdown(f"Симптомы: {', '.join(symptom_nodes)}")
    st.markdown(f"Лекарства: {', '.join(medicine_nodes)}")

with st.expander("Компьютерное зрение: OCR и классификация изображений", expanded=True):
    uploaded_file = st.file_uploader(
        "Загрузите медицинское изображение (рецепт, диагностическое заключение или рентген)",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
    )

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        st.image(image_bytes_to_pil(image_bytes), caption=uploaded_file.name, use_container_width=True)

        try:
            image_result, diagnosis_result = pipeline.diagnose_from_image(image_bytes)
            
            cls = image_result.classification
            ocr_text = image_result.ocr_text
            recommendations = image_result.recommendations

            st.subheader("Результат анализа")
            st.write(f"Тип изображения: **{cls['class_label']}**")
            st.write(f"Уверенность: **{cls['confidence']:.2f}**")
            st.json(cls["features"])

            st.subheader("Гибридные рекомендации")
            st.caption("Поиск похожих медицинских объектов по cosine similarity: визуальные признаки + OCR-текст.")
            for item in recommendations:
                st.markdown(
                    f"**{item['label']}**  \n"
                    f"Итоговая близость: `{item['score']:.2f}` | "
                    f"Визуальная: `{item['visual_score']:.2f}` | "
                    f"Текстовая: `{item['text_score']:.2f}`"
                )
                st.write(item["rationale"])

            st.subheader("OCR")
            if ocr_text.strip():
                st.text_area("Извлеченный текст", ocr_text, height=180)

                if st.button("Отправить OCR-текст в мед. анализ", use_container_width=True):
                    if diagnosis_result:
                        response_text = f"Диагноз: {', '.join([c['name'] for c in diagnosis_result.primary_candidates])}\n"
                        response_text += f"Симптомы: {', '.join(diagnosis_result.related_symptoms)}\n"
                        response_text += f"Лекарства: {', '.join(diagnosis_result.recommended_medicines)}\n"
                        response_text += f"Безопасность: {diagnosis_result.rules_check}"
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response_text,
                        })
                        st.success("OCR-текст обработан и добавлен в диалог ниже.")
            else:
                st.info("Текст на изображении не обнаружен.")
        except Exception as exc:
            st.error(f"Ошибка анализа изображения: {exc}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Введите симптом, болезнь или лекарство..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    result = pipeline.diagnose_from_text(user_input)
    
    response = f"**Основные кандидаты:**\n"
    for cand in result.primary_candidates:
        response += f"- {cand['name']} ({cand.get('score', 0):.2f})\n"
    
    response += f"\n**Симптомы:** {', '.join(result.related_symptoms)}\n"
    response += f"**Лекарства:** {', '.join(result.recommended_medicines)}\n"
    response += f"**Уверенность:** {result.confidence}\n"
    response += f"**Статус:** {result.rules_check}"

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
