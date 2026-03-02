import streamlit as st

from knowledge_graph import load_graph
from logic import process_text_message
from vision import analyze_image_bytes, image_bytes_to_pil

st.set_page_config(page_title="MedDiagnost Chat", page_icon="🩺", layout="wide")
st.title("Медицинский чат-ассистент")
st.caption(
    "Поддерживаются русские и английские названия (например: 'кашель', 'cough', 'flu')."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "graph" not in st.session_state:
    st.session_state.graph = load_graph()

graph = st.session_state.graph

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
            result = analyze_image_bytes(image_bytes)
            cls = result["classification"]
            ocr = result["ocr"]
            medical = result["medical_extraction"]

            st.subheader("Результат анализа")
            st.write(f"Тип изображения: **{cls['class_label']}**")
            st.write(f"Уверенность: **{cls['confidence']:.2f}**")
            st.json(cls["features"])

            st.subheader("OCR")
            if ocr["available"]:
                if ocr["text"].strip():
                    st.text_area("Извлеченный текст", ocr["text"], height=180)

                    st.subheader("Структурированный мед. разбор")
                    if medical["diagnosis_candidates"]:
                        st.markdown("**Предполагаемый диагноз / заключение:**")
                        for item in medical["diagnosis_candidates"]:
                            st.write(f"- {item}")
                    else:
                        st.write("Предполагаемый диагноз не выделен.")

                    if medical["icd_codes"]:
                        st.markdown(f"**МКБ-коды:** {', '.join(medical['icd_codes'])}")

                    if medical["prescriptions"]:
                        st.markdown("**Назначения:**")
                        for idx, rx in enumerate(medical["prescriptions"], start=1):
                            parts = [f"{idx}. {rx['line']}"]
                            if rx["dosage"]:
                                parts.append(f"доза: {rx['dosage']}")
                            if rx["frequency"]:
                                parts.append(f"частота: {rx['frequency']}")
                            st.write(" | ".join(parts))
                    else:
                        st.write("Назначения не выделены.")

                    if st.button("Отправить OCR-текст в мед. анализ", use_container_width=True):
                        bot_response = process_text_message(ocr["text"], graph)
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": f"Анализ OCR-текста:\n{bot_response}",
                            }
                        )
                        st.success("OCR-текст обработан и добавлен в диалог ниже.")
                else:
                    st.info("Текст на изображении не обнаружен.")
            else:
                st.warning(ocr["message"])
        except Exception as exc:
            st.error(f"Ошибка анализа изображения: {exc}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Введите симптом, болезнь или лекарство..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    bot_response = process_text_message(user_input, graph)

    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.markdown(bot_response)
