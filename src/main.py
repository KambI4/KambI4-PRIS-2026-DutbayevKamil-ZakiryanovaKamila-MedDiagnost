<<<<<<< HEAD
﻿import streamlit as st
=======
import streamlit as st
>>>>>>> 9689ecc (Update NLP module and fix logic)

from knowledge_graph import load_graph
from logic import process_text_message

st.set_page_config(page_title="MedDiagnost Chat", page_icon="🩺")
st.title("Медицинский чат-ассистент")
st.caption("Поддерживаются русские и английские названия (например: 'кашель', 'cough', 'flu').")

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