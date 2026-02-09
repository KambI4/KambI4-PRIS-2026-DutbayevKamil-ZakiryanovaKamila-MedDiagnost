# src/main.py
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx

from knowledge_graph import create_graph, find_related_entities

st.title("Medical Knowledge Graph 🩺")

# Загружаем граф
G = create_graph()

# Выбор узла
all_nodes = list(G.nodes())
selected_node = st.selectbox("Выберите симптом, болезнь или лекарство:", all_nodes)

# Поиск связей
if st.button("Найти связи"):
    results = find_related_entities(G, selected_node)
    if results:
        st.success(f"Связанные объекты: {', '.join(results)}")
    else:
        st.warning("Связей не найдено")

# Визуализация графа
st.write("### Визуализация графа знаний")

fig, ax = plt.subplots(figsize=(8, 6))
pos = nx.spring_layout(G)

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=2000,
    node_color="lightblue",
    edge_color="gray",
    font_size=10,
    ax=ax
)

st.pyplot(fig)
