import streamlit as st
from mock_data import test_entity
from logic import check_rules

st.title("🩺 Медицинский диагност")

temperature = st.number_input("Температура", value=test_entity["temperature"])
is_registered = st.checkbox("Пациент зарегистрирован", value=test_entity["is_registered"])

if st.button("Проверить"):
    data = {
        "temperature": temperature,
        "is_registered": is_registered,
        "complaint_text": test_entity["complaint_text"],
        "symptoms": test_entity["symptoms"]
    }

    st.write(check_rules(data))
