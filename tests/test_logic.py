from src.knowledge_graph import load_graph
from src.logic import check_rules, process_text_message


def test_process_text_message_finds_known_term():
    graph = load_graph()
    response = process_text_message("кашель", graph)
    assert "Кашель" in response


def test_process_text_message_empty():
    graph = load_graph()
    response = process_text_message("", graph)
    assert "Введите запрос" in response


def test_process_text_message_matches_alias():
    graph = load_graph()
    response = process_text_message("flu", graph)
    assert "Грипп" in response


def test_process_text_message_suggests_close_term():
    graph = load_graph()
    response = process_text_message("кашел", graph)
    assert "Кашель" in response


def test_check_rules_requires_registration(monkeypatch):
    monkeypatch.setattr(
        "src.logic.load_rules",
        lambda: {
            "critical_rules": {"must_be_registered": True},
            "thresholds": {"max_temperature": 39.0},
            "lists": {"danger_symptoms": []},
        },
    )
    result = check_rules({"is_registered": False, "temperature": 36.6, "symptoms": []})
    assert result == "Пациент не зарегистрирован"


def test_check_rules_detects_high_temperature(monkeypatch):
    monkeypatch.setattr(
        "src.logic.load_rules",
        lambda: {
            "critical_rules": {"must_be_registered": False},
            "thresholds": {"max_temperature": 39.0},
            "lists": {"danger_symptoms": []},
        },
    )
    result = check_rules({"is_registered": True, "temperature": 39.5, "symptoms": []})
    assert result == "Очень высокая температура"


def test_check_rules_detects_danger_symptom(monkeypatch):
    monkeypatch.setattr(
        "src.logic.load_rules",
        lambda: {
            "critical_rules": {"must_be_registered": False},
            "thresholds": {"max_temperature": 39.0},
            "lists": {"danger_symptoms": ["судороги"]},
        },
    )
    result = check_rules({"is_registered": True, "temperature": 37.0, "symptoms": ["судороги"]})
    assert result == "Опасный симптом: судороги"


def test_check_rules_passes_safe_patient(monkeypatch):
    monkeypatch.setattr(
        "src.logic.load_rules",
        lambda: {
            "critical_rules": {"must_be_registered": False},
            "thresholds": {"max_temperature": 39.0},
            "lists": {"danger_symptoms": ["судороги"]},
        },
    )
    result = check_rules({"is_registered": True, "temperature": 36.8, "symptoms": ["кашель"]})
    assert result == "Пациент прошел первичный осмотр"
