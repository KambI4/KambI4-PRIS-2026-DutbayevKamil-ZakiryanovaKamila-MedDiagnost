from src.knowledge_graph import load_graph
from src.logic import process_text_message


def test_process_text_message_finds_known_term():
    graph = load_graph()
    response = process_text_message("кашель", graph)
    assert "Кашель" in response


def test_process_text_message_empty():
    graph = load_graph()
    response = process_text_message("", graph)
    assert "Введите запрос" in response
