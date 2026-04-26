from src.recommendation import recommend_by_cosine, recommend_by_fuzzy, recommend_labeled_by_cosine


def test_recommendation_returns_results():
    docs = ["Грипп", "Мигрень", "Простуда"]
    result = recommend_by_cosine("грипп", docs, top_k=2)
    assert len(result) == 2


def test_fuzzy_returns_results():
    docs = ["Грипп", "Мигрень", "Простуда"]
    result = recommend_by_fuzzy("мигрень", docs, top_k=2)
    assert len(result) == 2


def test_labeled_recommendation_uses_disease_profile():
    items = [
        ("Грипп", "Грипп Температура Кашель Слабость Парацетамол"),
        ("Мигрень", "Мигрень Головная боль Тошнота Слабость Ибупрофен"),
        ("Простуда", "Простуда Кашель Слабость Ибупрофен"),
    ]
    result = recommend_labeled_by_cosine("кашель температура", items, top_k=1)
    assert result[0][0] == "Грипп"
    assert result[0][1] > 0


def test_recommendation_returns_empty_for_blank_query():
    docs = ["Грипп", "Мигрень", "Простуда"]
    assert recommend_by_cosine("   ", docs) == []
    assert recommend_by_fuzzy("", docs) == []


def test_labeled_recommendation_returns_empty_for_no_items():
    assert recommend_labeled_by_cosine("кашель", [], top_k=2) == []
