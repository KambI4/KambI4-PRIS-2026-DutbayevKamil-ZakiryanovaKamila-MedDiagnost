from src.recommendation import recommend_by_cosine, recommend_by_fuzzy, recommend_labeled_by_cosine


def test_recommendation_returns_results():
    docs = ["Грипп", "COVID-19", "Простуда"]
    result = recommend_by_cosine("грипп", docs, top_k=2)
    assert len(result) == 2


def test_fuzzy_returns_results():
    docs = ["Грипп", "COVID-19", "Простуда"]
    result = recommend_by_fuzzy("ковид", docs, top_k=2)
    assert len(result) == 2


def test_labeled_recommendation_uses_disease_profile():
    items = [
        ("Грипп", "Грипп Температура Кашель Слабость Парацетамол"),
        ("COVID-19", "COVID-19 Температура Головная боль Слабость Противовирусные"),
        ("Простуда", "Простуда Кашель Слабость Ибупрофен"),
    ]
    result = recommend_labeled_by_cosine("кашель температура", items, top_k=1)
    assert result[0][0] == "Грипп"
    assert result[0][1] > 0
