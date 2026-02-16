import json
import os
import re
from difflib import get_close_matches

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RULES_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'rules.json')

ALIASES_BY_CANONICAL = {
    "Грипп": ["грип", "flu", "influenza"],
    "COVID-19": ["covid", "covid19", "covid-19", "ковид", "ковид19", "коронавирус", "корона"],
    "Простуда": ["орви", "cold", "common cold"],
    "Температура": ["жар", "лихорадка", "fever", "temperature"],
    "Кашель": ["кашел", "cough"],
    "Головная боль": ["болит голова", "headache", "migraine"],
    "Слабость": ["упадок сил", "weakness", "fatigue"],
    "Парацетамол": ["paracetamol", "acetaminophen", "тайленол"],
    "Ибупрофен": ["ibuprofen", "нурофен"],
    "Противовирусные": ["антивирусные", "antiviral", "antivirals"],
}


def load_rules():
    # Fallback defaults let the app start even if data/raw/rules.json is absent.
    default_rules = {
        "critical_rules": {"must_be_registered": False},
        "thresholds": {"max_temperature": 39.0},
        "lists": {"danger_symptoms": []},
    }

    if not os.path.exists(RULES_PATH):
        return default_rules

    with open(RULES_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def check_rules(data):
    rules = load_rules()

    # HARD FILTER
    if rules["critical_rules"]["must_be_registered"] and not data["is_registered"]:
        return "Пациент не зарегистрирован"

    if data["temperature"] > rules["thresholds"]["max_temperature"]:
        return "Очень высокая температура"

    for symptom in data["symptoms"]:
        if symptom in rules["lists"]["danger_symptoms"]:
            return f"Опасный симптом: {symptom}"

    return "Пациент прошел первичный осмотр"


def _normalize_text(value):
    normalized = str(value).strip().lower().replace("ё", "е")
    normalized = normalized.replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _build_alias_index(graph):
    alias_to_node = {}
    for node in graph.nodes:
        canonical = str(node)
        alias_to_node[_normalize_text(canonical)] = canonical
        for alias in ALIASES_BY_CANONICAL.get(canonical, []):
            alias_to_node[_normalize_text(alias)] = canonical
    return alias_to_node


def _find_nodes_in_query(query, alias_index):
    query_norm = _normalize_text(query)

    if query_norm in alias_index:
        return [alias_index[query_norm]]

    found = []
    for alias, canonical in alias_index.items():
        if len(alias) >= 4 and alias in query_norm and canonical not in found:
            found.append(canonical)

    return found


def _suggest_nodes(query, alias_index):
    query_norm = _normalize_text(query)
    alias_keys = list(alias_index.keys())

    candidates = get_close_matches(query_norm, alias_keys, n=5, cutoff=0.72)
    suggestions = []
    for alias in candidates:
        canonical = alias_index[alias]
        if canonical not in suggestions:
            suggestions.append(canonical)

    return suggestions[:3]


def process_text_message(text, data_source):
    """
    Принимает текст пользователя, ищет узлы в графе знаний
    и возвращает текстовый ответ.
    """
    query = text.strip()

    if not query:
        return "Введите запрос, например: 'кашель' или 'грипп'."

    query_norm = _normalize_text(query)

    if "привет" in query_norm or "hello" in query_norm:
        return "Привет! Можете писать по-русски или по-английски: например, 'кашель', 'cough', 'flu'."

    alias_index = _build_alias_index(data_source)
    matched_nodes = _find_nodes_in_query(query, alias_index)

    if matched_nodes:
        if len(matched_nodes) == 1:
            target_node = matched_nodes[0]
            neighbors = list(data_source.neighbors(target_node))
            if neighbors:
                return f"Я нашел '{target_node}' в базе. С этим связано: {', '.join(map(str, neighbors))}."
            return f"Я нашел '{target_node}' в базе, но у него пока нет связей."

        lines = [f"В запросе найдено несколько терминов: {', '.join(matched_nodes)}."]
        for node in matched_nodes:
            neighbors = list(data_source.neighbors(node))
            if neighbors:
                lines.append(f"{node}: {', '.join(map(str, neighbors))}.")
            else:
                lines.append(f"{node}: пока нет связей.")
        return "\n".join(lines)

    suggestions = _suggest_nodes(query, alias_index)
    if suggestions:
        return f"Я не знаю такого термина. Возможно, вы имели в виду: {', '.join(suggestions)}."

    return "Я не знаю такого термина"
