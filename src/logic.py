<<<<<<< HEAD
﻿import json
import os
import re
from difflib import get_close_matches
=======
import re
from difflib import get_close_matches
import spacy
>>>>>>> 9689ecc (Update NLP module and fix logic)


<<<<<<< HEAD
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
    # Значения по умолчанию, чтобы приложение не падало без rules.json.
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

    # Критическая проверка.
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


def _best_fuzzy_node(value, alias_index, cutoff=0.82):
    matches = get_close_matches(value, list(alias_index.keys()), n=1, cutoff=cutoff)
    if matches:
        return alias_index[matches[0]]
    return None


def _extract_candidates(query):
    query_norm = _normalize_text(query)
    tokens = re.findall(r"[a-zA-Zа-яА-Я0-9]+", query_norm)
    candidates = [query_norm]
    candidates.extend(tokens)

    # Добавляем короткие фразы из соседних слов (например, "болит голова").
    for i in range(len(tokens)):
        if i + 1 < len(tokens):
            candidates.append(f"{tokens[i]} {tokens[i + 1]}")
        if i + 2 < len(tokens):
            candidates.append(f"{tokens[i]} {tokens[i + 1]} {tokens[i + 2]}")

    # Сохраняем порядок и уникальность.
    seen = set()
    uniq = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            uniq.append(candidate)
    return uniq


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

    # Нечеткое распознавание: учитываем опечатки как валидный ввод.
    if not matched_nodes:
        fuzzy_found = []
        for candidate in _extract_candidates(query):
            node = _best_fuzzy_node(candidate, alias_index, cutoff=0.82)
            if node and node not in fuzzy_found:
                fuzzy_found.append(node)
        matched_nodes = fuzzy_found

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
=======
# =========================
# NLP модель
# =========================

try:
    nlp = spacy.load("ru_core_news_sm")
except Exception:
    nlp = None


# =========================
# Дата (regex)
# =========================

MONTHS = (
    "января|февраля|марта|апреля|мая|июня|июля|августа|"
    "сентября|октября|ноября|декабря"
)

DATE_PATTERN = re.compile(
    rf"\b(\d{{1,2}}\s+(?:{MONTHS}))\b",
    re.IGNORECASE
)


def _extract_dates(text):
    return DATE_PATTERN.findall(text)


# =========================
# Вспомогательные
# =========================

def _normalize_text(text):
    return text.strip().lower().replace("ё", "е")


def _extract_named_entities(text):
    if not nlp:
        return []
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


def _extract_medical_entities(text, graph):
    if not nlp:
        return []

    doc = nlp(text.lower())
    lemmas = [token.lemma_ for token in doc]

    found = []

    for node in graph.nodes:
        node_tokens = node.lower().split()
        if any(token in lemmas for token in node_tokens):
            found.append(node)

    return list(set(found))


# =========================
# Главная функция
# =========================

def process_text_message(text, graph):

    query = text.strip()
    if not query:
        return "Введите медицинский запрос."

    response_parts = []

    # --- 1. NER через spaCy ---
    ner_entities = _extract_named_entities(query)

    # --- 2. DATE через regex ---
    regex_dates = _extract_dates(query)
    for dt in regex_dates:
        ner_entities.append((dt, "DATE"))

    if ner_entities:
        formatted = ", ".join([f"{t} ({l})" for t, l in ner_entities])
        response_parts.append(f"Распознаны сущности: {formatted}.")

    # --- 3. Медицинские сущности ---
    medical = _extract_medical_entities(query, graph)

    if medical:
        for node in medical:
            neighbors = list(graph.neighbors(node))
            if neighbors:
                response_parts.append(
                    f"Найдено: {node}. Связанные элементы: {', '.join(neighbors)}."
                )
            else:
                response_parts.append(f"{node} найден, но связей нет.")

    if response_parts:
        return "\n".join(response_parts)

    return "Я не смог распознать медицинские сущности в вашем запросе."
>>>>>>> 9689ecc (Update NLP module and fix logic)
