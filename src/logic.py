import json
import os
import re
from difflib import get_close_matches

import spacy


RULES_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "raw", "rules.json")
)

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

MONTHS = (
    "января|февраля|марта|апреля|мая|июня|июля|августа|"
    "сентября|октября|ноября|декабря"
)
DATE_PATTERN = re.compile(rf"\\b(\\d{{1,2}}\\s+(?:{MONTHS}))\\b", re.IGNORECASE)

try:
    NLP = spacy.load("ru_core_news_sm")
except Exception:
    NLP = None


def load_rules():
    default_rules = {
        "critical_rules": {"must_be_registered": False},
        "thresholds": {"max_temperature": 39.0},
        "lists": {"danger_symptoms": []},
    }

    if not os.path.exists(RULES_PATH):
        return default_rules

    with open(RULES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def check_rules(data):
    rules = load_rules()

    if rules["critical_rules"]["must_be_registered"] and not data.get("is_registered", False):
        return "Пациент не зарегистрирован"

    if data.get("temperature", 0) > rules["thresholds"]["max_temperature"]:
        return "Очень высокая температура"

    for symptom in data.get("symptoms", []):
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

    for i in range(len(tokens)):
        if i + 1 < len(tokens):
            candidates.append(f"{tokens[i]} {tokens[i + 1]}")
        if i + 2 < len(tokens):
            candidates.append(f"{tokens[i]} {tokens[i + 1]} {tokens[i + 2]}")

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


def _extract_dates(text):
    return DATE_PATTERN.findall(text)


def _extract_named_entities(text):
    if not NLP:
        return []
    doc = NLP(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


def _extract_medical_entities(text, graph):
    if not NLP:
        return []

    doc = NLP(text.lower())
    lemmas = [token.lemma_ for token in doc]

    found = []
    for node in graph.nodes:
        node_tokens = node.lower().split()
        if any(token in lemmas for token in node_tokens):
            found.append(node)

    return list(set(found))


def process_text_message(text, graph):
    query = text.strip()

    if not query:
        return "Введите запрос, например: 'кашель' или 'грипп'."

    query_norm = _normalize_text(query)
    if "привет" in query_norm or "hello" in query_norm:
        return "Привет! Можете писать по-русски или по-английски: например, 'кашель', 'cough', 'flu'."

    alias_index = _build_alias_index(graph)
    matched_nodes = _find_nodes_in_query(query, alias_index)

    if not matched_nodes:
        fuzzy_found = []
        for candidate in _extract_candidates(query):
            node = _best_fuzzy_node(candidate, alias_index, cutoff=0.82)
            if node and node not in fuzzy_found:
                fuzzy_found.append(node)
        matched_nodes = fuzzy_found

    response_parts = []

    ner_entities = _extract_named_entities(query)
    for dt in _extract_dates(query):
        ner_entities.append((dt, "DATE"))

    if ner_entities:
        formatted = ", ".join([f"{t} ({l})" for t, l in ner_entities])
        response_parts.append(f"Распознаны сущности: {formatted}.")

    nlp_medical = _extract_medical_entities(query, graph)
    if nlp_medical:
        for node in sorted(nlp_medical):
            if node not in matched_nodes:
                matched_nodes.append(node)

    if matched_nodes:
        if len(matched_nodes) == 1:
            target_node = matched_nodes[0]
            neighbors = list(graph.neighbors(target_node))
            if neighbors:
                response_parts.append(
                    f"Я нашел '{target_node}' в базе. С этим связано: {', '.join(map(str, neighbors))}."
                )
            else:
                response_parts.append(f"Я нашел '{target_node}' в базе, но у него пока нет связей.")
        else:
            response_parts.append(f"В запросе найдено несколько терминов: {', '.join(matched_nodes)}.")
            for node in matched_nodes:
                neighbors = list(graph.neighbors(node))
                if neighbors:
                    response_parts.append(f"{node}: {', '.join(map(str, neighbors))}.")
                else:
                    response_parts.append(f"{node}: пока нет связей.")

        return "\n".join(response_parts)

    suggestions = _suggest_nodes(query, alias_index)
    if suggestions:
        response_parts.append(
            f"Я не знаю такого термина. Возможно, вы имели в виду: {', '.join(suggestions)}."
        )
        return "\n".join(response_parts)

    if response_parts:
        return "\n".join(response_parts)

    return "Я не знаю такого термина"
