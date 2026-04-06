from __future__ import annotations

import math
import re
from difflib import get_close_matches
from collections import Counter
from typing import Sequence

try:
    from rapidfuzz import process as rf_process
except Exception:
    rf_process = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Zа-яА-Я0-9]+", text.lower().replace("ё", "е"))


def _simple_cosine_score(query: str, document: str) -> float:
    query_tokens = Counter(_tokenize(query))
    document_tokens = Counter(_tokenize(document))

    if not query_tokens or not document_tokens:
        return 0.0

    shared = set(query_tokens) & set(document_tokens)
    dot_product = sum(query_tokens[token] * document_tokens[token] for token in shared)
    query_norm = math.sqrt(sum(value * value for value in query_tokens.values()))
    document_norm = math.sqrt(sum(value * value for value in document_tokens.values()))

    if query_norm == 0 or document_norm == 0:
        return 0.0

    return dot_product / (query_norm * document_norm)


def recommend_by_cosine(query: str, documents: list[str], top_k: int = 3) -> list[tuple[str, float]]:
    if not query.strip() or not documents:
        return []

    if TfidfVectorizer is None or cosine_similarity is None:
        ranked = sorted(
            ((item, _simple_cosine_score(query, item)) for item in documents),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]

    corpus = [query] + documents
    vec = TfidfVectorizer(ngram_range=(1, 2))
    matrix = vec.fit_transform(corpus)
    sims = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

    ranked = sorted(zip(documents, sims), key=lambda x: x[1], reverse=True)
    return [(name, float(score)) for name, score in ranked[:top_k]]


def recommend_labeled_by_cosine(
    query: str,
    items: Sequence[tuple[str, str]],
    top_k: int = 3,
) -> list[tuple[str, float]]:
    if not query.strip() or not items:
        return []

    labels = [label for label, _ in items]
    documents = [document for _, document in items]
    ranked = recommend_by_cosine(query, documents, top_k=len(items))

    scored_labels = []
    for document, score in ranked:
        for index, original_document in enumerate(documents):
            if original_document == document:
                scored_labels.append((labels[index], score))
                documents[index] = f"__used__{index}"
                break

    return scored_labels[:top_k]


def recommend_by_fuzzy(query: str, choices: list[str], top_k: int = 3) -> list[tuple[str, float]]:
    if not query.strip() or not choices:
        return []

    if rf_process is not None:
        results = rf_process.extract(query, choices, limit=top_k)
        return [(name, float(score) / 100.0) for name, score, _ in results]

    fallback = get_close_matches(query, choices, n=top_k, cutoff=0.0)
    return [(name, 0.5) for name in fallback]
