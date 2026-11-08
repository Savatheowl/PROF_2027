"""Признаки материалов: текст, медиа, классификация, семантика (TF-IDF + SVD).

Таблица признаков — основа `dataset.csv`/`.parquet` и вход для этапов 3-4
(кластеризация и обучение). Семантика — TF-IDF + TruncatedSVD (не e5),
см. docs/ARCHITECTURE.md §8.1.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import pandas as pd

# Базовые категориальные колонки, которые копируются в датасет как строки
CATEGORICAL_COLUMNS = (
    "category",
    "subject",
    "topic",
    "education_level",
    "language",
    "material_type",
    "class_type",
    "file_type",
    "complexity_level",
    "moderation_verdict",
)

NUMERIC_BASE = (
    "word_count",
    "char_count",
    "sentence_count",
    "media_count",
)

# Русские стоп-слова (компактный список для TF-IDF на учебных текстах)
STOP_WORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
    "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще",
    "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли",
    "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь",
    "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей",
    "может", "они", "тут", "где", "есть", "надо", "ней", "для", "тебя", "чем",
    "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже", "себе", "под",
    "будет", "ж", "тогда", "кто", "этот", "того", "потому", "этого", "какой",
    "совсем", "ним", "здесь", "этом", "один", "почти", "мой", "тем", "чтобы",
    "нее", "сейчас", "были", "куда", "зачем", "всех", "никогда", "можно",
    "при", "об", "другой", "хоть", "после", "над", "больше", "тот", "через",
    "эти", "нас", "про", "всего", "них", "какая", "много", "разве", "три",
    "эту", "моя", "впрочем", "хорошо", "свою", "этой", "перед", "иногда",
    "лучше", "чуть", "том", "нельзя", "такой", "им", "более", "всегда",
    "конечно", "всю", "между", "это", "каждое", "года", "очень", "нет",
}

_WORD_RE = re.compile(r"[а-яё]+", re.IGNORECASE)


def materials_to_frame(materials: Sequence[dict]) -> pd.DataFrame:
    """Строит базовую таблицу из списка материалов (словарей из БД)."""
    rows = []
    for m in materials:
        row = {"material_id": m["id"], "title": m["title"] or ""}
        for col in CATEGORICAL_COLUMNS:
            row[col] = m.get(col)
        for col in NUMERIC_BASE:
            row[col] = int(m.get(col) or 0)
        row["complexity_score"] = m.get("complexity_score")
        row["methodology_score"] = m.get("methodology_score")
        row["text_content"] = m.get("text_content") or ""
        row["media_descriptions"] = m.get("media_descriptions") or []
        row["estimated_time_hours"] = m.get("estimated_time_hours")
        rows.append(row)
    return pd.DataFrame(rows)


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Производные признаки текста: длина слов/предложений, TTR, заглавные."""
    words = df["text_content"].map(_tokenize)

    total_words = words.map(len).astype(int)
    unique_words = words.map(lambda ws: len(set(ws))).astype(int)

    df["avg_word_len"] = np.where(
        total_words > 0, df["char_count"] / total_words.clip(lower=1), np.nan
    ).astype(float)
    df["avg_sentence_len_words"] = np.where(
        df["sentence_count"] > 0,
        total_words / df["sentence_count"].clip(lower=1),
        np.nan,
    ).astype(float)
    df["unique_words"] = unique_words
    df["ttr"] = np.where(total_words > 0, unique_words / total_words.clip(lower=1), np.nan).astype(float)

    # доля слов в верхнем регистре (заголовки)
    upper = df["text_content"].map(
        lambda t: sum(1 for ch in t if ch.isalpha() and ch.isupper())
    )
    df["uppercase_ratio"] = np.where(
        df["char_count"] > 0, upper / df["char_count"].clip(lower=1), np.nan
    ).astype(float)

    # числовые признаки из media_descriptions по типам
    df["media_image"] = df["media_descriptions"].map(lambda ms: _media_count(ms, "image"))
    df["media_audio"] = df["media_descriptions"].map(lambda ms: _media_count(ms, "audio"))
    df["media_video"] = df["media_descriptions"].map(lambda ms: _media_count(ms, "video"))
    df["has_media"] = df["media_count"] > 0
    return df


def add_semantic_features(
    df: pd.DataFrame,
    *,
    n_components: int = 25,
    min_df: int = 1,
    max_df: float = 1.0,
    text_col: str = "text_content",
    prefix: str = "svd",
):
    """TF-IDF + TruncatedSVD по текстам материалов.

    Возвращает (df, tfidf, svd). Если текстов слишком мало — компоненты
    не добавляются, модели возвращаются None.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    corpus = df[text_col].fillna("").tolist()
    tfidf = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        stop_words=list(STOP_WORDS),
        min_df=min_df,
        max_df=max_df,
    )
    try:
        matrix = tfidf.fit_transform(corpus)
    except ValueError:
        return df, None, None

    n_rows, n_features = matrix.shape
    k = min(int(n_components), max(0, n_rows - 1), max(0, n_features - 1))
    if k < 1:
        return df, tfidf, None

    svd = TruncatedSVD(n_components=k, random_state=42)
    reduced = svd.fit_transform(matrix)

    for i in range(k):
        df[f"{prefix}_{i}"] = reduced[:, i].astype(float)
    df["_svd_k"] = k
    return df, tfidf, svd


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def _media_count(media: list, kind: str) -> int:
    if not media:
        return 0
    return sum(1 for item in media if isinstance(item, dict) and item.get("type") == kind)
