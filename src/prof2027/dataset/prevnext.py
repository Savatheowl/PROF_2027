"""Prev/next связи между материалами.

Внутри предмета — по возрастанию сложности (далее по объёму текста).
Семантический сосед — по косинусной близости SVD-векторов (для этапа 3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_prev_next(df: pd.DataFrame) -> pd.DataFrame:
    """Проставляет prev/next внутри предмета (fallback — категория)."""
    df = df.copy()
    df["previous_material_id"] = np.nan
    df["next_material_id"] = np.nan
    df["has_previous"] = False
    df["has_next"] = False

    df["_group"] = df["subject"].fillna(df["category"]).fillna("__none__")

    # внутри группы сортируем по сложности (NaN в конец), затем по объёму
    for _g, group in df.groupby("_group", sort=False):
        if len(group) < 2:
            continue
        ordered = group.sort_values(
            by=["complexity_score", "word_count"],
            ascending=[True, True],
            na_position="last",
        )
        ids = ordered["material_id"].tolist()
        for i, mid in enumerate(ids):
            df.loc[df["material_id"] == mid, "previous_material_id"] = ids[i - 1] if i > 0 else np.nan
            df.loc[df["material_id"] == mid, "next_material_id"] = ids[i + 1] if i < len(ids) - 1 else np.nan
            df.loc[df["material_id"] == mid, "has_previous"] = i > 0
            df.loc[df["material_id"] == mid, "has_next"] = i < len(ids) - 1

    df = df.drop(columns=["_group"])
    return df


def add_semantic_neighbor(df: pd.DataFrame) -> pd.DataFrame:
    """Ближайший материал по косинусной близости SVD-векторов."""
    df = df.copy()
    df["semantic_neighbor_id"] = np.nan
    k = 0
    if "_svd_k" in df.columns and df["_svd_k"].notna().any():
        k = int(df["_svd_k"].fillna(0).max())
    if k < 1:
        return df

    cols = [f"svd_{i}" for i in range(k)]
    vectors = df[cols].fillna(0.0).to_numpy(dtype=float)
    norms = np.linalg.norm(vectors, axis=1)
    norms = np.where(norms == 0, 1.0, norms)  # избегаем деления на 0
    normalized = vectors / norms[:, None]

    ids = df["material_id"].to_numpy()
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -1.0)  # исключаем самого себя
    best = similarity.argmax(axis=1)

    for i in range(len(df)):
        df.iloc[i, df.columns.get_loc("semantic_neighbor_id")] = int(ids[best[i]])
    return df


def dataset_links(df: pd.DataFrame) -> list[dict]:
    """Список связей prev/next для записи в БД."""
    links = []
    for _i, row in df.iterrows():
        links.append(
            {
                "material_id": int(row["material_id"]),
                "has_previous": bool(row["has_previous"]),
                "has_next": bool(row["has_next"]),
                "previous_material_id": _int_or_none(row["previous_material_id"]),
                "next_material_id": _int_or_none(row["next_material_id"]),
            }
        )
    return links


def _int_or_none(value) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)
