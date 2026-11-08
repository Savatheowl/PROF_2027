"""Три стратегии кластеризации учебных материалов.

parallel    — параллельное изучение: группа материалов из РАЗНЫХ предметов,
              один материал из каждого предмета в группе.
sequential  — последовательное изучение: внутри предмета материалы по нарастанию
              сложности разбиваются на учебные шаги.
complexity  — группировка по сложности (KMeans по complexity_score).

Метрики считаются на эмбеддингах TF-IDF+SVD.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

STRATEGIES = ["parallel", "sequential", "complexity"]


def labels_parallel(records: list[dict]) -> np.ndarray:
    """Жадная раскладка по группам: в каждой группе не более одного материала
    из каждого предмета. Материалы берутся по нарастанию сложности."""
    subjects_order: dict[str, int] = {}
    labels = np.zeros(len(records), dtype=int)
    bins: list[set[str]] = []
    for m in sorted(records, key=lambda r: (float(r["complexity_score"] or 0), r["id"])):
        if m["subject"] not in subjects_order:
            subjects_order[m["subject"]] = len(subjects_order)
        for bin_i, sset in enumerate(bins):
            if m["subject"] not in sset:
                sset.add(m["subject"])
                labels[m["pos"]] = bin_i
                break
        else:
            bins.append({m["subject"]})
            labels[m["pos"]] = len(bins) - 1
    return labels


def labels_sequential(records: list[dict], group_size: int = 3) -> np.ndarray:
    """Внутри предмета материалы по сложности разбиваются на шаги по group_size."""
    labels = np.zeros(len(records), dtype=int)
    idx = 0
    for subject in sorted({r["subject"] for r in records}):
        group = [r for r in records if r["subject"] == subject]
        group.sort(key=lambda r: (float(r["complexity_score"] or 0), r["id"]))
        for i, m in enumerate(group):
            labels[m["pos"]] = idx + i // group_size
        idx += len(group)
    return labels


def labels_complexity(records: list[dict], n_clusters: int | None = None) -> np.ndarray:
    """KMeans по нормализованному complexity_score. Число кластеров ≤ 5."""
    n = len(records)
    if n == 0:
        return np.zeros(0, dtype=int)
    values = np.array([[float(r["complexity_score"] or 0.0)] for r in records])
    n_unique = len(np.unique(values[:, 0]))
    if n_clusters is None:
        n_clusters = max(1, min(5, n_unique))
    if n_clusters < 2 or n_unique < 2:
        return np.zeros(n, dtype=int)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(values)
    return km.labels_


def assign(strategy: str, records: list[dict], group_size: int = 3) -> np.ndarray:
    if strategy == "parallel":
        return labels_parallel(records)
    if strategy == "sequential":
        return labels_sequential(records, group_size)
    if strategy == "complexity":
        return labels_complexity(records)
    raise ValueError(f"неизвестная стратегия: {strategy}")
