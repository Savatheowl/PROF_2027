"""Задачи обучения: признаки, целевая метка, группы (по предмету).

Три задачи:
  complexity   — регрессия сложности (1–5) по текстовым признакам и compliance.
  methodology  — классификация вердикта модерации по текстовым признакам и сложности.
  cluster      — классификация шага последовательного изучения (cluster_sequential).

Признаки подбираются так, чтобы целевая метка НЕ была в признаках (нет утечки).
Группы для GroupKFold — предметы.
"""

from __future__ import annotations

import pandas as pd

from prof2027.agents.dataset.features import build_frame

TEXT_FEATURES = ["word_count", "char_count", "sentence_count", "media_count"]


def compliance_features(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("compliance_")] + ["compliant_share"]


TASKS = {
    "complexity": {
        "kind": "regression",
        "label": "Сложность материала",
        "target": "complexity_score",
        "features": lambda f: TEXT_FEATURES + compliance_features(f),
    },
    "methodology": {
        "kind": "classification",
        "label": "Вердикт модерации",
        "target": "moderation_verdict",
        "features": lambda f: TEXT_FEATURES + ["complexity_score"],
    },
    "cluster": {
        "kind": "classification",
        "label": "Шаг последовательного изучения",
        "target": "cluster_sequential",
        "features": lambda f: TEXT_FEATURES + compliance_features(f) + ["complexity_score"],
    },
}


def task_dataset(session, catalogs, task: str) -> dict:
    """Возвращает {task, kind, X, y, groups, features} или None если данных мало."""
    if task not in TASKS:
        raise ValueError(f"неизвестная задача: {task}, доступны: {', '.join(TASKS)}")
    spec = TASKS[task]
    frame = build_frame(session, catalogs)
    if len(frame) == 0:
        return None

    feats = spec["features"](frame)
    feats = [f for f in feats if f in frame.columns]
    X = frame[feats].astype(float)
    target = spec["target"]
    if target not in frame.columns:
        return None
    y = frame[target]
    groups = frame["subject"].fillna("—").astype(str)
    return {
        "task": task,
        "kind": spec["kind"],
        "label": spec["label"],
        "X": X,
        "y": y,
        "groups": groups,
        "features": feats,
        "ids": frame["id"].tolist(),
    }
