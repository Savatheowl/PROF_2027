"""Проверка дрейфа данных: сравнение статистики признаков с эталоном обучения."""

from __future__ import annotations

import numpy as np
import pandas as pd


def reference_stats(X: pd.DataFrame) -> dict:
    """mean/std/источники для каждого признака на обучающем корпусе."""
    stats = {}
    for col in X.columns:
        s = X[col].astype(float)
        stats[col] = {
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4),
        }
    return stats


def check_drift(X_new: pd.DataFrame, ref: dict) -> dict:
    """Сравнивает текущие признаки с эталоном. Возвращает {score, level, per_feature}."""
    per_feature = {}
    worst = 0.0
    for col, r in ref.items():
        if col not in X_new.columns:
            continue
        s = X_new[col].astype(float)
        std = r.get("std") or 0.0
        if std > 1e-9:
            z = abs(float(s.mean() - r["mean"]) / std)
        else:
            z = 0.0
        per_feature[col] = round(z, 3)
        worst = max(worst, z)
    if worst >= 2.0:
        level = "high"
    elif worst >= 1.0:
        level = "low"
    else:
        level = "none"
    return {"score": round(worst, 3), "level": level, "per_feature": per_feature}
