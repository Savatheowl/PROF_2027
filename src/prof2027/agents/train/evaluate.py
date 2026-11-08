"""Оценка моделей: GroupKFold по предметам (защита от утечки), метрики.

classification — accuracy, f1_macro, ROC-AUC (ovr); regression — RMSE, MAE, R².
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold


def _cv_classification(model, X, y, groups):
    acc, f1s, aucs = [], [], []
    classes = np.unique(y)
    gkf = GroupKFold(n_splits=min(3, groups.nunique()))
    for tr, te in gkf.split(X, y, groups):
        model.fit(X.iloc[tr], y.iloc[tr])
        p = model.predict(X.iloc[te])
        acc.append(accuracy_score(y.iloc[te], p))
        f1s.append(f1_score(y.iloc[te], p, average="macro", zero_division=0))
        try:
            proba = model.predict_proba(X.iloc[te])
            if len(classes) == 2:
                # выбираем колонку положительного класса
                pos = proba[:, 1]
            else:
                # OVR: подгоняем колонки под множество классов y
                mapping = {c: i for i, c in enumerate(model.classes_)}
                proba_aligned = np.zeros((len(te), len(classes)))
                for c in classes:
                    if c in mapping:
                        proba_aligned[:, list(classes).index(c)] = proba[:, mapping[c]]
                pos = proba_aligned
            aucs.append(roc_auc_score(y.iloc[te], pos, multi_class="ovr", average="macro"))
        except Exception:
            pass
    return _aggregate(acc), _aggregate(f1s), (_aggregate(aucs) if aucs else None)


def _cv_regression(model, X, y, groups):
    rmses, maes, r2s = [], [], []
    gkf = GroupKFold(n_splits=min(3, groups.nunique()))
    for tr, te in gkf.split(X, y, groups):
        model.fit(X.iloc[tr], y.iloc[tr])
        p = model.predict(X.iloc[te])
        rmses.append(float(np.sqrt(mean_squared_error(y.iloc[te], p))))
        maes.append(float(mean_absolute_error(y.iloc[te], p)))
        r2s.append(float(r2_score(y.iloc[te], p)))
    return _aggregate(rmses), _aggregate(maes), _aggregate(r2s)


def _aggregate(vals: list) -> dict:
    if not vals:
        return None
    import math

    arr = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
    if not arr:
        return None
    return {"mean": round(float(np.mean(arr)), 3), "std": round(float(np.std(arr)), 3)}


def evaluate(task: dict, model, *, fit_final: bool = False):
    """Оценивает модель GroupKFold. Возвращает метрики или None (данных мало)."""
    X, y, groups = task["X"], task["y"], task["groups"]
    if len(y) < 4 or groups.nunique() < 2:
        return None
    if task["kind"] == "classification" and y.nunique() < 2:
        return None
    try:
        if task["kind"] == "classification":
            acc, f1, auc = _cv_classification(model, X, y, groups)
            metrics = {"accuracy": acc, "f1_macro": f1, "roc_auc": auc}
        else:
            rmse, mae, r2 = _cv_regression(model, X, y, groups)
            metrics = {"rmse": rmse, "mae": mae, "r2": r2}
    except Exception:
        return None
    if fit_final:
        model.fit(X, y)
    return metrics


def best_key(metrics: dict | None, kind: str) -> str:
    if not metrics:
        return "n/a"
    if kind == "classification":
        score = metrics.get("roc_auc") or metrics.get("accuracy")
        return "roc_auc" if metrics.get("roc_auc") else "accuracy"
    return "r2"
