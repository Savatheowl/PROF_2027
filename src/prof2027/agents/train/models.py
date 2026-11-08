"""Реестр алгоритмов: минимум 3 на задачу. Обучение sklearn, сохранение joblib."""

from __future__ import annotations

import joblib
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def models_for(kind: str) -> list[tuple[str, object]]:
    """(имя, модель). Регрессия и классификация — по 3 алгоритма."""
    if kind == "regression":
        return [
            ("linear", Pipeline([("sc", StandardScaler()),
                                 ("m", LinearRegression())])),
            ("random_forest", RandomForestRegressor(n_estimators=200, random_state=42)),
            ("gradient_boosting", GradientBoostingRegressor(random_state=42)),
        ]
    return [
        ("logistic", Pipeline([("sc", StandardScaler()),
                               ("m", LogisticRegression(max_iter=2000, class_weight="balanced"))])),
        ("random_forest", RandomForestClassifier(n_estimators=200, random_state=42,
                                                 class_weight="balanced")),
        ("gradient_boosting", GradientBoostingClassifier(random_state=42)),
    ]


def save_model(model, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)
