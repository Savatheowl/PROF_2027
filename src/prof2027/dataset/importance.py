"""Важность признаков: корреляции, mutual information, permutation importance.

Работает на любом наборе строк; при малых выборках методы аккуратно
пропускаются (возвращается причина), а не падают.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

IMPORTANCE_METHODS = ("pearson", "spearman", "mutual_info", "permutation")

_SKIP_REASON = {
    "pearson": "Требуется ≥2 строки с непустой целью",
    "spearman": "Требуется ≥2 строки с непустой целью",
    "mutual_info": "Требуется ≥3 строки",
    "permutation": "Требуется ≥5 строк и ≥2 признаков",
}


def numeric_feature_columns(df: pd.DataFrame, exclude: set[str] | None = None) -> list[str]:
    """Колонки для анализа важности (числовые признаки, не id/метки)."""
    exclude = exclude or {"material_id"}
    cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if col.startswith("svd_") or col.startswith("_"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def _target_clean(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Выборка X/y для анализа важности (строки с непустой целью)."""
    if target not in df.columns:
        raise ValueError(f"Целевая колонка {target!r} отсутствует в датасете")
    y = df[target]
    rows = y.notna()
    if rows.sum() < 2:
        return pd.DataFrame(), pd.Series(dtype=float)
    X = df.loc[rows, numeric_feature_columns(df)]
    X = X.drop(columns=[target], errors="ignore")
    return X, y[rows].astype(float)


def compute_importance(
    df: pd.DataFrame,
    target: str,
    *,
    methods: tuple[str, ...] = IMPORTANCE_METHODS,
    n_permutations: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Таблица важности признаков по целевой переменной (числовая или метка)."""
    X, y = _target_clean(df, target)
    if X.shape[0] < 2:
        return _empty_importance(methods, _SKIP_REASON["pearson"])

    from scipy.stats import pearsonr, spearmanr
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    is_regression = pd.api.types.is_numeric_dtype(y)
    y_arr = y.to_numpy(dtype=float)
    y_encoded = y_arr if is_regression else y.astype("category").cat.codes.astype(int).to_numpy()

    # Единая числовая матрица признаков (смешанные dtype → float; NaN заменяем)
    X_num = X.apply(pd.to_numeric, errors="coerce")
    X_filled = np.nan_to_num(X_num.to_numpy(dtype=float))

    result = pd.DataFrame(index=X.columns)
    result["type"] = "numeric"

    if "pearson" in methods and X.shape[0] >= 2 and is_regression:
        values = []
        for x in X_filled.T:
            has_var = len(set(x)) > 1
            values.append(pearsonr(x, y_arr)[0] if has_var else np.nan)
        result["pearson"] = values
    if "spearman" in methods and X.shape[0] >= 3:
        values = []
        for x in X_filled.T:
            has_var = len(set(x)) > 1
            values.append(spearmanr(x, y_encoded)[0] if has_var else np.nan)
        result["spearman"] = values

    if "mutual_info" in methods and X.shape[0] >= 3:
        try:
            if is_regression:
                mi = mutual_info_regression(X_filled, y_arr, random_state=random_state)
            else:
                mi = mutual_info_classif(X_filled, y_encoded, random_state=random_state)
            result["mutual_info"] = mi
        except ValueError:
            result["mutual_info"] = np.nan

    if "permutation" in methods and X.shape[0] >= 5 and X.shape[1] >= 2 and is_regression:
        rng = np.random.default_rng(random_state)
        base = _ridge_score(X_filled, y_arr)
        scores = []
        for _col_idx in range(X_filled.shape[1]):
            perm_scores = []
            for _ in range(n_permutations):
                perm = rng.permutation(y_arr)
                perm_scores.append(_ridge_score(X_filled, perm))
            scores.append(float(base) - float(np.mean(perm_scores)))
        result["permutation"] = scores

    result = result.replace([np.inf, -np.inf], np.nan)
    return result.reset_index().rename(columns={"index": "feature"})


def _ridge_score(X: np.ndarray, y: np.ndarray) -> float:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error

    model = Ridge(alpha=1.0)
    model.fit(X, y)
    pred = model.predict(X)
    return float(mean_squared_error(y, pred))


def _empty_importance(methods: tuple[str, ...], reason: str) -> pd.DataFrame:
    rows = []
    for method in methods:
        rows.append({"feature": f"[{method}]", "type": "note", method: reason})
    return pd.DataFrame(rows)
