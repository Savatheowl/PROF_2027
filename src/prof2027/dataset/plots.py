"""Графики датасета: гистограммы, категориальные bar-chart, SVD-scatter, тепловая карта корреляций.

Matplotlib работает в режиме Agg (без GUI), поэтому модуль пригоден для CLI
и тестов на любой платформе. Русский текст поддерживается шрифтом DejaVu Sans
(по умолчанию в matplotlib).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Колонки для гистограмм распределений
HIST_COLUMNS = (
    "word_count",
    "char_count",
    "sentence_count",
    "media_count",
    "complexity_score",
    "methodology_score",
    "estimated_time_hours",
    "avg_word_len",
    "avg_sentence_len_words",
    "ttr",
    "uppercase_ratio",
)

# Категориальные колонки для bar-chart
BAR_COLUMNS = (
    "category",
    "subject",
    "education_level",
    "complexity_level",
    "moderation_verdict",
    "material_type",
    "file_type",
    "language",
)


def _numeric_columns(df: pd.DataFrame, *, exclude_svd: bool = True) -> list[str]:
    cols = []
    for c in df.columns:
        if c in ("material_id",):
            continue
        if exclude_svd and (c.startswith("svd_") or c.startswith("_")):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _save(fig, out_dir: Path, name: str) -> Path:
    fig.tight_layout()
    path = out_dir / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def histograms(df: pd.DataFrame, out_dir: Path) -> Path:
    """Гистограммы числовых признаков (сетка подграфиков)."""
    columns = [c for c in HIST_COLUMNS if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not columns:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "Нет числовых признаков", ha="center", va="center")
        ax.axis("off")
        return _save(fig, out_dir, "features_histograms.png")

    n = len(columns)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows))
    axes = np.atleast_1d(axes).ravel()
    fig.suptitle("Распределения числовых признаков", fontsize=14)
    for ax, col in zip(axes, columns):
        vals = df[col].dropna()
        if vals.empty:
            ax.text(0.5, 0.5, "нет данных", ha="center", va="center")
        else:
            ax.hist(vals, bins=min(30, max(5, int(np.sqrt(len(vals)) + 1))), color="#4C72B0", alpha=0.85)
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes[n:]:
        ax.axis("off")
    return _save(fig, out_dir, "features_histograms.png")


def categorical_bars(df: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    """Bar-chart топ значений для каждой категориальной колонки."""
    files: dict[str, Path] = {}
    for col in BAR_COLUMNS:
        if col not in df.columns:
            continue
        counts = df[col].fillna("(нет)").value_counts().head(10)
        if counts.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.2))
        labels = [str(v)[:24] for v in counts.index]
        bars = ax.bar(range(len(counts)), counts.values, color="#55A868")
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"Распределение: {col}", fontsize=12)
        ax.set_ylabel("Материалов")
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(int(val)),
                    ha="center", va="bottom", fontsize=8)
        files[col] = _save(fig, out_dir, f"categorical_{col}.png")
    return files


def svd_scatter(df: pd.DataFrame, out_dir: Path) -> Path | None:
    """Scatter первых двух компонент SVD, раскраска по категории."""
    if "svd_0" not in df.columns or "svd_1" not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 6))
    colors = plt.get_cmap("tab10")
    categories = df["category"].fillna("(нет)").astype(str)
    unique = sorted(categories.unique())
    color_map = {cat: colors(i % 10) for i, cat in enumerate(unique)}
    for cat in unique:
        mask = categories == cat
        ax.scatter(
            df.loc[mask, "svd_0"], df.loc[mask, "svd_1"],
            label=cat, s=60, alpha=0.85, color=color_map[cat], edgecolors="white", linewidths=0.5,
        )
        xs, ys = df.loc[mask, "svd_0"].tolist(), df.loc[mask, "svd_1"].tolist()
        for x, y in zip(xs, ys):
            ax.annotate("", (x, y))
    ax.set_xlabel("SVD-компонента 0", fontsize=10)
    ax.set_ylabel("SVD-компонента 1", fontsize=10)
    ax.set_title("Семантическое пространство (TF-IDF + SVD) — по категориям", fontsize=12)
    ax.legend(fontsize=8, loc="best")
    return _save(fig, out_dir, "svd_scatter.png")


def correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> Path | None:
    """Тепловая карта попарных корреляций числовых признаков (Pearson)."""
    cols = _numeric_columns(df)
    if len(cols) < 2:
        return None
    corr = df[cols].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(max(7, 0.6 * len(cols)), max(6, 0.6 * len(cols))))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=7)
    ax.set_yticklabels(cols, fontsize=7)
    ax.set_title("Корреляции числовых признаков (Pearson)", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(v) > 0.5 else "black")
    return _save(fig, out_dir, "correlation_heatmap.png")


def plot_all(df: pd.DataFrame, out_dir: Path) -> dict[str, Path | None]:
    """Все графики датасета. Возвращает карту «имя → файл (или None)»."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path | None] = {"histograms": histograms(df, out_dir)}
    files.update({f"bar_{k}": v for k, v in categorical_bars(df, out_dir).items()})
    files["svd_scatter"] = svd_scatter(df, out_dir)
    files["correlation_heatmap"] = correlation_heatmap(df, out_dir)
    return files
