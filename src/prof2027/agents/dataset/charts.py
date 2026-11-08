"""Графики датасета (matplotlib, Agg). Сохраняются в reports/charts/."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import pandas as pd  # noqa: E402

plt.rcParams["figure.dpi"] = 110


def make_charts(frame: pd.DataFrame, out_dir: Path, catalogs=None) -> list[Path]:
    """Строит и сохраняет графики. Возвращает пути к файлам."""
    out_dir.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []
    if len(frame) == 0:
        return made

    # 1. Распределение сложности
    fig, ax = plt.subplots()
    ax.hist(frame["complexity_score"].dropna(), bins=range(0, 8), edgecolor="white", color="#4C78A8")
    ax.set_title("Распределение сложности материалов")
    ax.set_xlabel("Сложность (1–5)")
    ax.set_ylabel("Материалов")
    fp = out_dir / "complexity_hist.png"
    fig.savefig(fp); plt.close(fig); made.append(fp)

    # 2. Методическое соответствие
    fig, ax = plt.subplots()
    ax.hist(frame["methodology_score"].dropna(), bins=20, edgecolor="white", color="#72B7B2")
    ax.set_title("Методическое соответствие (0–1)")
    ax.set_xlabel("Средний балл по 24 требованиям")
    fp = out_dir / "methodology_hist.png"
    fig.savefig(fp); plt.close(fig); made.append(fp)

    # 3. Топ предметов
    top = frame["subject"].value_counts().head(12)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top.index[::-1], top.values[::-1], color="#F58518")
    ax.set_title("Материалы по предметам (топ)")
    ax.set_xlabel("Количество")
    fig.tight_layout()
    fp = out_dir / "subjects_top.png"
    fig.savefig(fp); plt.close(fig); made.append(fp)

    # 4. Среднее соответствие по требованиям
    comp_cols = [c for c in frame.columns if c.startswith("compliance_")]
    if comp_cols:
        means = frame[comp_cols].mean()
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(range(len(means)), means.values, color="#F6CF65")
        ax.axhline(0.5, color="#D34E24", linestyle="--", label="порог соответствия")
        ax.set_title("Среднее соответствие по 24 требованиям")
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels([c.replace("compliance_", "") for c in means.index], rotation=90, fontsize=7)
        ax.set_ylim(0, 1)
        ax.legend()
        fig.tight_layout()
        fp = out_dir / "compliance_means.png"
        fig.savefig(fp); plt.close(fig); made.append(fp)

    # 5. Уровень образования
    if "education_level" in frame.columns:
        counts = frame["education_level"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(counts.values, labels=counts.index, autopct="%1.0f%%", startangle=90)
        ax.set_title("Материалы по уровню образования")
        fp = out_dir / "levels_pie.png"
        fig.savefig(fp); plt.close(fig); made.append(fp)

    return made
