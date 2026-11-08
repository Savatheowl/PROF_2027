"""Описание датасета: статистика по атрибутам и распределениям (Markdown + JSON)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from prof2027.agents.dataset.features import numeric_columns


def describe(frame: pd.DataFrame) -> dict:
    cat_cols = ["subject", "category", "education_level", "language",
                "material_type", "class_type", "moderation_verdict"]
    stats: dict = {
        "n_materials": int(len(frame)),
        "n_subjects": int(frame["subject"].nunique()) if len(frame) else 0,
        "n_categories": int(frame["category"].nunique()) if len(frame) else 0,
        "by_subject": _top_counts(frame, "subject", 15),
        "by_category": _top_counts(frame, "category", 15),
        "by_level": _top_counts(frame, "education_level", 10),
        "by_language": _top_counts(frame, "language", 5),
        "by_material_type": _top_counts(frame, "material_type", 10),
        "by_class_type": _top_counts(frame, "class_type", 12),
        "by_verdict": _top_counts(frame, "moderation_verdict", 5),
        "complexity": _dist(frame, "complexity_score"),
        "methodology": _dist(frame, "methodology_score"),
        "word_count": _dist(frame, "word_count"),
    }
    return stats


def _top_counts(frame: pd.DataFrame, col: str, n: int) -> list[dict]:
    if len(frame) == 0 or col not in frame.columns:
        return []
    counts = frame[col].fillna("").astype(str).value_counts().head(n)
    return [{"value": k, "count": int(v)} for k, v in counts.items()]


def _dist(frame: pd.DataFrame, col: str) -> dict:
    if len(frame) == 0 or col not in frame.columns:
        return {}
    s = frame[col].dropna().astype(float)
    if len(s) == 0:
        return {}

    def _num(x: float):
        if x is None:
            return None
        import math

        return None if math.isnan(x) or math.isinf(x) else round(float(x), 3)

    return {
        "mean": _num(s.mean()),
        "std": _num(s.std()),
        "min": _num(s.min()),
        "q1": _num(s.quantile(0.25)),
        "median": _num(s.median()),
        "q3": _num(s.quantile(0.75)),
        "max": _num(s.max()),
    }


def write_description(path: Path, stats: dict, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Описание датасета", ""]
    lines.append(f"Материалов: **{stats['n_materials']}** · предметов: {stats['n_subjects']} · "
                 f"категорий: {stats['n_categories']}")
    lines.append("")

    def block(title: str, items: list[dict], fmt: str = "{value}: {count}"):
        lines.append(f"## {title}")
        lines.append("")
        for it in items:
            lines.append(f"- {fmt.format(**it)}")
        lines.append("")

    block("По предметам", stats["by_subject"])
    block("По категориям", stats["by_category"])
    block("По уровню образования", stats["by_level"])
    block("По языку", stats["by_language"])
    block("По типу материала", stats["by_material_type"])
    block("По типу занятия", stats["by_class_type"])
    block("По вердикту модерации", stats["by_verdict"])

    for label, key in [("Сложность", "complexity"), ("Методическое соответствие", "methodology"),
                       ("Объём (слов)", "word_count")]:
        lines.append(f"## {label}")
        lines.append("")
        d = stats.get(key) or {}
        lines.append(f"- mean={d.get('mean')}, std={d.get('std')}, "
                     f"медиана={d.get('median')}, диапазон [{d.get('min')}, {d.get('max')}]")
        lines.append("")

    lines.append("## Соответствие требованиям")
    lines.append("")
    for col in [c for c in frame.columns if c.startswith("compliance_")]:
        mean = frame[col].mean()
        lines.append(f"- {col}: mean={mean:.3f}")
    lines.append("")
    lines.append("_Сформировано автоматически._")

    path.write_text("\n".join(lines), encoding="utf-8")
    (path.with_suffix(".json")).write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
