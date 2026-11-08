"""Важность признаков: корреляция признаков с целевыми метками.

Целевые метки: complexity_score и methodology_score. Важность = |корреляция
Пирсона| признака с меткой. Отчёт сохраняется в JSON и Markdown.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from prof2027.agents.dataset.features import numeric_columns


def compute_importance(frame: pd.DataFrame, targets: list[str] | None = None) -> list[dict]:
    """Возвращает отсортированный список {feature, target, corr, importance}."""
    targets = targets or ["complexity_score", "methodology_score"]
    feats = numeric_columns(frame)
    out: list[dict] = []
    for target in targets:
        if target not in frame.columns:
            continue
        for f in feats:
            if f not in frame.columns or frame[f].nunique() < 2:
                continue
            corr = frame[f].corr(frame[target])
            if pd.isna(corr):
                continue
            out.append({"feature": f, "target": target, "corr": round(float(corr), 3),
                        "importance": round(abs(float(corr)), 3)})
    out.sort(key=lambda x: (-x["importance"], x["feature"]))
    return out


def top_features(importance: list[dict], target: str, n: int = 10) -> list[dict]:
    return [d for d in importance if d["target"] == target][:n]


def write_importance(path: Path, importance: list[dict], frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    targets = sorted({d["target"] for d in importance})
    lines = ["# Важность признаков датасета", ""]
    for target in targets:
        lines.append(f"## По метке: {target}")
        lines.append("")
        lines.append("| Признак | Корреляция | Важность |")
        lines.append("|---|---:|---:|")
        for d in [x for x in importance if x["target"] == target][:15]:
            lines.append(f"| {d['feature']} | {d['corr']} | {d['importance']} |")
        lines.append("")
    lines.append(f"Материалов в датасете: **{len(frame)}**.")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
