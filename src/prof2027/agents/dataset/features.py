"""Признаки материалов: числовые признаки + соответствие требованиям (24 колонки).

Все признаки собираются из БД, без обращения к AI-моделям. Результат — pandas
DataFrame и features_json в таблице materials.
"""

from __future__ import annotations

import json

import pandas as pd

from prof2027.db import Material, MaterialCompliance, MethodologicalRequirement

# Числовые признаки, используемые в ML (этапы 3–4).
BASE_FEATURES = [
    "word_count",
    "char_count",
    "sentence_count",
    "media_count",
    "complexity_score",
    "methodology_score",
    "compliant_share",
]


def compliance_map(session) -> dict[int, dict[int, float]]:
    """material_id -> {requirement_id: score}."""
    out: dict[int, dict[int, float]] = {}
    for row in session.query(MaterialCompliance).all():
        out.setdefault(row.material_id, {})[row.requirement_id] = row.score
    return out


def build_frame(session, catalogs) -> pd.DataFrame:
    """Собирает плоский DataFrame: один ряд на материал.

    Колонки: идентификация, категориальные атрибуты, числовые признаки,
    соответствие каждому требованию (compliance_1..N), доля соответствия.
    """
    req_ids = sorted(r.id for r in session.query(MethodologicalRequirement).all())
    comp = compliance_map(session)

    rows = []
    for m in session.query(Material).all():
        cmap = comp.get(m.id, {})
        row = {
            "id": m.id,
            "title": m.title,
            "subject": m.subject,
            "category": m.category,
            "education_level": m.education_level,
            "language": m.language,
            "material_type": m.material_type,
            "class_type": m.class_type,
            "moderation_verdict": m.moderation_verdict,
            "word_count": m.word_count,
            "char_count": m.char_count,
            "sentence_count": m.sentence_count,
            "media_count": m.media_count,
            "complexity_score": float(m.complexity_score or 0.0),
            "complexity_level": m.complexity_level,
            "methodology_score": float(m.methodology_score or 0.0),
        }
        scores = [cmap.get(rid, 0.0) for rid in req_ids]
        for i, s in enumerate(scores, 1):
            row[f"compliance_{i}"] = s
        row["compliant_share"] = round(sum(1 for s in scores if s >= 0.5) / len(scores), 3) if scores else 0.0
        rows.append(row)

    return pd.DataFrame(rows)


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    """Имена числовых признаков, доступных в DataFrame."""
    cols = list(BASE_FEATURES)
    cols += [c for c in frame.columns if c.startswith("compliance_")]
    return cols


def write_features_json(session, frame: pd.DataFrame) -> int:
    """Сохраняет features_json в таблицу materials. Возвращает число обновлённых."""
    num = numeric_columns(frame)
    updated = 0
    for row in frame.itertuples(index=False):
        m = session.get(Material, row.id)
        if m is None:
            continue
        m.features_json = json.dumps(
            {c: (None if pd.isna(getattr(row, c)) else float(getattr(row, c))) for c in num},
            ensure_ascii=False,
        )
        updated += 1
    return updated
