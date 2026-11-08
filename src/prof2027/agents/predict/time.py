"""Оценка времени изучения материала.

Эвристика, калиброванная уровнем сложности: скорость чтения падает с ростом
сложности. Время = слова / скорость + медиа-доля + фиксированные накладные.
"""

from __future__ import annotations

from prof2027.db import Material

# слов в час по уровню сложности (индекс 0 = очень низкая ... 4 = очень высокая)
WORDS_PER_HOUR = [600.0, 500.0, 400.0, 300.0, 200.0]
COMPLEXITY_ORDER = ["очень низкая", "низкая", "средняя", "высокая", "очень высокая"]

BASE_HOURS = 0.25      # накладные: открыть, настроиться, записать
MEDIA_HOURS = 0.5      # за каждый медиафайл
MAX_HOURS = 8.0        # верхний предел на один материал


def words_per_hour(complexity_level: str, complexity_score: float = 3.0) -> float:
    try:
        idx = COMPLEXITY_ORDER.index(complexity_level or "")
    except ValueError:
        idx = max(0, min(4, int(round((complexity_score or 3.0) - 1))))
    return WORDS_PER_HOUR[idx]


def estimate_hours(word_count: int, media_count: int,
                   complexity_level: str, complexity_score: float = 3.0) -> float:
    speed = words_per_hour(complexity_level, complexity_score)
    hours = BASE_HOURS + (word_count or 0) / speed + (media_count or 0) * MEDIA_HOURS
    return round(min(hours, MAX_HOURS), 2)


def estimate_material(m: Material) -> float:
    return estimate_hours(m.word_count or 0, m.media_count or 0,
                          m.complexity_level or "", m.complexity_score or 3.0)


def run_time_estimate(session) -> dict:
    """Заполняет estimated_time_hours для всех материалов. Возвращает сводку."""
    rows = []
    for m in session.query(Material).all():
        hours = estimate_material(m)
        m.estimated_time_hours = hours
        rows.append({"id": m.id, "title": m.title, "subject": m.subject,
                     "hours": hours})
    return {"materials": len(rows), "total_hours": round(sum(r["hours"] for r in rows), 2),
            "items": rows}
