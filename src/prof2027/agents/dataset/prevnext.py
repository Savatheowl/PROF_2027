"""prev/next: порядок изучения материалов.

Внутри одного предмета материалы сортируются по сложности (complexity_score),
затем по объёму (word_count). Параллельное изучение возможно между разными
предметами — внутри предмета всегда последовательно.
"""

from __future__ import annotations

from prof2027.db import Material


def assign_prev_next(session) -> int:
    """Вычисляет has_previous/has_next и prev/next id для всех материалов.

    Возвращает число обновлённых материалов.
    """
    mats = session.query(Material).order_by(Material.id).all()
    by_subject: dict[str, list[Material]] = {}
    for m in mats:
        by_subject.setdefault(m.subject or "—", []).append(m)

    updated = 0
    for subject_mats in by_subject.values():
        ordered = sorted(
            subject_mats,
            key=lambda m: (float(m.complexity_score or 0.0), float(m.word_count or 0.0), m.id),
        )
        for pos, m in enumerate(ordered):
            prev_id = ordered[pos - 1].id if pos > 0 else None
            next_id = ordered[pos + 1].id if pos < len(ordered) - 1 else None
            m.previous_material_id = prev_id
            m.next_material_id = next_id
            m.has_previous = int(prev_id is not None)
            m.has_next = int(next_id is not None)
            updated += 1
    return updated


def chain(session, subject: str):
    """Возвращает цепочку id материалов предмета в порядке изучения."""
    mats = session.query(Material).filter(Material.subject == subject).all()
    ordered = sorted(mats, key=lambda m: (float(m.complexity_score or 0.0), float(m.word_count or 0.0), m.id))
    return [(m.id, m.title, m.complexity_score, m.complexity_level) for m in ordered]
