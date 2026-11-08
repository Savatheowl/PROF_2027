"""Формирование ответов бота: команды, статус, материалы, траектория, загрузка файлов."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from prof2027.agents.predict.trajectory import build_trajectory
from prof2027.db import Material, get_engine, init_db, make_session

MAX_MATERIALS_IN_LIST = 15


def _session(cfg):
    engine = get_engine(cfg.db_path)
    init_db(engine)
    return make_session(engine)


def build_start_text(cfg) -> str:
    return (
        "Привет! Я бот PROF_2027 — помогаю управлять учебными материалами.\n\n"
        "Команды:\n"
        "/help — список команд\n"
        "/status — статистика по материалам\n"
        "/materials — список материалов\n"
        "/subjects — предметы в базе\n"
        "/trajectory [часы_в_день] [дедлайн_дней] — построить план обучения\n"
        "\nПришли файл (PDF/DOCX/PPTX/...), чтобы добавить материал."
    )


def build_help_text() -> str:
    return (
        "/help — эта справка\n"
        "/status — сколько материалов, предметов, по вердиктам\n"
        "/materials — список материалов (заголовок, предмет, сложность, вердикт)\n"
        "/subjects — предметы и число материалов\n"
        "/trajectory 2 10 — траектория: 2 ч/день, дедлайн 10 дней\n"
        "\nФайлы: отправьте документ — бот прогонит загрузчик (OCR, классификацию d1-2B, "
        "аннотацию LaciaSUM, модерацию) и сохранит материал."
    )


def build_status_text(cfg) -> str:
    with _session(cfg)() as s:
        mats = s.query(Material).all()
    n = len(mats)
    subjects = len({m.subject for m in mats})
    verdicts: dict[str, int] = {}
    for m in mats:
        verdicts[m.moderation_verdict] = verdicts.get(m.moderation_verdict, 0) + 1
    hours = round(sum(float(m.estimated_time_hours or 0.0) for m in mats), 2)
    if n == 0:
        return "База пуста. Запустите загрузку файлами или prof2027 loader run."
    parts = ", ".join(f"{k}: {v}" for k, v in sorted(verdicts.items()))
    return f"Материалов: {n}\nПредметов: {subjects}\nВердикты: {parts}\nЧасов изучения: {hours}"


def build_materials_text(cfg, limit: int = MAX_MATERIALS_IN_LIST) -> str:
    with _session(cfg)() as s:
        rows = s.query(Material).order_by(Material.id).all()
    if not rows:
        return "Материалов нет."
    lines = []
    for m in rows[:limit]:
        cplx = m.complexity_score if m.complexity_score is not None else "—"
        lines.append(f"#{m.id} {m.title} · {m.subject} · сложность {cplx} · {m.moderation_verdict}")
    extra = f"\n...и ещё {len(rows) - limit}" if len(rows) > limit else ""
    return "\n".join(lines) + extra


def build_subjects_text(cfg) -> str:
    with _session(cfg)() as s:
        rows = s.query(Material).all()
    counts: dict[str, int] = {}
    for m in rows:
        counts[m.subject or "—"] = counts.get(m.subject or "—", 0) + 1
    if not counts:
        return "Предметов нет."
    return "\n".join(f"{k}: {v}" for k, v in sorted(counts.items()))


def build_trajectory_text(cfg, catalogs, hours_per_day: float = 2.0,
                          deadline_days: int = 14, with_gantt: bool = True) -> str:
    params = {"user": "telegram", "hours_per_day": hours_per_day,
              "deadline_days": deadline_days}
    gantt_out = None
    if with_gantt:
        gantt_out = cfg.root / "reports" / "predict" / \
            f"bot_gantt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    result = build_trajectory(cfg, catalogs, params, gantt_out=gantt_out)
    status = "в срок" if result["meets_deadline"] else "НЕ в срок"
    text = (
        f"План обучения ({result['total_days']} дн., {result['total_hours']} ч., дедлайн "
        f"{result['deadline_days']} дн.) — {status}\n\n"
    )
    for d in result["plan"]:
        items = ", ".join(f"{it['title']} ({it['subject']}, {it['hours']} ч.)"
                          for it in d["items"])
        text += f"День {d['day']}: {items or '—'}\n"
    return text, result


def build_loader_result_text(summary: dict) -> str:
    s = summary["stats"]
    lines = [f"Загрузка завершена: принято {s.get('ok', 0)}, ошибок {s.get('error', 0)}"]
    details = [r for r in summary["results"] if r["status"] in ("ok", "error")]
    for r in details[:10]:
        lines.append(f"• {r['source']}: {r['status']} — {r.get('reason', '')}")
    lines.append(f"Отчёт: {summary['report']}")
    return "\n".join(lines)
