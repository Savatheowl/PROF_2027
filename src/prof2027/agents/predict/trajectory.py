"""Построение траектории обучения.

Учитывает: часы в день (hours_per_day), дедлайн (deadline_days), prev/next по
сложности внутри предмета и параллельное изучение разных предметов в один день.
Результат сохраняется в таблицу trajectories.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from prof2027.agents.predict.time import estimate_material
from prof2027.db import Material, Trajectory, get_engine, init_db, make_session


def _load_filtered(session, params: dict):
    q = session.query(Material)
    if params.get("subject"):
        q = q.filter(Material.subject == params["subject"])
    elif params.get("category"):
        q = q.filter(Material.category == params["category"])
    if params.get("level"):
        q = q.filter(Material.education_level == params["level"])
    return q.order_by(Material.id).all()


def build_trajectory(cfg, catalogs, params: dict, *, db=None,
                     gantt_out: Path | str | None = None) -> dict:
    engine = get_engine(db or cfg.db_path)
    init_db(engine)
    Session = make_session(engine)

    hours_per_day = float(params.get("hours_per_day", 2.0) or 2.0)
    deadline_days = int(params.get("deadline_days", 14) or 14)

    with Session() as s:
        mats = _load_filtered(s, params)
        chains: dict[str, list] = {}
        for m in mats:
            chains.setdefault(m.subject or "—", []).append(m)
        for chain in chains.values():
            chain.sort(key=lambda m: (float(m.complexity_score or 0.0), m.id))

        plan: list[dict] = []
        day = 1
        while any(chains.values()):
            day_hours = 0.0
            items = []
            active = list(chains.keys())
            # заполняем день: проходим по предметам, пока хоть один материал влезает
            while True:
                progress = False
                for subject in active:
                    chain = chains[subject]
                    if not chain:
                        continue
                    m = chain[0]
                    h = estimate_material(m)
                    if day_hours + h <= hours_per_day:
                        chain.pop(0)
                        items.append({"material_id": m.id, "title": m.title,
                                      "subject": m.subject, "hours": h})
                        day_hours += h
                        progress = True
                if not progress:
                    break
            if not items:
                # материал длиннее дневного лимита — планируем его отдельным днём
                for subject in active:
                    chain = chains[subject]
                    if chain:
                        m = chain.pop(0)
                        h = estimate_material(m)
                        items.append({"material_id": m.id, "title": m.title,
                                      "subject": m.subject, "hours": h})
                        day_hours += h
                        break
            plan.append({"day": day, "hours": round(day_hours, 2), "items": items})
            day += 1

        total_days = len(plan)
        total_hours = round(sum(d["hours"] for d in plan), 2)
        meets_deadline = total_days <= deadline_days

        out = {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "params": params,
            "total_days": total_days,
            "total_hours": total_hours,
            "deadline_days": deadline_days,
            "meets_deadline": meets_deadline,
            "plan": plan,
        }
        tr = Trajectory(
            user_ref=params.get("user", "demo"),
            input_json=json.dumps(params, ensure_ascii=False),
            output_json=json.dumps(out, ensure_ascii=False),
        )
        s.add(tr)
        s.commit()
        out["trajectory_id"] = tr.id

    if gantt_out:
        from prof2027.agents.predict.gantt import gantt_chart

        out["gantt"] = str(gantt_chart(plan, Path(gantt_out)))
    return out
