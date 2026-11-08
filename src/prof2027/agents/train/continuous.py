"""Continuous: проверка дрейфа данных против эталонов обученных версий."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from prof2027.agents.train.drift import check_drift
from prof2027.agents.train.tasks import TASKS, task_dataset
from prof2027.db import ModelVersion, get_engine, init_db, make_session


def run_continuous(cfg, catalogs, *, db=None, out_dir=None) -> dict:
    engine = get_engine(db or cfg.db_path)
    init_db(engine)
    Session = make_session(engine)
    out_dir = Path(out_dir) if out_dir else cfg.root / "reports" / "continuous"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    with Session() as s:
        versions = s.query(ModelVersion).filter_by(is_active=True).order_by(ModelVersion.id).all()
        for mv in versions:
            try:
                drift_ref = json.loads(mv.drift_json or "{}")
                ref = drift_ref.get("reference") or {}
            except Exception:
                ref = {}
            if not ref or mv.task not in TASKS:
                continue
            ds = task_dataset(s, catalogs, mv.task)
            if ds is None or len(ds["y"]) == 0:
                continue
            drift = check_drift(ds["X"], ref)
            mv.drift_json = json.dumps({**drift_ref, **drift, "checked_at":
                                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
                                       ensure_ascii=False)
            results[mv.id] = {"task": mv.task, "algorithm": mv.algorithm, "version": mv.version,
                              **drift}
        s.commit()

    (out_dir / "summary.json").write_text(
        json.dumps({"created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "versions": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"versions": results, "report_dir": str(out_dir)}
