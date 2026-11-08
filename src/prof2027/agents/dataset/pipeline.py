"""Pipeline датасета: признаки → prev/next → важность → описание → графики."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from prof2027.agents.dataset.charts import make_charts
from prof2027.agents.dataset.describe import describe, write_description
from prof2027.agents.dataset.features import build_frame, write_features_json
from prof2027.agents.dataset.importance import compute_importance, write_importance
from prof2027.agents.dataset.prevnext import assign_prev_next
from prof2027.db import get_engine, init_db, make_session


def run_dataset(cfg, catalogs, *, db=None, out_dir=None) -> dict:
    engine = get_engine(db or cfg.db_path)
    init_db(engine)
    Session = make_session(engine)

    out_dir = Path(out_dir) if out_dir else cfg.root / "reports" / "dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    with Session() as s:
        frame = build_frame(s, catalogs)
        features_updated = write_features_json(s, frame)
        prev_next_updated = assign_prev_next(s)
        s.commit()

    importance = compute_importance(frame)
    imp_file = write_importance(out_dir / "importance.md", importance, frame)
    stats = describe(frame)
    desc_file = write_description(out_dir / "description.md", stats, frame)
    charts = make_charts(frame, out_dir / "charts", catalogs)

    summary = {
        "materials": int(len(frame)),
        "subjects": stats["n_subjects"],
        "features_updated": features_updated,
        "prev_next_updated": prev_next_updated,
        "importance_file": str(imp_file),
        "description_file": str(desc_file),
        "charts": [str(p) for p in charts],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
