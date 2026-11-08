"""Pipeline кластеризации: эмбеддинги → 3 стратегии → метрики → t-SNE → БД."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from prof2027.agents.cluster.embeddings import TextEmbedder
from prof2027.agents.cluster.metrics import cluster_metrics, tsne_plot
from prof2027.agents.cluster.strategies import STRATEGIES, assign
from prof2027.db import ClusterRun, Material, get_engine, init_db, make_session

EMBED_TEXT_MAX = 3000


def _load_records(session) -> list[dict]:
    records = []
    for m in session.query(Material).order_by(Material.id).all():
        records.append({
            "id": m.id,
            "title": m.title,
            "subject": m.subject,
            "complexity_score": float(m.complexity_score or 0.0),
            "text": f"{m.title or ''}. {(m.text_content or '')[:EMBED_TEXT_MAX]}",
        })
    for i, r in enumerate(records):
        r["pos"] = i
    return records


def run_cluster(cfg, catalogs, *, strategy: str = "all", group_size: int = 3,
                db=None, out_dir=None) -> dict:
    engine = get_engine(db or cfg.db_path)
    init_db(engine)
    Session = make_session(engine)

    out_dir = Path(out_dir) if out_dir else cfg.root / "reports" / "cluster"
    out_dir.mkdir(parents=True, exist_ok=True)

    with Session() as s:
        records = _load_records(s)
        if not records:
            return {"materials": 0, "strategies": {}, "message": "БД пуста: сначала loader run"}

        embedder = TextEmbedder()
        texts = [r["text"] for r in records]
        if len(records) == 1:
            X = np.zeros((1, 2))
        else:
            X = embedder.fit_transform(texts)

        strategies = [strategy] if strategy != "all" else STRATEGIES
        results = {}
        titles = [r["title"] for r in records]

        for st in strategies:
            labels = assign(st, records, group_size)
            metrics = cluster_metrics(X, labels)
            run = ClusterRun(
                strategy=st,
                algorithm="greedy" if st in ("parallel", "sequential") else "kmeans",
                params_json=json.dumps({"group_size": group_size}, ensure_ascii=False),
                metrics_json=json.dumps(metrics, ensure_ascii=False),
                n_clusters=metrics["n_clusters"],
            )
            s.add(run)
            s.flush()
            run_id = run.id

            # сохраняем метки в materials
            if st == "parallel":
                for r, lab in zip(records, labels.tolist()):
                    m = s.get(Material, r["id"])
                    if m:
                        m.cluster_parallel = int(lab)
            elif st == "sequential":
                for r, lab in zip(records, labels.tolist()):
                    m = s.get(Material, r["id"])
                    if m:
                        m.cluster_sequential = int(lab)

            chart = tsne_plot(X, labels, st, out_dir / "charts", titles)
            results[st] = {
                "run_id": run_id,
                "metrics": metrics,
                "chart": str(chart),
            }

        s.commit()

    (out_dir / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "materials": len(records),
        "strategies": results,
        "report_dir": str(out_dir),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
