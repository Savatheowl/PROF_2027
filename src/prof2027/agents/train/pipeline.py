"""Pipeline обучения: задачи × алгоритмы, GroupKFold, версии, дрейф."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from prof2027.agents.train.drift import check_drift, reference_stats
from prof2027.agents.train.evaluate import evaluate
from prof2027.agents.train.models import models_for, save_model
from prof2027.agents.train.tasks import TASKS, task_dataset
from prof2027.db import ModelVersion, get_engine, init_db, make_session


def _data_hash(X) -> str:
    h = hashlib.sha256()
    h.update(X.round(4).to_csv(index=False).encode("utf-8"))
    return h.hexdigest()


def _metric_value(metrics: dict | None, kind: str) -> float:
    """Значение метрики для выбора лучшей модели (выше — лучше)."""
    if not metrics:
        return -1.0
    if kind == "classification":
        m = metrics.get("roc_auc") or metrics.get("accuracy")
    else:
        r2 = metrics.get("r2")
        if isinstance(r2, dict) and r2.get("mean") is not None:
            return float(r2["mean"])
        rmse = metrics.get("rmse")
        if isinstance(rmse, dict) and rmse.get("mean") is not None:
            return -float(rmse["mean"])  # ошибка: ниже — лучше
        return -1.0
    if isinstance(m, dict) and m.get("mean") is not None:
        return float(m["mean"])
    return float(m) if m is not None else -1.0


def run_train(cfg, catalogs, *, task: str = "all", db=None,
              artifacts_dir=None, out_dir=None) -> dict:
    engine = get_engine(db or cfg.db_path)
    init_db(engine)
    Session = make_session(engine)

    artifacts_dir = Path(artifacts_dir) if artifacts_dir else cfg.root / "data" / "artifacts" / "models"
    out_dir = Path(out_dir) if out_dir else cfg.root / "reports" / "train"
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [task] if task != "all" else list(TASKS)
    results: dict = {}

    with Session() as s:
        for tname in tasks:
            ds = task_dataset(s, catalogs, tname)
            entry = {"task": tname, "label": TASKS[tname]["label"], "kind": TASKS[tname]["kind"],
                     "models": [], "best": None}
            if ds is None or len(ds["y"]) < 4:
                entry["note"] = "недостаточно данных для обучения"
                results[tname] = entry
                continue

            best_score, best_info = -1.0, None
            for algo_name, model in models_for(ds["kind"]):
                metrics = evaluate(ds, model, fit_final=True)
                if metrics is None:
                    entry["models"].append({"algorithm": algo_name, "metrics": None,
                                            "note": "GroupKFold невозможен: мало групп/классов"})
                    continue
                version = 1 + s.query(ModelVersion).filter_by(task=tname, algorithm=algo_name).count()
                fpath = artifacts_dir / f"{tname}_{algo_name}_v{version}.joblib"
                save_model(model, fpath)
                ref = reference_stats(ds["X"])
                drift = {"reference": ref, "score": 0.0, "level": "none", "per_feature": {}}
                mv = ModelVersion(
                    task=tname, algorithm=algo_name, version=version,
                    file_path=str(fpath),
                    feature_schema_json=json.dumps(ds["features"], ensure_ascii=False),
                    metrics_json=json.dumps(metrics, ensure_ascii=False),
                    drift_json=json.dumps(drift, ensure_ascii=False),
                    data_hash=_data_hash(ds["X"]), is_active=True,
                )
                s.add(mv)
                s.flush()
                entry["models"].append({"algorithm": algo_name, "metrics": metrics,
                                        "version": version, "file": str(fpath)})

                score = _metric_value(metrics, ds["kind"])
                if score > best_score:
                    best_score = score
                    best_info = {"algorithm": algo_name, "version": version, "metrics": metrics,
                                 "file": str(fpath)}
            entry["best"] = best_info
            entry["best_key"] = "roc_auc" if ds["kind"] == "classification" else "r2"
            results[tname] = entry

        s.commit()

    summary = {"created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "tasks": results}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    return summary
