"""Тесты агента-обучения: задачи, GroupKFold, версии, дрейф."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from prof2027.agents.train.drift import check_drift, reference_stats
from prof2027.agents.train.evaluate import evaluate
from prof2027.agents.train.models import models_for
from prof2027.agents.train.pipeline import run_train
from prof2027.agents.train.continuous import run_continuous
from prof2027.agents.train.tasks import TASKS, task_dataset
from prof2027.db import (
    Material,
    MaterialCompliance,
    MethodologicalRequirement,
    ModelVersion,
    get_engine,
    init_db,
    make_session,
    seed_catalogs,
)


def _seed(tmp_path, cfg, cat):
    engine = get_engine(tmp_path / "tr.db")
    init_db(engine)
    seed_catalogs(engine, cat)
    Session = make_session(engine)
    rows = [
        ("Алгебра", "Дроби", 1.0, "approved", 0),
        ("Алгебра", "Уравнения", 3.0, "needs_review", 0),
        ("Алгебра", "Интегралы", 5.0, "approved", 1),
        ("Физика", "Механика", 2.0, "approved", 2),
        ("Физика", "Оптика", 4.0, "needs_review", 3),
        ("Информатика", "Алгоритмы", 3.0, "approved", 1),
    ]
    with Session() as s:
        reqs = {r.id: r for r in s.query(MethodologicalRequirement).all()}
        for i, (subj, title, cplx, verdict, cl_seq) in enumerate(rows):
            m = Material(
                title=title, source_hash=f"hash_tr_{i}", category=subj, subject=subj,
                text_content=f"Материал про {title}. " * 30,
                word_count=120 + i * 7, char_count=700 + i * 40,
                sentence_count=6 + i, media_count=i % 2,
                complexity_score=cplx, complexity_level="средняя",
                methodology_score=0.6, moderation_verdict=verdict,
                cluster_sequential=cl_seq,
            )
            s.add(m)
            s.flush()
            for rid in reqs:
                s.add(MaterialCompliance(material_id=m.id, requirement_id=rid,
                                         score=0.55 + (i % 3) * 0.15, is_compliant=True))
        s.commit()
    return engine


def _ds(engine, cat, task):
    with make_session(engine)() as s:
        return task_dataset(s, cat, task)


def test_task_dataset_complexity(tmp_path, cfg, cat):
    engine = _seed(tmp_path, cfg, cat)
    ds = _ds(engine, cat, "complexity")
    assert ds is not None
    assert ds["kind"] == "regression"
    assert "complexity_score" not in ds["features"]  # целевая метка не в признаках
    assert len(ds["X"]) == 6
    assert ds["groups"].nunique() == 3


def test_task_dataset_methodology(tmp_path, cfg, cat):
    engine = _seed(tmp_path, cfg, cat)
    ds = _ds(engine, cat, "methodology")
    assert ds["kind"] == "classification"
    assert ds["y"].nunique() >= 2
    assert "compliance_1" not in ds["features"]  # нет утечки через compliance


def test_task_dataset_cluster(tmp_path, cfg, cat):
    engine = _seed(tmp_path, cfg, cat)
    ds = _ds(engine, cat, "cluster")
    assert ds["kind"] == "classification"
    assert ds["y"].nunique() >= 2


def test_models_registry():
    assert len(models_for("regression")) == 3
    assert len(models_for("classification")) == 3


def test_evaluate_groupkfold(tmp_path, cfg, cat):
    engine = _seed(tmp_path, cfg, cat)
    ds = _ds(engine, cat, "complexity")
    model = models_for("regression")[0][1]
    metrics = evaluate(ds, model)
    assert metrics is not None
    assert "rmse" in metrics and "r2" in metrics
    assert metrics["rmse"]["mean"] >= 0


def test_drift_functions():
    X = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    ref = reference_stats(X)
    assert ref["a"]["mean"] == 2.0
    d = check_drift(X, ref)
    assert d["level"] == "none"
    X2 = pd.DataFrame({"a": [5, 6, 7], "b": [10, 20, 30]})
    d2 = check_drift(X2, ref)
    assert d2["score"] > 2.0 and d2["level"] == "high"


def test_run_train_end_to_end(tmp_path, cfg, cat):
    engine = _seed(tmp_path, cfg, cat)
    summary = run_train(cfg, cat, task="all", db=tmp_path / "tr.db",
                        artifacts_dir=tmp_path / "art", out_dir=tmp_path / "out")
    tasks = summary["tasks"]
    assert set(tasks) == {"complexity", "methodology", "cluster"}
    for tname, entry in tasks.items():
        trained = [m for m in entry["models"] if m.get("metrics")]
        assert len(trained) >= 2, f"{tname}: мало обученных моделей: {trained}"
        assert entry["best"] is not None
    with make_session(engine)() as s:
        versions = s.query(ModelVersion).all()
        assert len(versions) >= 6
        for mv in versions:
            assert Path(mv.file_path).exists()
            feats = json.loads(mv.feature_schema_json)
            assert feats
            assert json.loads(mv.metrics_json)
    files = list((tmp_path / "art").glob("*.joblib"))
    assert len(files) >= 6


def test_run_continuous(tmp_path, cfg, cat):
    engine = _seed(tmp_path, cfg, cat)
    run_train(cfg, cat, task="complexity", db=tmp_path / "tr.db",
              artifacts_dir=tmp_path / "art2", out_dir=tmp_path / "out2")
    result = run_continuous(cfg, cat, db=tmp_path / "tr.db", out_dir=tmp_path / "cont")
    assert len(result["versions"]) >= 1
    for v in result["versions"].values():
        assert v["level"] in ("none", "low", "high")
    with make_session(engine)() as s:
        mv = s.query(ModelVersion).first()
        dj = json.loads(mv.drift_json)
        assert "checked_at" in dj
