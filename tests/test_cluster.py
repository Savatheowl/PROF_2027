"""Тесты агента-кластеризации: стратегии, метрики, pipeline, t-SNE."""

from pathlib import Path

import numpy as np

from prof2027.agents.cluster.embeddings import TextEmbedder
from prof2027.agents.cluster.metrics import cluster_metrics, tsne_plot
from prof2027.agents.cluster.pipeline import run_cluster
from prof2027.agents.cluster.strategies import (
    assign,
    labels_complexity,
    labels_parallel,
    labels_sequential,
)
from prof2027.db import (
    ClusterRun,
    Material,
    get_engine,
    init_db,
    make_session,
    seed_catalogs,
)


def _seed(tmp_path, cfg, cat):
    engine = get_engine(tmp_path / "cl.db")
    init_db(engine)
    seed_catalogs(engine, cat)
    Session = make_session(engine)
    rows = [
        # (subject, title, complexity)
        ("Алгебра", "Дроби", 1.0),
        ("Алгебра", "Квадратные уравнения", 3.0),
        ("Алгебра", "Интегралы", 5.0),
        ("Физика", "Механика", 2.0),
        ("Физика", "Оптика", 4.0),
        ("Информатика", "Алгоритмы", 3.0),
    ]
    with Session() as s:
        for i, (subj, title, cplx) in enumerate(rows):
            s.add(Material(
                title=title, source_hash=f"hash_cl_{i}", category=subj,
                subject=subj, text_content=f"Материал про {title}. " * 30,
                complexity_score=cplx, methodology_score=0.6,
            ))
        s.commit()
    return engine


def _records(engine):
    with make_session(engine)() as s:
        from prof2027.agents.cluster.pipeline import _load_records

        return _load_records(s)


def test_labels_parallel_no_subject_repeat(tmp_path, cfg, cat):
    from collections import defaultdict

    engine = _seed(tmp_path, cfg, cat)
    recs = _records(engine)
    labels = labels_parallel(recs)
    counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r, lab in zip(recs, labels.tolist()):
        counts[lab][r["subject"]] += 1
    # в каждой параллельной группе не более одного материала из каждого предмета
    for subj_counts in counts.values():
        assert all(c <= 1 for c in subj_counts.values())
    assert len(counts) >= 1


def test_labels_sequential_groups(tmp_path, cfg, cat):
    engine = _seed(tmp_path, cfg, cat)
    recs = _records(engine)
    labels = labels_sequential(recs, group_size=2)
    subj_labels: dict[str, set] = {}
    for r, lab in zip(recs, labels.tolist()):
        subj_labels.setdefault(r["subject"], set()).add(lab)
    # Алгебра: 3 материала по группам по 2 → 2 шага; Физика: 2 материала → 1 шаг
    assert len(subj_labels["Алгебра"]) == 2
    assert len(subj_labels["Физика"]) == 1


def test_labels_complexity_shape(tmp_path, cfg, cat):
    engine = _seed(tmp_path, cfg, cat)
    recs = _records(engine)
    labels = labels_complexity(recs)
    assert labels.shape == (6,)
    assert labels.max() < 5


def test_assign_unknown_raises():
    import pytest

    with pytest.raises(ValueError):
        assign("nope", [])


def test_cluster_metrics(tmp_path, cfg, cat):
    engine = _seed(tmp_path, cfg, cat)
    recs = _records(engine)
    labels = labels_parallel(recs)
    X = np.random.RandomState(0).rand(len(recs), 8)
    m = cluster_metrics(X, labels)
    assert m["n_clusters"] >= 2
    assert "silhouette" in m


def test_tsne_plot(tmp_path):
    X = np.random.RandomState(1).rand(12, 6)
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    fp = tsne_plot(X, labels, "parallel", tmp_path, ["a"] * 12)
    assert fp.exists()


def test_run_cluster_end_to_end(tmp_path, cfg, cat):
    engine = _seed(tmp_path, cfg, cat)
    out = tmp_path / "out"
    result = run_cluster(cfg, cat, strategy="all", db=tmp_path / "cl.db", out_dir=out)
    assert result["materials"] == 6
    assert set(result["strategies"]) == {"parallel", "sequential", "complexity"}
    for st, info in result["strategies"].items():
        assert info["metrics"]["n_clusters"] >= 1
        assert Path(info["chart"]).exists()
    with make_session(engine)() as s:
        runs = s.query(ClusterRun).all()
        assert len(runs) == 3
        m = s.query(Material).first()
        assert m.cluster_parallel is not None
        assert m.cluster_sequential is not None


def test_embedder_shape(tmp_path, cfg, cat):
    engine = _seed(tmp_path, cfg, cat)
    recs = _records(engine)
    emb = TextEmbedder(n_components=6)
    X = emb.fit_transform([r["text"] for r in recs])
    assert X.shape == (6, 6)
    X2 = emb.transform(["новый текст про дроби и уравнения"])
    assert X2.shape == (1, 6)
