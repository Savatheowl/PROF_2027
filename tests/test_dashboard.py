"""Тесты дашборда: функции данных и сборка Dash-приложения."""

from pathlib import Path

import pytest

from prof2027.dashboard import data as ddata
from prof2027.dashboard.app import make_dashboard
from prof2027.db import (
    ClusterRun,
    Material,
    MaterialCompliance,
    MethodologicalRequirement,
    ModelVersion,
    Trajectory,
    User,
    get_engine,
    init_db,
    make_session,
    seed_catalogs,
)


def _seed(tmp_path, cfg, cat):
    engine = get_engine(tmp_path / "db.db")
    init_db(engine)
    seed_catalogs(engine, cat)
    Session = make_session(engine)
    with Session() as s:
        s.add(User(username="admin", password_hash="x" * 64, role="admin"))
        m = Material(title="Дроби", source_hash="h1", subject="Алгебра",
                     category="Математика и статистика", education_level="school",
                     language="ru", material_type="student_material",
                     text_content="Текст. " * 30, word_count=300,
                     complexity_score=2.0, methodology_score=0.7,
                     moderation_verdict="approved", estimated_time_hours=1.0)
        s.add(m)
        s.flush()
        reqs = {r.id: r for r in s.query(MethodologicalRequirement).all()}
        for rid in reqs:
            s.add(MaterialCompliance(material_id=m.id, requirement_id=rid, score=0.8,
                                     is_compliant=True))
        s.add(ClusterRun(strategy="parallel", algorithm="greedy",
                         metrics_json='{"silhouette": 0.4}', n_clusters=2))
        s.add(ModelVersion(task="complexity", algorithm="random_forest", version=1,
                           feature_schema_json='["word_count"]',
                           metrics_json='{"rmse": {"mean": 0.5}}',
                           drift_json='{"level": "none"}'))
        s.add(Trajectory(user_ref="u", input_json="{}", output_json='{"total_days": 2}'))
        s.commit()
    return engine


def test_data_kpis(tmp_path, cfg, cat):
    engine = _seed(tmp_path, cfg, cat)
    with make_session(engine)() as s:
        k = ddata.kpis(s)
    assert k["materials"] == 1
    assert k["subjects"] == 1
    assert k["approved"] == 1
    assert k["total_hours"] == 1.0


def test_data_tables(tmp_path, cfg, cat):
    engine = _seed(tmp_path, cfg, cat)
    with make_session(engine)() as s:
        assert len(ddata.materials_rows(s)) == 1
        assert ddata.subject_counts(s)[0]["count"] == 1
        assert len(ddata.compliance_means(s)) == 24
        assert ddata.cluster_runs(s)[0]["strategy"] == "parallel"
        assert ddata.model_versions(s)[0]["task"] == "complexity"
        assert ddata.trajectories(s)[0]["days"] == 2
        assert ddata.users(s)[0]["username"] == "admin"
    assert isinstance(ddata.report_files(cfg), list)


def test_make_dashboard_builds(cfg):
    app = make_dashboard(cfg)
    assert app.server is not None
    # главный layout содержит Location и страницу
    children = app.layout.children
    assert any(getattr(c, "id", None) == "url" for c in children)
    assert any(getattr(c, "id", None) == "page" for c in children)


def test_role_tabs():
    from prof2027.dashboard.app import ROLE_TABS

    assert set(ROLE_TABS["admin"]) == {"overview", "materials", "subjects", "dataset",
                                       "cluster", "models", "drift", "trajectories",
                                       "reports", "admin"}
    assert len(ROLE_TABS["analyst"]) == 9
    assert len(ROLE_TABS["viewer"]) == 5
