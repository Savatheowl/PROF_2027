"""Тесты REST API: авторизация, чтение данных, запуски агентов."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from prof2027.api.app import DEFAULT_API_KEY, make_api
from prof2027.config import Config
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
    engine = get_engine(tmp_path / "api.db")
    init_db(engine)
    seed_catalogs(engine, cat)
    Session = make_session(engine)
    with Session() as s:
        reqs = {r.id: r for r in s.query(MethodologicalRequirement).all()}
        m = Material(title="Дроби", source_hash="api_h1", category="Математика и статистика",
                     subject="Алгебра", education_level="school", language="ru",
                     material_type="student_material", text_content="Текст. " * 30,
                     word_count=300, complexity_score=2.0, complexity_level="низкая",
                     methodology_score=0.8, moderation_verdict="approved",
                     estimated_time_hours=1.0,
                     features_json='{"word_count": 300, "complexity_score": 2.0}')
        s.add(m)
        s.flush()
        for rid in reqs:
            s.add(MaterialCompliance(material_id=m.id, requirement_id=rid, score=0.8,
                                     is_compliant=True))
        s.add(ModelVersion(task="complexity", algorithm="random_forest", version=1,
                           file_path="x.joblib", feature_schema_json='["word_count"]',
                           metrics_json='{"rmse": {"mean": 0.5}}', drift_json='{"level": "none"}'))
        s.commit()
    return engine


@pytest.fixture
def client(tmp_path, cfg, cat, monkeypatch):
    monkeypatch.setenv("PROF2027_DB_PATH", str(tmp_path / "api.db"))
    _seed(tmp_path, cfg, cat)
    from prof2027.api import app as api_mod

    api_mod._config.cache_clear()
    c = Config()
    app = make_api(c)
    return TestClient(app)


def test_health_open(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["materials"] == 1


def test_auth_required(client):
    r = client.get("/materials")
    assert r.status_code == 401
    r2 = client.get("/materials", headers={"X-API-Key": "wrong"})
    assert r2.status_code == 401


def test_auth_ok(client):
    r = client.get("/materials", headers={"X-API-Key": DEFAULT_API_KEY})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["items"][0]["subject"] == "Алгебра"


def test_catalogs(client):
    headers = {"X-API-Key": DEFAULT_API_KEY}
    r = client.get("/catalogs/subjects", headers=headers)
    assert r.status_code == 200
    assert len(r.json()) >= 1
    r = client.get("/catalogs/methodology", headers=headers)
    reqs = [t for c in r.json() for t in c["requirements"]]
    assert len(reqs) == 24


def test_material_detail(client):
    headers = {"X-API-Key": DEFAULT_API_KEY}
    r = client.get("/materials/1", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["title"] == "Дроби"
    assert len(body["compliance"]) == 24
    assert body["features"]["complexity_score"] == 2.0


def test_material_404(client):
    r = client.get("/materials/999", headers={"X-API-Key": DEFAULT_API_KEY})
    assert r.status_code == 404


def test_dataset_stats(client):
    r = client.get("/dataset/stats", headers={"X-API-Key": DEFAULT_API_KEY})
    assert r.status_code == 200
    assert r.json()["n_materials"] == 1


def test_dataset_importance(client):
    r = client.get("/dataset/importance", headers={"X-API-Key": DEFAULT_API_KEY})
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_cluster_runs_and_models(client):
    headers = {"X-API-Key": DEFAULT_API_KEY}
    r = client.get("/cluster/runs", headers=headers)
    assert r.status_code == 200
    r = client.get("/models", headers=headers)
    assert r.status_code == 200
    assert r.json()[0]["task"] == "complexity"
    r = client.get("/models/best", headers=headers)
    assert r.status_code == 200
    assert "complexity" in r.json()


def test_predict_time(client):
    r = client.post("/predict/time", json={"word_count": 500, "media_count": 1,
                                           "complexity_level": "средняя"},
                    headers={"X-API-Key": DEFAULT_API_KEY})
    assert r.status_code == 200
    assert r.json()["hours"] > 0


def test_predict_time_all(client):
    r = client.post("/predict/time/all", headers={"X-API-Key": DEFAULT_API_KEY})
    assert r.status_code == 200
    assert r.json()["materials"] == 1
    assert r.json()["total_hours"] > 0


def test_trajectory(client):
    r = client.post("/predict/trajectory",
                    json={"user": "t", "hours_per_day": 2.0, "deadline_days": 10,
                          "gantt": False},
                    headers={"X-API-Key": DEFAULT_API_KEY})
    assert r.status_code == 200
    body = r.json()
    assert body["total_days"] >= 1
    assert body["trajectory_id"] is not None


def test_trajectories_and_kpis(client):
    headers = {"X-API-Key": DEFAULT_API_KEY}
    r = client.get("/trajectories", headers=headers)
    assert r.status_code == 200
    r = client.get("/dashboard/kpis", headers=headers)
    assert r.status_code == 200
    assert r.json()["materials"] == 1


def test_loader_run_missing_manifest(client, tmp_path):
    r = client.post("/loader/run", json={"manifest": "nonexistent.yaml"},
                    headers={"X-API-Key": DEFAULT_API_KEY})
    assert r.status_code == 404
