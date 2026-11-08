"""Тесты агента-отчётов: сбор данных и генерация отчётов по модулям А–Д."""

from pathlib import Path

from prof2027.agents.report.modules import collect
from prof2027.agents.report.pipeline import SECTIONS, TITLES, run_report, _section_a
from prof2027.db import (
    ClusterRun,
    Material,
    MaterialCompliance,
    MethodologicalRequirement,
    ModelVersion,
    Trajectory,
    get_engine,
    init_db,
    make_session,
    seed_catalogs,
)


def _seed(tmp_path, cfg, cat, monkeypatch):
    dbp = tmp_path / "rep.db"
    monkeypatch.setenv("PROF2027_DB_PATH", str(dbp))
    engine = get_engine(dbp)
    init_db(engine)
    seed_catalogs(engine, cat)
    Session = make_session(engine)
    with Session() as s:
        reqs = {r.id: r for r in s.query(MethodologicalRequirement).all()}
        m = Material(title="Дроби", source_hash="rep_h1", category="Математика и статистика",
                     subject="Алгебра", education_level="school", language="ru",
                     material_type="student_material", file_type=".pdf",
                     text_content="Текст. " * 30, word_count=300,
                     complexity_score=2.0, complexity_level="низкая",
                     methodology_score=0.8, moderation_verdict="approved",
                     estimated_time_hours=1.0)
        s.add(m)
        s.flush()
        for rid in reqs:
            s.add(MaterialCompliance(material_id=m.id, requirement_id=rid, score=0.8,
                                     is_compliant=True))
        s.add(ClusterRun(strategy="parallel", algorithm="greedy",
                         metrics_json='{"silhouette": 0.42}', n_clusters=2))
        s.add(ModelVersion(task="complexity", algorithm="random_forest", version=1,
                           file_path="m.joblib", feature_schema_json='["word_count"]',
                           metrics_json='{"rmse": {"mean": 0.5}}',
                           drift_json='{"level": "low"}'))
        s.add(Trajectory(user_ref="u", input_json="{}",
                         output_json='{"total_days": 2, "total_hours": 1.0,'
                                     ' "deadline_days": 10, "meets_deadline": true}'))
        s.commit()
    return engine


def test_collect(tmp_path, cfg, cat, monkeypatch):
    _seed(tmp_path, cfg, cat, monkeypatch)
    from prof2027.config import Config

    data = collect(Config())
    assert len(data["materials"]) == 1
    assert data["by_verdict"]["approved"] == 1
    assert data["by_file_type"][".pdf"] == 1
    assert len(data["compliance_means"]) == 24
    assert len(data["clusters"]) == 1
    assert len(data["models"]) == 1
    assert len(data["trajectories"]) == 1
    assert data["total_hours"] == 1.0


def test_run_report_all(tmp_path, cfg, cat, monkeypatch):
    _seed(tmp_path, cfg, cat, monkeypatch)
    from prof2027.config import Config

    rcfg = Config()
    out = tmp_path / "rep"
    result = run_report(rcfg, cat, out_dir=out)
    assert result["sections"] == ["A", "B", "C", "D", "E"]
    assert Path(result["report"]).exists()
    text = Path(result["report"]).read_text(encoding="utf-8")
    for s in SECTIONS:
        assert TITLES[s] in text
        assert (out / f"module_{s}.md").exists()
    # ключевое содержимое
    assert "Алгебра" in text
    assert "random_forest" in text
    assert "parallel" in text
    assert "X-API-Key" in text


def test_run_report_subset(tmp_path, cfg, cat, monkeypatch):
    _seed(tmp_path, cfg, cat, monkeypatch)
    from prof2027.config import Config

    result = run_report(Config(), cat, sections=["A", "D"], out_dir=tmp_path / "rep2")
    text = Path(result["report"]).read_text(encoding="utf-8")
    assert "А. Сбор и обработка" in text
    assert "Г. Интеграция" in text
    assert "В. Моделирование" not in text


def test_section_a_bullets(tmp_path, cfg, cat, monkeypatch):
    _seed(tmp_path, cfg, cat, monkeypatch)
    from prof2027.config import Config

    data = collect(Config())
    text = _section_a(cfg, data)
    assert "Вердикты модерации" in text
    assert "approved" in text
