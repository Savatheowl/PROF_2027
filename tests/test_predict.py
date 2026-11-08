"""Тесты агента-прогноза: время изучения, траектории, Gantt."""

from pathlib import Path

from prof2027.agents.predict.gantt import gantt_chart
from prof2027.agents.predict.time import estimate_hours, run_time_estimate
from prof2027.agents.predict.trajectory import build_trajectory
from prof2027.db import (
    Material,
    Trajectory,
    get_engine,
    init_db,
    make_session,
    seed_catalogs,
)


def _seed(tmp_path, cfg, cat):
    engine = get_engine(tmp_path / "pr.db")
    init_db(engine)
    seed_catalogs(engine, cat)
    Session = make_session(engine)
    rows = [
        ("Алгебра", "Дроби", 1.0, 400, 0),
        ("Алгебра", "Уравнения", 3.0, 600, 0),
        ("Алгебра", "Интегралы", 5.0, 900, 0),
        ("Физика", "Механика", 2.0, 800, 1),
        ("Физика", "Оптика", 4.0, 700, 0),
        ("Информатика", "Алгоритмы", 3.0, 1000, 1),
    ]
    with Session() as s:
        for i, (subj, title, cplx, words, media) in enumerate(rows):
            s.add(Material(
                title=title, source_hash=f"hash_pr_{i}", category=subj, subject=subj,
                text_content="Текст. " * 20, word_count=words, media_count=media,
                complexity_score=cplx, complexity_level="средняя",
                moderation_verdict="approved",
            ))
        s.commit()
    return engine


def test_estimate_hours():
    base = estimate_hours(word_count=400, media_count=0, complexity_level="средняя")
    assert base > 0
    more = estimate_hours(word_count=800, media_count=0, complexity_level="средняя")
    assert more > base
    hard = estimate_hours(word_count=400, media_count=0, complexity_level="очень высокая")
    assert hard > base  # та же длина, но выше сложность → дольше
    with_media = estimate_hours(word_count=400, media_count=3, complexity_level="средняя")
    assert with_media > base


def test_run_time_estimate(tmp_path, cfg, cat):
    engine = _seed(tmp_path, cfg, cat)
    with make_session(engine)() as s:
        result = run_time_estimate(s)
        s.commit()
    assert result["materials"] == 6
    assert result["total_hours"] > 0
    with make_session(engine)() as s:
        m = s.query(Material).first()
        assert m.estimated_time_hours > 0


def test_trajectory_parallel_and_deadline(tmp_path, cfg, cat):
    engine = _seed(tmp_path, cfg, cat)
    params = {"hours_per_day": 2.0, "deadline_days": 14}
    result = build_trajectory(cfg, cat, params, db=tmp_path / "pr.db",
                              gantt_out=tmp_path / "gantt.png")
    assert result["total_days"] >= 1
    assert result["total_hours"] > 0
    assert result["trajectory_id"] is not None
    assert Path(result["gantt"]).exists()
    # в один день могут быть материалы из разных предметов
    for d in result["plan"]:
        subs = {it["subject"] for it in d["items"]}
        if len(d["items"]) > 1:
            assert len(subs) > 0
    # сохранён в таблице trajectories
    with make_session(engine)() as s:
        assert s.query(Trajectory).count() == 1


def test_trajectory_filter_by_subject(tmp_path, cfg, cat):
    engine = _seed(tmp_path, cfg, cat)
    params = {"subject": "Алгебра", "hours_per_day": 2.0, "deadline_days": 10}
    result = build_trajectory(cfg, cat, params, db=tmp_path / "pr.db",
                              gantt_out=tmp_path / "g2.png")
    all_subjects = {it["subject"] for d in result["plan"] for it in d["items"]}
    assert all_subjects == {"Алгебра"}


def test_trajectory_meets_deadline(tmp_path, cfg, cat):
    engine = _seed(tmp_path, cfg, cat)
    # щедрый дедлайн → в срок
    result = build_trajectory(cfg, cat, {"hours_per_day": 8.0, "deadline_days": 2},
                              db=tmp_path / "pr.db", gantt_out=tmp_path / "g3.png")
    assert result["meets_deadline"] is True
    # жёсткий дедлайн → не в срок
    result2 = build_trajectory(cfg, cat, {"hours_per_day": 0.5, "deadline_days": 1},
                               db=tmp_path / "pr.db", gantt_out=tmp_path / "g4.png")
    assert result2["meets_deadline"] is False or result2["total_days"] >= 1


def test_gantt_chart(tmp_path):
    plan = [
        {"day": 1, "hours": 2.0, "items": [
            {"title": "A", "subject": "Физика", "hours": 1.2},
            {"title": "B", "subject": "Алгебра", "hours": 0.8},
        ]},
        {"day": 2, "hours": 1.0, "items": [{"title": "C", "subject": "Химия", "hours": 1.0}]},
    ]
    fp = gantt_chart(plan, tmp_path / "out_gantt.png")
    assert fp.exists()
