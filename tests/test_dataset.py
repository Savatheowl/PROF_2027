"""Тесты агента-датасета: признаки, prev/next, важность, описание, графики."""

from pathlib import Path

from prof2027.agents.dataset.charts import make_charts
from prof2027.agents.dataset.describe import describe
from prof2027.agents.dataset.features import build_frame, numeric_columns, write_features_json
from prof2027.agents.dataset.importance import compute_importance
from prof2027.agents.dataset.prevnext import assign_prev_next, chain
from prof2027.agents.dataset.pipeline import run_dataset
from prof2027.db import (
    Material,
    MaterialCompliance,
    MethodologicalRequirement,
    get_engine,
    init_db,
    make_session,
    seed_catalogs,
)


def _seed(tmp_path, cfg, cat, n=3, subject="Алгебра", complexities=(5.0, 1.0, 3.0)):
    engine = get_engine(tmp_path / "dset.db")
    init_db(engine)
    seed_catalogs(engine, cat)
    Session = make_session(engine)
    with Session() as s:
        reqs = {r.id: r for r in s.query(MethodologicalRequirement).all()}
        for i, cplx in enumerate(complexities):
            m = Material(
                title=f"Материал {i + 1}",
                source_hash=f"test{i}_hash",
                category="Математика и статистика",
                subject=subject,
                education_level="school",
                language="ru",
                material_type="student_material",
                class_type="lesson",
                text_content="Текст материала. " * 20,
                annotation="Аннотация.",
                word_count=100 + i * 10,
                char_count=600 + i * 60,
                sentence_count=5 + i,
                media_count=i,
                complexity_score=cplx,
                complexity_level="средняя",
                methodology_score=0.6,
                moderation_verdict="approved",
            )
            s.add(m)
            s.flush()
            for rid, req in reqs.items():
                s.add(MaterialCompliance(material_id=m.id, requirement_id=rid,
                                         score=0.7, is_compliant=True))
        s.commit()
    return engine


def test_build_frame_columns(tmp_path, cfg, cat):
    engine = _seed(tmp_path, cfg, cat)
    with make_session(engine)() as s:
        frame = build_frame(s, cat)
    assert len(frame) == 3
    assert {"id", "subject", "complexity_score", "compliant_share"}.issubset(frame.columns)
    assert len([c for c in frame.columns if c.startswith("compliance_")]) == 24
    assert (frame["compliant_share"] == 1.0).all()
    assert numeric_columns(frame)[0] == "word_count"


def test_prev_next_ordering(tmp_path, cfg, cat):
    engine = _seed(tmp_path, cfg, cat)
    with make_session(engine)() as s:
        updated = assign_prev_next(s)
        s.commit()
        assert updated == 3
        ch = chain(s, "Алгебра")
        # порядок по сложности: 1.0, 3.0, 5.0
        assert [c[2] for c in ch] == [1.0, 3.0, 5.0]
        first = s.get(Material, ch[0][0])
        last = s.get(Material, ch[-1][0])
        assert first.has_previous == 0 and first.has_next == 1
        assert first.next_material_id == ch[1][0]
        assert last.has_previous == 1 and last.has_next == 0
        assert last.previous_material_id == ch[1][0]


def test_prev_next_separate_subjects(tmp_path, cfg, cat):
    engine = _seed(tmp_path, cfg, cat, n=2, subject="Алгебра", complexities=(2.0, 1.0))
    with make_session(engine)() as s:
        m = Material(title="Физика", source_hash="phys_hash",
                     category="Естественные науки", subject="Физика",
                     text_content="x", complexity_score=9.0)
        s.add(m)
        s.commit()
        assign_prev_next(s)
        s.commit()
        ch_alg = chain(s, "Алгебра")
        ch_phys = chain(s, "Физика")
    assert len(ch_alg) == 2 and len(ch_phys) == 1


def test_importance_and_describe(tmp_path, cfg, cat):
    engine = _seed(tmp_path, cfg, cat)
    with make_session(engine)() as s:
        frame = build_frame(s, cat)
    imp = compute_importance(frame)
    assert imp and imp[0]["importance"] >= imp[-1]["importance"]
    stats = describe(frame)
    assert stats["n_materials"] == 3
    assert stats["n_subjects"] == 1
    assert stats["by_subject"][0]["value"] == "Алгебра"


def test_run_dataset_end_to_end(tmp_path, cfg, cat):
    engine = _seed(tmp_path, cfg, cat)
    out = tmp_path / "out"
    summary = run_dataset(cfg, cat, db=tmp_path / "dset.db", out_dir=out)
    assert summary["materials"] == 3
    assert summary["features_updated"] == 3
    assert summary["prev_next_updated"] == 3
    assert (out / "importance.md").exists()
    assert (out / "description.md").exists()
    assert len(summary["charts"]) >= 4
    for p in summary["charts"]:
        assert Path(p).exists()
    # features_json записан
    with make_session(engine)() as s:
        m = s.query(Material).first()
        assert "complexity_score" in m.features_json


def test_charts_empty_frame(tmp_path, cfg, cat):
    import pandas as pd

    empty = pd.DataFrame(columns=["complexity_score", "methodology_score", "subject",
                                  "education_level", "compliance_1"])
    assert make_charts(empty, tmp_path / "charts", cat) == []
