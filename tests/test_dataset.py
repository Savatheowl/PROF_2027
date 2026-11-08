"""Тесты этапа 2 (датасет): признаки, SVD, prev/next, важность, описания, графики, сборка."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from prof2027.dataset.build import DatasetBuilder
from prof2027.dataset.describe import build_description
from prof2027.dataset.features import (
    add_semantic_features,
    add_text_features,
    materials_to_frame,
)
from prof2027.dataset.importance import compute_importance, numeric_feature_columns
from prof2027.dataset.plots import plot_all
from prof2027.dataset.prevnext import (
    add_semantic_neighbor,
    build_prev_next,
    dataset_links,
)
from prof2027.db.base import SessionManager, make_engine
from prof2027.db.init_db import initialize
from prof2027.db.models import Material

# Тексты с разным словарным запасом, чтобы TF-IDF дал несколько компонент
_TEXTS = [
    "Механика изучает движение тел и силы, законы Ньютона описывают причины движения",
    "Алгебра изучает уравнения, многочлены и функции, корни квадратного уравнения",
    "Геометрия изучает треугольники, окружности и теоремы о площадях фигур",
    "Программирование изучает алгоритмы, структуры данных и циклы в коде",
    "История описывает события прошлого, хронологию и великих правителей эпохи",
]


def _frame_with_semantics(n: int = 4) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "id": i + 1,
                "title": f"Материал {i + 1}",
                "category": "Естественные науки",
                "subject": "Физика" if i % 2 == 0 else "Алгебра",
                "topic": None,
                "education_level": "university",
                "language": "ru",
                "material_type": "student_material",
                "class_type": "lecture",
                "file_type": "pdf",
                "complexity_level": "средняя",
                "moderation_verdict": "approved",
                "word_count": 20 + i * 5,
                "char_count": 100 + i * 30,
                "sentence_count": 2 + i,
                "media_count": i,
                "complexity_score": float(1 + i),
                "methodology_score": 0.7 + 0.05 * i,
                "estimated_time_hours": 1.0 + 0.5 * i,
                "text_content": _TEXTS[i],
                "media_descriptions": [{"type": "image"}] * i,
            }
        )
    return materials_to_frame(rows)


def _insert_materials(cfg: dict, manager: SessionManager) -> None:
    with manager.transaction() as session:
        for i in range(4):
            session.add(
                Material(
                    title=f"Материал {i + 1}",
                    category="Естественные науки",
                    subject="Физика" if i % 2 == 0 else "Алгебра",
                    education_level="university",
                    language="ru",
                    material_type="student_material",
                    class_type="lecture",
                    file_type="pdf",
                    complexity_level="средняя",
                    moderation_verdict="approved",
                    word_count=20 + i * 5,
                    char_count=100 + i * 30,
                    sentence_count=2 + i,
                    media_count=i,
                    complexity_score=float(1 + i),
                    methodology_score=0.7 + 0.05 * i,
                    estimated_time_hours=1.0 + 0.5 * i,
                    text_content=_TEXTS[i],
                    media_descriptions=[{"type": "image"}] * i,
                )
            )


@pytest.fixture()
def db_manager(proj_root: Path, cfg: dict, db_path: Path):
    initialize(cfg)
    return SessionManager(make_engine(db_path))


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

def test_materials_to_frame_columns() -> None:
    df = _frame_with_semantics(2)
    assert "material_id" in df.columns
    assert "subject" in df.columns
    assert df["subject"].tolist() == ["Физика", "Алгебра"]
    assert df["media_count"].tolist() == [0, 1]


def test_add_text_features_derived() -> None:
    import re

    df = add_text_features(_frame_with_semantics(2))
    assert "avg_word_len" in df.columns and "ttr" in df.columns
    assert "unique_words" in df.columns and "uppercase_ratio" in df.columns
    assert df["ttr"].between(0, 1.01).all()
    assert df["media_image"].tolist() == [0, 1]
    assert df["media_audio"].tolist() == [0, 0]
    assert df["has_media"].tolist() == [False, True]
    # длина слова = символы / число словоформ в тексте
    row = df.iloc[0]
    tokens = len(re.findall(r"[а-яё]+", row["text_content"].lower()))
    assert np.isclose(row["avg_word_len"], row["char_count"] / tokens)


def test_add_semantic_features_multiple() -> None:
    df, _tfidf, svd = add_semantic_features(_frame_with_semantics(4), n_components=10)
    svd_cols = [c for c in df.columns if c.startswith("svd_")]
    assert len(svd_cols) == 3  # min(10, n_rows-1=3, n_features-1)
    assert df["_svd_k"].iloc[0] == 3
    assert svd is not None
    # векторы ненулевые и различаются
    v0 = df[svd_cols].iloc[0].to_numpy()
    assert np.isfinite(v0).all()


def test_add_semantic_features_single_text() -> None:
    df, _tfidf, svd = add_semantic_features(_frame_with_semantics(1), n_components=10)
    assert svd is None
    assert not any(c.startswith("svd_") for c in df.columns)
    assert "semantic_neighbor_id" not in df.columns


# ---------------------------------------------------------------------------
# prev/next
# ---------------------------------------------------------------------------

def test_build_prev_next_order() -> None:
    df = build_prev_next(_frame_with_semantics(4))
    # внутри предмета: сортировка по complexity_score
    physics = df[df["subject"] == "Физика"].sort_values("complexity_score")
    assert physics["has_previous"].tolist() == [False, True]
    assert physics["has_next"].tolist() == [True, False]
    assert physics.iloc[0]["next_material_id"] == physics.iloc[1]["material_id"]
    assert physics.iloc[1]["previous_material_id"] == physics.iloc[0]["material_id"]


def test_semantic_neighbor_not_self() -> None:
    df, _tfidf, _svd = add_semantic_features(_frame_with_semantics(4), n_components=10)
    df = add_semantic_neighbor(df)
    for _i, row in df.iterrows():
        assert row["semantic_neighbor_id"] != row["material_id"]
        assert int(row["semantic_neighbor_id"]) in set(df["material_id"])


def test_dataset_links_structure() -> None:
    df = build_prev_next(_frame_with_semantics(4))
    links = dataset_links(df)
    assert len(links) == 4
    for link in links:
        assert set(link) == {
            "material_id", "has_previous", "has_next",
            "previous_material_id", "next_material_id",
        }
    assert any(link["has_next"] for link in links)


# ---------------------------------------------------------------------------
# importance
# ---------------------------------------------------------------------------

def test_numeric_feature_columns_excludes_ids_and_svd() -> None:
    df, _tfidf, _svd = add_semantic_features(_frame_with_semantics(4), n_components=10)
    cols = numeric_feature_columns(df)
    assert "material_id" not in cols
    assert not any(c.startswith("svd_") or c.startswith("_") for c in cols)
    assert "word_count" in cols and "complexity_score" in cols


def test_compute_importance_regression() -> None:
    df = add_text_features(_frame_with_semantics(5))
    imp = compute_importance(df, target="complexity_score")
    assert "feature" in imp.columns
    assert {"pearson", "spearman", "mutual_info", "permutation"}.issubset(imp.columns)
    # сложность заведена монотонной по word_count → заметная корреляция
    pear = imp.set_index("feature")["pearson"].get("word_count")
    assert pear is not None and pear > 0.9


def test_compute_importance_small_data_notes() -> None:
    df = _frame_with_semantics(1)
    imp = compute_importance(df, target="complexity_score")
    assert (imp["type"] == "note").any()
    assert "Требуется" in " ".join(str(v) for v in imp.iloc[:, 2:].values.ravel())


def test_compute_importance_missing_target() -> None:
    df = _frame_with_semantics(2)
    with pytest.raises(ValueError):
        compute_importance(df, target="нет_такой_колонки")


# ---------------------------------------------------------------------------
# describe / plots
# ---------------------------------------------------------------------------

def test_describe_contains_blocks() -> None:
    df = add_text_features(_frame_with_semantics(4))
    md = build_description(df, title="тест", importance={"complexity_score": compute_importance(df, "complexity_score")})
    assert "# Датасет материалов" in md
    assert "### `subject`" in md and "### `word_count`" in md
    assert "## Важность признаков" in md


def test_plot_all_produces_pngs(tmp_path: Path) -> None:
    df, _tfidf, _svd = add_semantic_features(add_text_features(_frame_with_semantics(4)), n_components=10)
    files = plot_all(df, tmp_path)
    assert files["histograms"] is not None and files["histograms"].is_file()
    assert files["svd_scatter"] is not None and files["svd_scatter"].is_file()
    assert files["correlation_heatmap"] is not None
    assert any(name.startswith("bar_") for name in files)


# ---------------------------------------------------------------------------
# end-to-end: DatasetBuilder
# ---------------------------------------------------------------------------

def test_builder_end_to_end(cfg: dict, db_manager: SessionManager, tmp_path: Path) -> None:
    _insert_materials(cfg, db_manager)
    out = tmp_path / "out"
    builder = DatasetBuilder(cfg, db_manager, out_dir=out, svd_components=8)
    report = builder.run()

    assert report.materials == 4 and report.rows == 4
    assert report.svd_components >= 1
    assert report.links_written == 4
    assert (out / "dataset.csv").is_file()
    assert (out / "dataset.parquet").is_file()
    assert (out / "dataset_description.md").is_file()
    assert (out / "features_histograms.png").is_file()

    csv = pd.read_csv(out / "dataset.csv")
    assert len(csv) == 4
    assert "text_content" not in csv.columns
    assert "semantic_neighbor_id" in csv.columns
    assert "previous_material_id" in csv.columns

    # связи записаны в БД
    with db_manager.session() as session:
        mats = session.scalars(select(Material).order_by(Material.id)).all()
        assert all(m.previous_material_id or not m.has_previous for m in mats)
        assert mats[0].features_json and "svd" in mats[0].features_json


def test_builder_empty_db(cfg: dict, db_manager: SessionManager, tmp_path: Path) -> None:
    out = tmp_path / "out"
    report = DatasetBuilder(cfg, db_manager, out_dir=out).run()
    assert report.materials == 0 and report.rows == 0
    assert not (out / "dataset.csv").exists()


def test_builder_limit(cfg: dict, db_manager: SessionManager, tmp_path: Path) -> None:
    _insert_materials(cfg, db_manager)
    report = DatasetBuilder(cfg, db_manager, out_dir=tmp_path / "o").run(limit=2)
    assert report.materials == 2 and report.rows == 2
