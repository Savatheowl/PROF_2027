"""Сборка датасета: БД → таблица признаков → SVD → prev/next → важность → описания → графики.

Оркестратор `prof2027 dataset build` (см. docs/ARCHITECTURE.md, Шаг 2).
Результаты:
- `dataset.csv`, `dataset.parquet` — таблица признаков (без сырых текстов);
- `dataset_description.md` — описание атрибутов + важность признаков;
- PNG — гистограммы, bar-chart, SVD-scatter, тепловая карта корреляций.
- в БД записываются prev/next-поля и features_json.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import pandas as pd
from sqlalchemy import select

from prof2027.core.paths import ensure_dirs
from prof2027.dataset.describe import COLUMN_MEANINGS, build_description
from prof2027.dataset.features import (
    add_semantic_features,
    add_text_features,
    materials_to_frame,
)
from prof2027.dataset.importance import compute_importance
from prof2027.dataset.plots import plot_all
from prof2027.dataset.prevnext import add_semantic_neighbor, build_prev_next, dataset_links
from prof2027.db.base import SessionManager
from prof2027.db.models import Material

logger = logging.getLogger(__name__)

# Колонки, которые не попадают в файлы (сырой текст и JSON-описания медиа — в БД)
_HEAVY_COLUMNS = ("text_content", "media_descriptions")

# Целевые переменные для анализа важности признаков
DEFAULT_TARGETS = ("complexity_score", "methodology_score")


@dataclass
class DatasetReport:
    """Итоги одного запуска сборки датасета."""

    materials: int = 0
    rows: int = 0
    columns: int = 0
    svd_components: int = 0
    links_written: int = 0
    files: dict[str, Path | None] = field(default_factory=dict)
    importance: dict[str, pd.DataFrame] = field(default_factory=dict)


class DatasetBuilder:
    """Оркестратор этапа 2 (модуль датасета)."""

    def __init__(
        self,
        cfg: dict,
        manager: SessionManager,
        *,
        out_dir: Path | None = None,
        svd_components: int = 25,
        targets: tuple[str, ...] = DEFAULT_TARGETS,
    ) -> None:
        self.cfg = cfg
        self.manager = manager
        self.svd_components = max(2, int(svd_components))
        self.targets = targets
        self.out_dir = out_dir or (ensure_dirs()["artifacts_dir"] / "dataset")

    # ------------------------------------------------------------------
    # Шаг 1: выгрузка материалов из БД
    # ------------------------------------------------------------------
    def _load_materials(self, limit: int | None = None) -> list[dict]:
        with self.manager.session() as session:
            stmt = select(Material).order_by(Material.id)
            if limit:
                stmt = stmt.limit(int(limit))
            return [_material_to_dict(m) for m in session.scalars(stmt).all()]

    # ------------------------------------------------------------------
    # Сборка таблицы признаков
    # ------------------------------------------------------------------
    def _build_frame(self, materials: list[dict]) -> pd.DataFrame:
        df = materials_to_frame(materials)
        if df.empty:
            return df
        df = add_text_features(df)
        df, _tfidf, _svd = add_semantic_features(df, n_components=self.svd_components)
        df = build_prev_next(df)
        df = add_semantic_neighbor(df)
        return df

    # ------------------------------------------------------------------
    # Сохранение файлов
    # ------------------------------------------------------------------
    def _view(self, df: pd.DataFrame) -> pd.DataFrame:
        """Таблица признаков без сырых текстов и JSON-медиа (что попадает в файлы)."""
        return df.drop(columns=[c for c in _HEAVY_COLUMNS if c in df.columns]).copy()

    def _save_frame(self, df: pd.DataFrame) -> dict[str, Path]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        view = self._view(df)

        csv_path = self.out_dir / "dataset.csv"
        view.to_csv(csv_path, index=False, encoding="utf-8-sig")

        parquet_path = self.out_dir / "dataset.parquet"
        try:
            view.to_parquet(parquet_path, index=False)
        except Exception as exc:  # pragma: no cover — fallback при отсутствии движка
            logger.warning("Запись parquet не удалась: %s", exc)
            parquet_path = None

        return {"csv": csv_path, "parquet": parquet_path}

    # ------------------------------------------------------------------
    # Запись prev/next и features_json в БД
    # ------------------------------------------------------------------
    def _write_links(self, df: pd.DataFrame) -> int:
        links = dataset_links(df)
        if not links:
            return 0

        svd_cols = [c for c in df.columns if c.startswith("svd_")]
        count = 0
        with self.manager.transaction() as session:
            for link in links:
                mat = session.get(Material, link["material_id"])
                if mat is None:
                    continue
                mat.has_previous = link["has_previous"]
                mat.has_next = link["has_next"]
                mat.previous_material_id = link["previous_material_id"]
                mat.next_material_id = link["next_material_id"]

                row = df.loc[df["material_id"] == link["material_id"]].iloc[0]
                features = dict(mat.features_json or {})
                features["svd_components"] = len(svd_cols)
                if svd_cols:
                    features["svd"] = [float(row[c]) for c in svd_cols]
                for feat in ("ttr", "avg_word_len", "avg_sentence_len_words", "uppercase_ratio"):
                    value = row.get(feat)
                    if pd.notna(value):
                        features[feat] = float(value)
                mat.features_json = features
                count += 1
        return count

    # ------------------------------------------------------------------
    # Запуск
    # ------------------------------------------------------------------
    def run(self, *, limit: int | None = None) -> DatasetReport:
        materials = self._load_materials(limit=limit)
        logger.info("Выгружено материалов: %d", len(materials))
        if not materials:
            return DatasetReport(materials=0)

        df = self._build_frame(materials)
        svd_k = int(df.get("_svd_k", pd.Series(dtype=int)).max()) if "_svd_k" in df.columns else 0
        view = self._view(df)

        importance: dict[str, pd.DataFrame] = {}
        for target in self.targets:
            if target in df.columns:
                importance[target] = compute_importance(df, target)

        description = build_description(
            view,
            title=f"{len(view)} материалов, {len(view.columns)} атрибутов",
            meanings=COLUMN_MEANINGS,
            importance=importance,
        )

        files: dict[str, Path | None] = {}
        files.update(self._save_frame(df))
        files["description"] = self._save_text(description, "dataset_description.md")
        files.update(plot_all(view, self.out_dir))

        links_written = self._write_links(df)

        report = DatasetReport(
            materials=len(materials),
            rows=len(df),
            columns=len(df.columns),
            svd_components=svd_k,
            links_written=links_written,
            files=files,
            importance=importance,
        )
        logger.info("Датасет собран: %d строк, %d признаков, SVD k=%d", len(df), len(df.columns), svd_k)
        return report

    def _save_text(self, text: str, name: str) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / name
        path.write_text(text, encoding="utf-8")
        return path


def _material_to_dict(m: Material) -> dict:
    return {
        "id": m.id,
        "title": m.title,
        "category": m.category,
        "subject": m.subject,
        "topic": m.topic,
        "education_level": m.education_level,
        "language": m.language,
        "material_type": m.material_type,
        "class_type": m.class_type,
        "file_type": m.file_type,
        "complexity_level": m.complexity_level,
        "moderation_verdict": m.moderation_verdict,
        "word_count": m.word_count,
        "char_count": m.char_count,
        "sentence_count": m.sentence_count,
        "media_count": m.media_count,
        "complexity_score": m.complexity_score,
        "methodology_score": m.methodology_score,
        "estimated_time_hours": m.estimated_time_hours,
        "text_content": m.text_content,
        "media_descriptions": m.media_descriptions,
    }
