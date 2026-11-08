"""Тесты конвейера загрузчика: фейковые AI-адаптеры, dedup, skip-ai, manifest, отчёт."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select

from prof2027.agents.loader.pipeline import (
    ST_DUPLICATE,
    ST_FAILED,
    ST_OK,
    ST_REJECTED_SIGNATURE,
    ST_SKIPPED_TOO_LARGE,
    ST_SKIPPED_UNSUPPORTED,
    LoaderPipeline,
)
from prof2027.agents.loader.report import render_markdown
from prof2027.ai.classifier import ClassificationResult, ComplianceResult
from prof2027.db.base import SessionManager, make_engine
from prof2027.db.init_db import initialize
from prof2027.db.models import Material, MaterialCompliance
from prof2027.domain import catalogs as cat
from prof2027.domain.enums import (
    ComplexityLevel,
    EducationLevel,
    Language,
    MaterialType,
    ModerationVerdict,
)


class FakeClassifier:
    """Замена d1-2B для тестов: детерминированные ответы без загрузки модели."""

    def __init__(self, *, score: float = 0.85, verdict: str = "approved") -> None:
        self.score = score
        self.verdict = verdict

    def classify(self, text: str, title: str, hint: str | None = None) -> ClassificationResult:
        return ClassificationResult(
            category="Естественные науки",
            subject="Физика",
            topic=title,
            education_level=EducationLevel.UNIVERSITY.value,
            language=Language.RU.value,
            material_type=MaterialType.STUDENT_MATERIAL.value,
            class_type="lecture",
            complexity_score=3.0,
            complexity_level=ComplexityLevel.MEDIUM.value,
            confidences={"category": 0.92, "subject": 0.88, "complexity": 0.7},
        )

    def moderate(
        self,
        text: str,
        title: str,
        hint: str | None = None,
        thresholds: dict | None = None,
    ) -> ComplianceResult:
        requirements = cat.load_methodology()
        scores = {r.text: self.score for r in requirements}
        compliant = {r.text: self.score >= 0.5 for r in requirements}
        return ComplianceResult(
            scores=scores,
            compliant=compliant,
            methodology_score=self.score,
            verdict=self.verdict,
            confidence=0.9,
        )


class FakeSummarizer:
    def summarize(self, text: str) -> str:
        return "Краткая аннотация: материал о законах Ньютона."


_TXT = (
    "Изучение физики начинается с механики. Механика изучает движение тел и причины этого движения. "
    "Первый закон Ньютона утверждает: тело сохраняет состояние покоя или равномерного прямолинейного движения, "
    "пока на него не действует внешняя сила. Второй закон связывает силу, массу и ускорение. "
    "Третий закон говорит о равенстве сил действия и противодействия. Эти законы лежат в основе классической "
    "механики и используются в задачах о движении тел."
)


@pytest.fixture()
def manager(proj_root: Path, cfg: dict, db_path: Path):
    initialize(cfg)
    return SessionManager(make_engine(db_path))


@pytest.fixture()
def source_dir(tmp_path: Path) -> Path:
    d = tmp_path / "source"
    d.mkdir()
    (d / "physics.txt").write_text(_TXT, encoding="utf-8")
    (d / "archive.zip").write_bytes(b"PK\x03\x04rest-of-archive")
    (d / "fake.pdf").write_bytes(b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\x00")
    return d


def _pipeline(manager, cfg, **kw) -> LoaderPipeline:
    return LoaderPipeline(
        cfg, manager,
        classifier=kw.get("classifier", FakeClassifier()),
        summarizer=kw.get("summarizer", FakeSummarizer()),
        ocr=None,
        root=None,
    )


def test_pipeline_full_run(manager, cfg: dict, source_dir: Path, db_path: Path) -> None:
    pipeline = _pipeline(manager, cfg)
    report = pipeline.run(source_dir)

    assert report.total == 3
    counts = report.counts
    assert counts[ST_OK] == 1
    assert counts[ST_SKIPPED_UNSUPPORTED] == 1  # .zip
    assert counts[ST_REJECTED_SIGNATURE] == 1   # fake.pdf

    accepted = report.accepted
    assert accepted[0].subject == "Физика"
    assert accepted[0].verdict == ModerationVerdict.APPROVED.value

    # БД
    with manager.session() as session:
        material = session.scalar(select(Material).where(Material.source_hash == accepted[0].source_hash))
        assert material is not None
        assert material.category == "Естественные науки"
        assert material.subject == "Физика"
        assert material.education_level == EducationLevel.UNIVERSITY.value
        assert material.complexity_level == ComplexityLevel.MEDIUM.value
        assert material.methodology_score == pytest.approx(0.85)
        assert material.moderation_verdict == ModerationVerdict.APPROVED.value
        assert material.annotation == "Краткая аннотация: материал о законах Ньютона."
        assert material.word_count > 40
        compliance_rows = session.scalars(
            select(MaterialCompliance).where(MaterialCompliance.material_id == material.id)
        ).all()
        assert len(compliance_rows) == 24


def test_dedup_on_second_run(manager, cfg: dict, source_dir: Path) -> None:
    pipeline = _pipeline(manager, cfg)
    first = pipeline.run(source_dir)
    second = pipeline.run(source_dir)
    assert first.counts[ST_OK] == 1
    assert second.counts[ST_DUPLICATE] == 1
    assert second.counts[ST_OK] == 0


def test_skip_ai_saves_defaults(manager, cfg: dict, source_dir: Path) -> None:
    pipeline = LoaderPipeline(cfg, manager, classifier=None, summarizer=None, ocr=None)
    report = pipeline.run(source_dir, skip_ai=True)
    accepted = report.accepted
    assert len(accepted) == 1
    assert accepted[0].subject is None
    assert accepted[0].verdict == ModerationVerdict.PENDING.value

    with manager.session() as session:
        material = session.scalar(
            select(Material).where(Material.source_hash == accepted[0].source_hash)
        )
        assert material.education_level == EducationLevel.UNKNOWN.value
        assert material.material_type == MaterialType.CANNOT_DETERMINE.value
        assert material.moderation_verdict == ModerationVerdict.PENDING.value
        assert material.complexity_score is None


def test_limit_argument(manager, cfg: dict, source_dir: Path) -> None:
    pipeline = _pipeline(manager, cfg)
    report = pipeline.run(source_dir, limit=1)
    assert report.total == 1


def test_total_run_limit(manager, source_dir: Path) -> None:
    """Общий лимит запуска: при бюджете 0 МБ все файлы пропускаются."""
    mini = {
        "ingestion": {
            "allowed_extensions": [".txt", ".pdf", ".zip"],
            "max_file_mb": 100,
            "max_video_mb": 500,
            "max_total_mb_per_run": 0,
            "allow_archives": False,
        },
        "moderation": {},
    }
    pipeline = LoaderPipeline(mini, manager, classifier=None, summarizer=None, ocr=None)
    report = pipeline.run(source_dir)
    assert report.counts[ST_SKIPPED_TOO_LARGE] == 3
    assert report.counts[ST_OK] == 0


def test_manifest_run(manager, cfg: dict, source_dir: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(
        f"{source_dir / 'physics.txt'}\n"
        f"# комментарий игнорируется\n"
        f"{source_dir / 'missing.pdf'}\n",
        encoding="utf-8",
    )
    pipeline = _pipeline(manager, cfg)
    report = pipeline.run(manifest)
    assert report.total == 2
    assert report.counts[ST_OK] == 1
    assert report.counts[ST_FAILED] == 1  # файла нет


def test_report_markdown(manager, cfg: dict, source_dir: Path) -> None:
    report = _pipeline(manager, cfg).run(source_dir)
    md = render_markdown(report)
    assert "# Отчёт о загрузке материалов" in md
    assert "## Итоги" in md
    assert "Физика" in md
    assert "archive.zip" in md or "archive" in md
    assert "fake.pdf" in md


def test_moderation_verdict_thresholds() -> None:
    from prof2027.ai.classifier import ClassifierAdapter

    thresholds = {"approved_threshold": 0.7, "needs_review_threshold": 0.4}
    assert ClassifierAdapter._verdict(0.85, thresholds) == ModerationVerdict.APPROVED.value
    assert ClassifierAdapter._verdict(0.5, thresholds) == ModerationVerdict.NEEDS_REVIEW.value
    assert ClassifierAdapter._verdict(0.2, thresholds) == ModerationVerdict.REJECTED.value


def test_questions_built_from_catalogs() -> None:
    """Вопросы d1-2B строятся из YAML-каталогов (без загрузки модели)."""
    from prof2027.ai.classifier import ClassifierAdapter

    adapter = ClassifierAdapter(root=None)
    meta = adapter._questions_metadata()
    assert "category" in meta and "complexity" in meta and "topic" in meta
    assert "subject" not in meta  # предмет ушёл во второй проход
    assert len(meta["category"]["criteria"]) == 10  # 9 категорий + cannot_determine

    # второй проход: дисциплины только выбранной категории
    subj_q = adapter._subject_question("Естественные науки")
    assert len(subj_q["criteria"]) == 6  # 5 дисциплин + cannot_determine
    assert "Физика" in subj_q["criteria"] and "Биология" in subj_q["criteria"]
    all_q = adapter._subject_question(None)
    assert len(all_q["criteria"]) == 36  # 35 дисциплин + cannot_determine

    comp = adapter._questions_compliance()
    assert len(comp["compliance"]["criteria"]) == 24
    assert comp["compliance"]["multiple"] is True


def test_topic_filter_removes_prompt_echo() -> None:
    from prof2027.ai.classifier import ClassifierAdapter

    assert ClassifierAdapter._answer_topic(
        {"text": "Тема: колебания и волны", "found": 0.9}, "fallback"
    ) == "Тема: колебания и волны"
    # модель повторила служебный текст — отбрасываем
    assert ClassifierAdapter._answer_topic(
        {"text": "Биология УЧЕБНЫЙ МАТЕРИАЛ: БИОЛОГИЯ", "found": 0.9}, "fallback"
    ) == "fallback"
    assert ClassifierAdapter._answer_topic(None, "fallback") == "fallback"
