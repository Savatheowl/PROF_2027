"""Конвейер загрузчика: discover → validate → hash → dedup → copy → parse → classify → moderate → save.

Поток одного файла:
1. discover   — обход папки или чтение manifest'а;
2. validate   — единый фильтр (расширение, сигнатура, размер, имя);
3. hash       — SHA-256 содержимого;
4. dedup      — пропуск, если такой source_hash уже есть в БД;
5. copy       — копия в data/raw/ (для офлайн-архива и PyInstaller);
6. parse      — текст и медиа (pdf/docx/pptx/html/epub/fb2/txt/rtf/image/audio/video);
7. classify   — d1-2B: категория, дисциплина, уровень, тип, сложность, тема;
8. moderate   — d1-2B: 24 методических требования + вердикт;
9. save       — Material + MaterialCompliance в БД.

Адаптеры AI инъекцируются (фейки в тестах); при `skip_ai` материал
сохраняется с дефолтными атрибутами и вердиктом pending.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from prof2027.ai.base import ModelAdapter
from prof2027.core import validation
from prof2027.core.paths import ensure_dirs, project_root
from prof2027.core.validation import ValidationResult
from prof2027.db.base import SessionManager
from prof2027.db.models import (
    Material,
    MaterialCompliance,
    MethodologicalRequirement,
)
from prof2027.domain.enums import (
    EducationLevel,
    Language,
    MaterialType,
    ModerationVerdict,
)
from prof2027.parsing.extract import ParsedDocument, parse_file

logger = logging.getLogger(__name__)

_HASH_CHUNK = 1024 * 1024

# Статусы одного файла в отчёте
ST_OK = "ok"
ST_SKIPPED_UNSUPPORTED = "skipped_unsupported"
ST_REJECTED_SIGNATURE = "rejected_signature"
ST_SKIPPED_TOO_LARGE = "skipped_too_large"
ST_REJECTED_PATH = "rejected_path"
ST_DUPLICATE = "duplicate"
ST_FAILED = "failed"

_STATUS_RU = {
    ST_OK: "принято",
    ST_SKIPPED_UNSUPPORTED: "пропущено (расширение)",
    ST_REJECTED_SIGNATURE: "отклонено (сигнатура)",
    ST_SKIPPED_TOO_LARGE: "пропущено (размер)",
    ST_REJECTED_PATH: "отклонено (имя)",
    ST_DUPLICATE: "дубликат",
    ST_FAILED: "ошибка обработки",
}


@dataclass
class SourceItem:
    """Результат обработки одного файла."""

    path: Path
    status: str
    reason: str | None = None
    validation: ValidationResult | None = None
    file_type: str | None = None
    size_bytes: int = 0
    source_hash: str | None = None
    copied_path: Path | None = None
    title: str | None = None
    category: str | None = None
    subject: str | None = None
    topic: str | None = None
    education_level: str | None = None
    material_type: str | None = None
    complexity_level: str | None = None
    methodology_score: float | None = None
    verdict: str | None = None
    word_count: int = 0
    material_id: int | None = None


@dataclass
class RunReport:
    source: str
    started_at: datetime
    finished_at: datetime | None = None
    items: list[SourceItem] = field(default_factory=list)
    limit: int | None = None
    skip_ai: bool = False
    max_pages: int = 20

    @property
    def counts(self) -> dict[str, int]:
        out = {status: 0 for status in _STATUS_RU}
        for item in self.items:
            out[item.status] = out.get(item.status, 0) + 1
        return out

    @property
    def total(self) -> int:
        return len(self.items)

    @property
    def accepted(self) -> list[SourceItem]:
        return [i for i in self.items if i.status == ST_OK]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(_HASH_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _read_manifest(source: Path) -> list[Path]:
    """Читает manifest: JSON-список путей или текстовый файл (путь на строку)."""
    suffix = source.suffix.lower()
    if suffix == ".json":
        data = json.loads(source.read_text("utf-8"))
        if isinstance(data, dict):
            data = data.get("files", [])
        paths: list[Path] = []
        for entry in data:
            if isinstance(entry, str):
                paths.append(Path(entry))
            elif isinstance(entry, dict) and entry.get("path"):
                paths.append(Path(entry["path"]))
        return paths
    # текстовый manifest
    lines = source.read_text("utf-8", errors="replace").splitlines()
    paths = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        raw_path = line.split("|", 1)[0].strip()
        if raw_path:
            paths.append(Path(raw_path))
    return paths


def _discover(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted(p for p in source.rglob("*") if p.is_file())
    if source.is_file():
        return _read_manifest(source)
    raise FileNotFoundError(f"Источник не найден: {source}")


class LoaderPipeline:
    """Конвейер загрузки. Адаптеры AI можно подменить фейками в тестах."""

    def __init__(
        self,
        cfg: dict,
        manager: SessionManager,
        *,
        ocr: ModelAdapter | None = None,
        classifier: ModelAdapter | None = None,
        summarizer: ModelAdapter | None = None,
        root: Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.manager = manager
        self.ocr = ocr
        self.classifier = classifier
        self.summarizer = summarizer
        self.root = root or project_root()
        self.paths = ensure_dirs(self.root)

    # ------------------------------------------------------------------
    # Основной запуск
    # ------------------------------------------------------------------

    def run(
        self,
        source: str | Path,
        *,
        limit: int | None = None,
        max_pages: int = 20,
        skip_ai: bool = False,
        hint: str | None = None,
    ) -> RunReport:
        source_path = Path(source)
        report = RunReport(
            source=str(source_path),
            started_at=datetime.now(),
            limit=limit,
            skip_ai=skip_ai,
            max_pages=max_pages,
        )

        discovered = _discover(source_path)
        if limit is not None and limit > 0:
            discovered = discovered[:limit]
        logger.info("Обнаружено файлов: %d", len(discovered))

        total_limit_mb = int(self.cfg.get("ingestion", {}).get("max_total_mb_per_run", 8192))
        budget_bytes = total_limit_mb * 1024 * 1024
        accepted_bytes = 0

        for path in discovered:
            try:
                if accepted_bytes + _file_size(path) > budget_bytes:
                    item = SourceItem(
                        path=path,
                        status=ST_SKIPPED_TOO_LARGE,
                        reason=f"Превышен общий лимит запуска {total_limit_mb} МБ",
                        size_bytes=_file_size(path),
                    )
                else:
                    item = self._process(path, max_pages=max_pages, skip_ai=skip_ai, hint=hint)
                    if item.status == ST_OK:
                        accepted_bytes += item.size_bytes
            except Exception as exc:  # noqa: BLE001
                logger.exception("Файл %s: обработка прервана", path)
                item = SourceItem(
                    path=path,
                    status=ST_FAILED,
                    reason=f"Необработанная ошибка: {exc}",
                )
            report.items.append(item)
            self._log_item(item)

        report.finished_at = datetime.now()
        return report

    # ------------------------------------------------------------------
    # Обработка одного файла
    # ------------------------------------------------------------------

    def _process(
        self,
        path: Path,
        *,
        max_pages: int,
        skip_ai: bool,
        hint: str | None,
    ) -> SourceItem:
        if not path.is_file():
            return SourceItem(
                path=path,
                status=ST_FAILED,
                reason="Файл не существует",
                size_bytes=0,
            )

        # 2. validate
        result = validation.validate_file(path, self.cfg)
        if not result.ok:
            return SourceItem(
                path=path,
                status=result.status.value,
                reason=result.reason,
                validation=result,
                size_bytes=result.size_bytes,
            )

        # 3. hash
        source_hash = _sha256(path)

        # 4. dedup
        with self.manager.session() as session:
            existing = session.scalar(
                select(Material.id).where(Material.source_hash == source_hash)
            )
        if existing is not None:
            return SourceItem(
                path=path,
                status=ST_DUPLICATE,
                reason="Такой файл уже есть в базе (source_hash совпадает)",
                validation=result,
                source_hash=source_hash,
                file_type=result.detected_ext,
                size_bytes=result.size_bytes,
            )

        # 5. copy
        copied = self._copy_to_raw(path)

        title = path.stem or path.name
        try:
            # 6. parse
            parsed = parse_file(path, ocr=self._ocr_callable(), max_pages=max_pages)
            text = parsed.text.strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Парсинг %s не удался: %s", path.name, exc)
            if copied is not None:
                copied.unlink(missing_ok=True)
            return SourceItem(
                path=path,
                status=ST_FAILED,
                reason=f"Ошибка извлечения текста: {exc}",
                validation=result,
                file_type=result.detected_ext,
                size_bytes=result.size_bytes,
                source_hash=source_hash,
            )

        # 7-8. классификация и модерация
        classification = None
        compliance = None
        annotation = None
        if not skip_ai and self.classifier is not None and text:
            try:
                classification = self.classifier.classify(text[:12_000], title, hint)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Классификация %s не удалась: %s", path.name, exc)
            if classification is not None:
                try:
                    compliance = self.classifier.moderate(
                        text[:12_000], title, hint,
                        thresholds=self.cfg.get("moderation", {}),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Модерация %s не удалась: %s", path.name, exc)
        if not skip_ai and self.summarizer is not None and text:
            try:
                annotation = self.summarizer.summarize(text[:8_000])
            except Exception as exc:  # noqa: BLE001
                logger.warning("Аннотация %s не удалась: %s", path.name, exc)

        # 9. save
        try:
            material_id = self._save(
                path=path,
                copied=copied,
                parsed=parsed,
                source_hash=source_hash,
                classification=classification,
                compliance=compliance,
                annotation=annotation,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Сохранение %s не удалось: %s", path.name, exc)
            return SourceItem(
                path=path,
                status=ST_FAILED,
                reason=f"Ошибка сохранения: {exc}",
                validation=result,
                file_type=result.detected_ext,
                size_bytes=result.size_bytes,
                source_hash=source_hash,
            )

        return SourceItem(
            path=path,
            status=ST_OK,
            validation=result,
            file_type=result.detected_ext or path.suffix.lstrip("."),
            size_bytes=result.size_bytes,
            source_hash=source_hash,
            copied_path=copied,
            title=title,
            category=getattr(classification, "category", None),
            subject=getattr(classification, "subject", None),
            topic=getattr(classification, "topic", None),
            education_level=getattr(classification, "education_level", None),
            material_type=getattr(classification, "material_type", None),
            complexity_level=getattr(classification, "complexity_level", None),
            methodology_score=getattr(compliance, "methodology_score", None),
            verdict=getattr(compliance, "verdict", ModerationVerdict.PENDING.value),
            word_count=parsed.word_count,
            material_id=material_id,
        )

    # ------------------------------------------------------------------
    # Шаги
    # ------------------------------------------------------------------

    def _ocr_callable(self) -> Callable | None:
        if self.ocr is None:
            return None
        return lambda file_path, max_pages: self.ocr.extract_text(file_path, max_pages=max_pages)

    def _copy_to_raw(self, path: Path) -> Path:
        dest_dir = self.paths["raw_dir"]
        dest = dest_dir / path.name
        counter = 1
        while dest.exists():
            dest = dest_dir / f"{path.stem}_{counter}{path.suffix}"
            counter += 1
        shutil.copy2(path, dest)
        return dest

    def _save(
        self,
        *,
        path: Path,
        copied: Path,
        parsed: ParsedDocument,
        source_hash: str,
        classification,
        compliance,
        annotation: str | None,
    ) -> int:
        requirements_map: dict[str, MethodologicalRequirement] = {}
        with self.manager.session() as session:
            req_rows = session.scalars(select(MethodologicalRequirement)).all()
            requirements_map = {r.text: r for r in req_rows}

        title = path.stem or path.name
        features = {
            "file_type": parsed.file_type,
            "words": parsed.word_count,
            "chars": parsed.char_count,
            "sentences": parsed.sentence_count,
            "media": parsed.media_count,
            "confidences": getattr(classification, "confidences", None),
            "complexity_score": getattr(classification, "complexity_score", None),
        }

        with self.manager.transaction() as session:
            material = Material(
                source_hash=source_hash,
                file_path=str(copied),
                file_type=parsed.file_type,
                title=title,
                category=getattr(classification, "category", None),
                subject=getattr(classification, "subject", None),
                topic=getattr(classification, "topic", None),
                education_level=getattr(classification, "education_level", EducationLevel.UNKNOWN.value),
                language=getattr(classification, "language", Language.OTHER.value),
                material_type=getattr(classification, "material_type", MaterialType.CANNOT_DETERMINE.value),
                class_type=getattr(classification, "class_type", "unknown"),
                text_content=parsed.text[:400_000] or None,
                annotation=annotation,
                media_descriptions=parsed.media or None,
                word_count=parsed.word_count,
                char_count=parsed.char_count,
                sentence_count=parsed.sentence_count,
                media_count=parsed.media_count,
                complexity_score=getattr(classification, "complexity_score", None),
                complexity_level=getattr(classification, "complexity_level", None),
                methodology_score=getattr(compliance, "methodology_score", None),
                moderation_verdict=getattr(compliance, "verdict", ModerationVerdict.PENDING.value),
                is_generated=False,
                features_json=features,
            )
            session.add(material)
            session.flush()  # material.id

            if compliance is not None and requirements_map:
                for req_text, score in compliance.scores.items():
                    req = requirements_map.get(req_text)
                    if req is None:
                        continue
                    session.add(
                        MaterialCompliance(
                            material_id=material.id,
                            requirement_id=req.id,
                            score=float(score),
                            is_compliant=compliance.compliant.get(req_text),
                        )
                    )
            return material.id

    # ------------------------------------------------------------------
    # Логирование
    # ------------------------------------------------------------------

    @staticmethod
    def _log_item(item: SourceItem) -> None:
        label = _STATUS_RU.get(item.status, item.status)
        extra = ""
        if item.status == ST_OK:
            subject = item.subject or "—"
            verdict = item.verdict or ModerationVerdict.PENDING.value
            extra = f" | {item.file_type} | {subject} | {verdict}"
        elif item.reason:
            extra = f" | {item.reason}"
        logger.info("%s %s%s", label, item.path.name, extra)
