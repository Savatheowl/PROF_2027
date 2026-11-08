"""Единый фильтр входных файлов.

Один валидатор используется везде: CLI-loader, POST /materials,
Telegram-бот, загрузка из дашборда.

Проверки:
1. расширение — в белом списке;
2. сигнатура (magic bytes) — реальный тип совпадает с расширением;
3. размер — лимиты из config.yaml;
4. имя — без path traversal и недопустимых символов.

Неподходящие файлы не роняют запуск: они получают статус и причину,
которые попадают в отчёт.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from prof2027.core.magic import detect

_ILLEGAL_NAME = re.compile(r"[\x00-\x1f<>:\"/\\|?*]")
_MEDIA_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".mp3", ".webp", ".gif"}
# Форматы без жёсткой сигнатуры: допустим любой текстовый контент
_TEXT_EXTENSIONS = {".txt", ".md", ".html", ".htm", ".rtf", ".fb2"}
# Растровые изображения: контейнеры взаимозаменяемы (PIL открывает любой)
_RASTER_FORMATS = {"png", "jpg", "jpeg", "webp", "bmp", "tiff", "gif"}


class ValidationStatus(str, Enum):
    OK = "ok"
    SKIPPED_UNSUPPORTED = "skipped_unsupported"
    REJECTED_SIGNATURE = "rejected_signature"
    SKIPPED_TOO_LARGE = "skipped_too_large"
    REJECTED_PATH = "rejected_path"


@dataclass
class ValidationResult:
    ok: bool
    status: ValidationStatus
    reason: str | None = None
    detected_ext: str | None = None
    size_bytes: int = 0
    details: dict[str, Any] = field(default_factory=dict)


def allowed_extensions(cfg: dict) -> set[str]:
    raw = cfg.get("ingestion", {}).get("allowed_extensions", [])
    return {str(ext).strip().lower() for ext in raw}


def normalize_ext(filename: str) -> str | None:
    """Возвращает расширение в виде ".pdf" (без учёта регистра)."""
    name = filename.strip()
    if "." not in name:
        return None
    return "." + name.rsplit(".", 1)[1].lower()


def validate_name(filename: str) -> bool:
    """Имя файла: только имя без каталогов, без NUL и служебных символов."""
    if not filename or len(filename) > 255:
        return False
    if "\x00" in filename:
        return False
    if filename in (".", ".."):
        return False
    if "\\" in filename or "/" in filename:
        return False
    return not _ILLEGAL_NAME.search(filename)


def _size_limit_mb(cfg: dict, ext: str | None) -> int:
    ingestion = cfg.get("ingestion", {})
    if ext and ext in _MEDIA_EXTENSIONS:
        return int(ingestion.get("max_video_mb", 500))
    return int(ingestion.get("max_file_mb", 100))


def validate_file(path: Path, cfg: dict) -> ValidationResult:
    """Проверяет файл на диске. Возвращает результат и причину."""
    filename = path.name
    if not validate_name(filename):
        return ValidationResult(False, ValidationStatus.REJECTED_PATH, "Недопустимое имя файла")

    ext = normalize_ext(filename)
    allowed = allowed_extensions(cfg)
    if ext not in allowed:
        return ValidationResult(
            False,
            ValidationStatus.SKIPPED_UNSUPPORTED,
            f"Расширение {ext} не в белом списке",
            detected_ext=None,
        )

    size = path.stat().st_size if path.exists() else 0
    limit_mb = _size_limit_mb(cfg, ext)
    if size > limit_mb * 1024 * 1024:
        return ValidationResult(
            False,
            ValidationStatus.SKIPPED_TOO_LARGE,
            f"Размер {size / 1024 / 1024:.1f} МБ больше лимита {limit_mb} МБ",
            size_bytes=size,
        )

    with path.open("rb") as fh:
        header = fh.read(65536)
    detected = detect(header)
    details = {"detected_ext": detected, "size_bytes": size}
    signature_error = _check_signature(ext, detected, header)
    if signature_error is not None:
        return ValidationResult(
            False,
            ValidationStatus.REJECTED_SIGNATURE,
            signature_error,
            detected_ext=detected,
            size_bytes=size,
            details=details,
        )
    return ValidationResult(
        True,
        ValidationStatus.OK,
        size_bytes=size,
        detected_ext=detected,
        details=details,
    )


def validate_bytes(data: bytes, filename: str, cfg: dict) -> ValidationResult:
    """Проверяет файл, ещё не сохранённый на диск (API/бот)."""
    if not validate_name(filename):
        return ValidationResult(False, ValidationStatus.REJECTED_PATH, "Недопустимое имя файла")

    ext = normalize_ext(filename)
    allowed = allowed_extensions(cfg)
    if ext not in allowed:
        return ValidationResult(
            False,
            ValidationStatus.SKIPPED_UNSUPPORTED,
            f"Расширение {ext} не в белом списке",
        )

    limit_mb = _size_limit_mb(cfg, ext)
    if len(data) > limit_mb * 1024 * 1024:
        return ValidationResult(
            False,
            ValidationStatus.SKIPPED_TOO_LARGE,
            f"Размер {len(data) / 1024 / 1024:.1f} МБ больше лимита {limit_mb} МБ",
            size_bytes=len(data),
        )

    detected = detect(data[:65536])
    signature_error = _check_signature(ext, detected, data[:65536])
    if signature_error is not None:
        return ValidationResult(
            False,
            ValidationStatus.REJECTED_SIGNATURE,
            signature_error,
            detected_ext=detected,
            size_bytes=len(data),
        )
    return ValidationResult(
        True,
        ValidationStatus.OK,
        detected_ext=detected,
        size_bytes=len(data),
    )


def _check_signature(ext: str, detected: str | None, header: bytes) -> str | None:
    """Возвращает текст ошибки, если сигнатура не соответствует расширению."""
    if detected is not None:
        if not _compatible(ext, detected):
            return f"Сигнатура ({detected}) не совпадает с расширением ({ext})"
        return None
    # Сигнатура не распознана
    if ext in _TEXT_EXTENSIONS:
        if b"\x00" in header[:4096]:
            return f"Сигнатура не распознана, контент бинарный, а расширение {ext} текстовое"
        return None  # текстовый контент для текстового расширения — норм
    return "Сигнатура не распознана (возможно, файл переименован)"


def _compatible(ext: str, detected: str) -> bool:
    """Допустимые пары «расширение — сигнатура»."""
    ext = ext.lstrip(".")
    if ext == detected:
        return True
    if ext in _RASTER_FORMATS and detected in _RASTER_FORMATS:
        return True
    aliases = {
        "htm": {"html"},
        "html": {"htm"},
        "md": {"txt"},
        "fb2": {"html", "txt"},
        "mp4": {"mov"},
        "mov": {"mp4"},
        "txt": {"md"},
        "doc": {"rtf"},
        "rtf": {"doc"},
    }
    return ext in aliases and detected in aliases[ext]


def check_total_limit(files_bytes: list[int], cfg: dict) -> bool:
    """Проверка суммарного лимита запуска (max_total_mb_per_run)."""
    limit_mb = int(cfg.get("ingestion", {}).get("max_total_mb_per_run", 8192))
    return sum(files_bytes) <= limit_mb * 1024 * 1024
