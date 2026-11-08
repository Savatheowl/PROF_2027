"""Единый валидатор входных файлов.

Используется везде: loader, API, Telegram, дашборд.
Проверки: расширение (белый список), сигнатура (magic bytes), размер, имя файла.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

TEXT_LIKE = {".txt", ".md", ".html", ".htm"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
ARCHIVE_EXTS = {".zip", ".rar", ".7z", ".gz", ".tar", ".exe", ".msi", ".dll", ".bat", ".ps1", ".js", ".py", ".svg", ".iso", ".bin"}


@dataclass
class ValidationResult:
    status: str  # ok | skipped_unsupported | rejected_signature | skipped_too_large | rejected_path | skipped_archive
    reason: str = ""


def check_signature(path: Path, ext: str) -> bool:
    """Сверяет magic bytes с расширением."""
    with open(path, "rb") as f:
        head = f.read(16)
    if ext == ".pdf":
        return head.startswith(b"%PDF")
    if ext == ".png":
        return head.startswith(b"\x89PNG")
    if ext in (".jpg", ".jpeg"):
        return head.startswith(b"\xff\xd8\xff")
    if ext == ".bmp":
        return head.startswith(b"BM")
    if ext == ".webp":
        return len(head) >= 12 and head.startswith(b"RIFF") and head[8:12] == b"WEBP"
    if ext == ".tiff":
        return head.startswith(b"II*\x00") or head.startswith(b"MM\x00*")
    if ext == ".rtf":
        return head.startswith(b"{\\rtf")
    if ext == ".doc":
        return head.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1")  # OLE2
    if ext in (".docx", ".pptx"):
        return head.startswith(b"PK\x03\x04")  # ZIP/OOXML
    if ext == ".mp4":
        return len(head) >= 8 and head[4:8] == b"ftyp"
    if ext == ".mov":
        return len(head) >= 12 and head[4:8] == b"ftyp" and head[8:12] in (b"qt  ", b"moov", b"isom")
    if ext == ".avi":
        return len(head) >= 12 and head.startswith(b"RIFF") and head[8:12] == b"AVI "
    if ext == ".mkv":
        return head.startswith(b"\x1a\x45\xdf\xa3")
    if ext in (".html", ".htm"):
        return head.lstrip()[:1] == b"<"
    if ext in (".txt", ".md"):
        # текстовый файл: не содержит NUL и почти весь в печатных символах
        try:
            text = head.decode("utf-8", errors="replace")
            printable = sum(1 for ch in text if ch.isprintable() or ch in "\r\n\t")
            return not text.count("\x00") and printable / max(1, len(text)) > 0.9
        except Exception:
            return False
    return True


def safe_name(name: str) -> bool:
    return bool(name) and ".." not in name and "/" not in name and "\\" not in name and "\x00" not in name


def validate_file(path: Path, cfg, total_mb: float = 0.0) -> ValidationResult:
    """Проверяет файл. Возвращает статус и причину."""
    if not path.exists() or not path.is_file():
        return ValidationResult("skipped_unsupported", f"файл не существует: {path}")

    name = path.name
    if not safe_name(name):
        return ValidationResult("rejected_path", "небезопасное имя файла")

    ext = path.suffix.lower()
    allowed = cfg.allowed_extensions
    if ext not in allowed:
        if ext in ARCHIVE_EXTS:
            return ValidationResult("skipped_unsupported", f"архив/исполняемый файл не принимается: {ext}")
        return ValidationResult("skipped_unsupported", f"расширение {ext or '(без расширения)'} не в белом списке")

    size_mb = path.stat().st_size / (1024 * 1024)
    limit_mb = cfg.ingestion["max_video_mb"] if ext in VIDEO_EXTS else cfg.ingestion["max_file_mb"]
    if size_mb > limit_mb:
        return ValidationResult("skipped_too_large", f"размер {size_mb:.1f} МБ > лимит {limit_mb} МБ")
    max_total = cfg.ingestion["max_total_mb_per_run"]
    if total_mb + size_mb > max_total:
        return ValidationResult("skipped_too_large", f"суммарный размер превышает {max_total} МБ за запуск")

    if not check_signature(path, ext):
        return ValidationResult("rejected_signature", "сигнатура не соответствует расширению")

    return ValidationResult("ok")


def quarantine(path: Path, cfg) -> Path:
    """Кладёт подозрительный файл в карантин. Возвращает путь к копии."""
    qdir = cfg.root / "data" / "quarantine"
    qdir.mkdir(parents=True, exist_ok=True)
    dst = qdir / f"{path.stem}_{path.name}"
    shutil.copy2(path, dst)
    return dst
