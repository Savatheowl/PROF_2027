"""Тесты единого фильтра входных файлов."""

from __future__ import annotations

from pathlib import Path

import pytest

from prof2027.core import validation
from prof2027.core.validation import (
    ValidationResult,
    ValidationStatus,
    validate_bytes,
    validate_file,
    validate_name,
)


def _cfg(max_file_mb: int = 100, max_video_mb: int = 500) -> dict:
    return {
        "ingestion": {
            "allowed_extensions": [
                ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".txt", ".md",
                ".html", ".htm", ".png", ".jpg", ".jpeg", ".mp4", ".mp3", ".epub", ".fb2",
            ],
            "max_file_mb": max_file_mb,
            "max_video_mb": max_video_mb,
            "max_total_mb_per_run": 8192,
            "allow_archives": False,
        }
    }


def _write(tmp_path: Path, name: str, data: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


def test_valid_pdf_passes(tmp_path: Path) -> None:
    p = _write(tmp_path, "file.pdf", b"%PDF-1.7\ncontent\n")
    result = validate_file(p, _cfg())
    assert result.ok
    assert result.status == ValidationStatus.OK


def test_unsupported_extension(tmp_path: Path) -> None:
    p = _write(tmp_path, "archive.zip", b"PK\x03\x04rest")
    result = validate_file(p, _cfg())
    assert not result.ok
    assert result.status == ValidationStatus.SKIPPED_UNSUPPORTED


def test_exe_in_zip_name(tmp_path: Path) -> None:
    p = _write(tmp_path, "installer.exe", b"MZ\x90\x00")
    result = validate_file(p, _cfg())
    assert not result.ok
    assert result.status == ValidationStatus.SKIPPED_UNSUPPORTED


def test_renamed_exe_signature(tmp_path: Path) -> None:
    # Расширение .pdf, но содержимое — PE-файл
    p = _write(tmp_path, "fake.pdf", b"MZ\x90\x00\x03\x00\x00\x00")
    result = validate_file(p, _cfg())
    assert not result.ok
    assert result.status == ValidationStatus.REJECTED_SIGNATURE


def test_jpg_jpeg_alias(tmp_path: Path) -> None:
    p = _write(tmp_path, "photo.jpeg", b"\xff\xd8\xff\xe0\x00\x10JFIF")
    result = validate_file(p, _cfg())
    assert result.ok


def test_png_as_jpg_accepted(tmp_path: Path) -> None:
    """PNG-контент с расширением .jpg — безопасный растровый формат."""
    p = _write(tmp_path, "photo.jpg", b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
    result = validate_file(p, _cfg())
    assert result.ok


def test_too_large(tmp_path: Path) -> None:
    p = _write(tmp_path, "big.pdf", b"%PDF-1.7" + b"x" * (2 * 1024 * 1024))
    result = validate_file(p, _cfg(max_file_mb=1))
    assert not result.ok
    assert result.status == ValidationStatus.SKIPPED_TOO_LARGE


def test_video_uses_video_limit(tmp_path: Path) -> None:
    p = _write(tmp_path, "video.mp4", b"\x00\x00\x00\x18ftypmp42" + b"x" * (200 * 1024 * 1024))
    result = validate_file(p, _cfg(max_file_mb=100, max_video_mb=500))
    assert result.ok  # 200 МБ < 500 МБ, хотя > 100 МБ


def test_validate_bytes(tmp_path: Path) -> None:
    ok = validate_bytes(b"%PDF-1.7\n", "doc.pdf", _cfg())
    assert ok.ok
    bad = validate_bytes(b"MZ\x90\x00", "doc.pdf", _cfg())
    assert bad.status == ValidationStatus.REJECTED_SIGNATURE
    unsup = validate_bytes(b"x", "script.py", _cfg())
    assert unsup.status == ValidationStatus.SKIPPED_UNSUPPORTED


def test_name_checks() -> None:
    assert validate_name("lesson_01.pdf")
    assert not validate_name("../lesson.pdf")
    assert not validate_name("a\\b.pdf")
    assert not validate_name("bad\x00name.pdf")
    assert not validate_name("..")


def test_total_limit() -> None:
    assert validation.check_total_limit([1 * 1024 * 1024], _cfg())
    assert not validation.check_total_limit([9000 * 1024 * 1024], _cfg())
