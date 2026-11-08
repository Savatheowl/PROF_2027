"""Тесты валидатора входных файлов."""

from pathlib import Path

from prof2027.validation import validate_file


def test_txt_ok(tmp_path, cfg):
    p = tmp_path / "file.txt"
    p.write_text("Привет, это тест материала.", encoding="utf-8")
    assert validate_file(p, cfg).status == "ok"


def test_unsupported_extension(tmp_path, cfg):
    p = tmp_path / "file.exe"
    p.write_bytes(b"MZ\x90\x00\x03\x00\x00\x00\x00\x00\x00\x00")
    vr = validate_file(p, cfg)
    assert vr.status == "skipped_unsupported"


def test_exe_renamed_to_pdf_rejected(tmp_path, cfg):
    p = tmp_path / "virus.pdf"
    p.write_bytes(b"MZ\x90\x00\x03\x00\x00\x00\x00\x00\x00\x00" + b"\x00" * 40)
    vr = validate_file(p, cfg)
    assert vr.status == "rejected_signature"


def test_signature_mismatch_for_pdf(tmp_path, cfg):
    p = tmp_path / "fake.pdf"
    p.write_text("not a pdf at all", encoding="utf-8")
    assert validate_file(p, cfg).status == "rejected_signature"


def test_archive_not_allowed(tmp_path, cfg):
    p = tmp_path / "archive.zip"
    p.write_bytes(b"PK\x03\x04" + b"\x00" * 40)
    vr = validate_file(p, cfg)
    assert vr.status == "skipped_unsupported"


def test_too_large(tmp_path, cfg, monkeypatch):
    p = tmp_path / "big.txt"
    p.write_text("x" * 50_000, encoding="utf-8")  # ~50 KB
    monkeypatch.setitem(cfg.ingestion, "max_file_mb", 0.001)  # 1 KB
    assert validate_file(p, cfg).status == "skipped_too_large"


def test_unsafe_name_rejected(tmp_path, cfg):
    from prof2027.validation import safe_name

    assert not safe_name("../evil.txt")
    assert not safe_name("evil\\..txt")
    assert not safe_name("a\x00b.txt")
    assert safe_name("good_name.pdf")


def test_missing_file(tmp_path, cfg):
    vr = validate_file(tmp_path / "nope.pdf", cfg)
    assert vr.status == "skipped_unsupported"
