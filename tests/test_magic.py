"""Тесты определения типа файла по сигнатуре."""

from __future__ import annotations

import struct
import zipfile
from io import BytesIO

from prof2027.core import magic


def _pdf() -> bytes:
    return b"%PDF-1.7\n%proftest\n1 0 obj\n<<>>\nendobj\n"


def _png() -> bytes:
    return b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"


def _jpg() -> bytes:
    return b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01"


def _zip_with(names: list[str]) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in names:
            zf.writestr(name, "content")
    return buf.getvalue()


def _mp3() -> bytes:
    # ID3v2
    return b"ID3\x03\x00\x00\x00\x00\x00\x00TIT2\x00\x00\x00\x05Test"


def test_pdf() -> None:
    assert magic.detect(_pdf()) == "pdf"


def test_png_and_jpg() -> None:
    assert magic.detect(_png()) == "png"
    assert magic.detect(_jpg()) == "jpg"


def test_docx_pptx() -> None:
    assert magic.detect(_zip_with(["[Content_Types].xml", "word/document.xml"])) == "docx"
    assert magic.detect(_zip_with(["[Content_Types].xml", "ppt/presentation.xml"])) == "pptx"


def test_epub() -> None:
    assert magic.detect(_zip_with(["mimetype"])) == "epub"


def test_zip_kind_from_partial_header() -> None:
    """Определение zip-типов по первым 64 КБ (центральный каталог в конце)."""
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("content.xhtml", "x" * 200_000)
    header = buf.getvalue()[:65536]
    assert magic.detect(header) == "epub"

    buf2 = BytesIO()
    with zipfile.ZipFile(buf2, "w") as zf:
        zf.writestr("[Content_Types].xml", "x")
        zf.writestr("word/document.xml", "y" * 200_000)
        zf.writestr("docProps/core.xml", "z")
    header2 = buf2.getvalue()[:65536]
    assert magic.detect(header2) == "docx"


def test_mp3() -> None:
    assert magic.detect(_mp3()) == "mp3"
    assert magic.detect(b"\xff\xfb\x90\x64") == "mp3"


def test_html_and_txt() -> None:
    html = b"<!DOCTYPE html><html><head><title>t</title></head><body>p</body></html>"
    assert magic.detect(html) == "html"
    assert magic.detect("Просто текст без разметки.\nВторая строка.".encode("utf-8")) == "txt"


def test_avi_mp4_mkv() -> None:
    avi = b"RIFF" + struct.pack("<I", 100) + b"AVI LIST"
    assert magic.detect(avi) == "avi"
    mp4 = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00"
    assert magic.detect(mp4) == "mp4"
    assert magic.detect(b"\x1a\x45\xdf\xa3\x93\x42\x82\x88") == "mkv"


def test_unknown() -> None:
    assert magic.detect(b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00") is None


def test_empty() -> None:
    assert magic.detect(b"") is None
