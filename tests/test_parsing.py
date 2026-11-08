"""Тесты парсеров форматов на реально создаваемых сэмплах."""

from __future__ import annotations

import zipfile
from pathlib import Path

from prof2027.parsing.extract import ParsedDocument, parse_file

_TXT = (
    "Изучение физики начинается с механики. Механика изучает движение тел и причины этого движения. "
    "Первый закон Ньютона утверждает: тело сохраняет состояние покоя или равномерного прямолинейного движения, "
    "пока на него не действует внешняя сила. Второй закон связывает силу, массу и ускорение. "
    "Третий закон говорит о равенстве сил действия и противодействия. Эти законы лежат в основе классической "
    "механики и используются в задачах о движении тел."
)


def _make_txt(tmp_path: Path, name: str = "mechanics.txt") -> Path:
    p = tmp_path / name
    p.write_text(_TXT, encoding="utf-8")
    return p


def test_txt(tmp_path: Path) -> None:
    parsed = parse_file(_make_txt(tmp_path))
    assert isinstance(parsed, ParsedDocument)
    assert parsed.word_count > 40
    assert parsed.file_type == "txt"


def test_docx(tmp_path: Path) -> None:
    import docx

    d = docx.Document()
    d.add_paragraph(_TXT)
    d.add_paragraph("Вопросы для самопроверки. Каков второй закон Ньютона?")
    p = tmp_path / "lesson.docx"
    d.save(str(p))
    parsed = parse_file(p)
    assert parsed.word_count > 40
    assert parsed.file_type == "docx"


def test_pptx(tmp_path: Path) -> None:
    from pptx import Presentation

    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = "Физика. Законы Ньютона"
    body = slide.placeholders[1]
    body.text = _TXT
    p = tmp_path / "presentation.pptx"
    prs.save(str(p))
    parsed = parse_file(p)
    assert parsed.word_count > 40
    assert parsed.file_type == "pptx"


def test_html(tmp_path: Path) -> None:
    p = tmp_path / "page.html"
    p.write_text(
        f"<!DOCTYPE html><html><head><title>Тема</title></head>"
        f"<body><h1>Физика</h1><p>{_TXT}</p><img alt='схема'></body></html>",
        encoding="utf-8",
    )
    parsed = parse_file(p)
    assert parsed.word_count > 40
    assert parsed.media_count >= 1


def test_epub(tmp_path: Path) -> None:
    p = tmp_path / "book.epub"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr("content.xhtml", f"<html><body><p>{_TXT}</p></body></html>")
    parsed = parse_file(p)
    assert parsed.word_count > 40
    assert parsed.file_type == "epub"


def test_truncated_epub(tmp_path: Path) -> None:
    """Обрезанный epub (нет центрального каталога) читается по локальным записям."""
    full = tmp_path / "full.epub"
    with zipfile.ZipFile(full, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr("content.xhtml", f"<html><body><p>{_TXT}</p></body></html>")
    data = full.read_bytes()
    # отрезаем центральный каталог и end-of-central-directory
    eocd = data.rfind(b"PK\x05\x06")
    assert eocd > 0
    truncated = tmp_path / "truncated.epub"
    truncated.write_bytes(data[:eocd])
    parsed = parse_file(truncated)
    assert parsed.word_count > 40
    assert parsed.file_type == "epub"


def test_fb2(tmp_path: Path) -> None:
    p = tmp_path / "book.fb2"
    p.write_text(
        f'<?xml version="1.0"?><FictionBook><body><section><p>{_TXT}</p></section></body></FictionBook>',
        encoding="utf-8",
    )
    parsed = parse_file(p)
    assert parsed.word_count > 40
    assert parsed.file_type == "fb2"


def test_pdf_digital(tmp_path: Path) -> None:
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(72, 72, 540, 720), _TXT, fontsize=10)
    p = tmp_path / "document.pdf"
    doc.save(str(p))
    doc.close()
    parsed = parse_file(p)  # без OCR-модели: цифровой текст
    assert parsed.word_count > 40
    assert parsed.file_type == "pdf"


def test_png_image(tmp_path: Path) -> None:
    from PIL import Image

    img = Image.new("RGB", (40, 30), "white")
    p = tmp_path / "scheme.png"
    img.save(p)
    parsed = parse_file(p)  # OCR отсутствует -> текст пустой
    assert parsed.file_type == "png"
    assert parsed.media_count == 1
    assert parsed.media[0]["type"] == "image"


def test_audio(tmp_path: Path) -> None:
    p = tmp_path / "audio.mp3"
    p.write_bytes(b"ID3\x03\x00\x00\x00\x00\x00\x00TIT2\x00\x00\x00\x05Test")
    parsed = parse_file(p)
    assert parsed.media_count == 1
    assert parsed.media[0]["type"] == "audio"


def test_unsupported_format_raises(tmp_path: Path) -> None:
    p = tmp_path / "script.py"
    p.write_text("print(1)", encoding="utf-8")
    import pytest

    with pytest.raises(ValueError):
        parse_file(p)
