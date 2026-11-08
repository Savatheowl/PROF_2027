"""Тесты парсинга форматов."""

from pathlib import Path

from prof2027.parsing import parse_file


def _make_pdf(path: Path) -> None:
    import pymupdf as fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Kinematics: point, body, velocity, acceleration.", fontsize=12)
    doc.save(str(path))
    doc.close()


def test_pdf(tmp_path):
    p = tmp_path / "doc.pdf"
    _make_pdf(p)
    parsed = parse_file(p, tmp_path / "media")
    assert "Kinematics" in parsed.text
    assert parsed.word_count > 0
    assert parsed.title == "doc"


def test_docx(tmp_path):
    from docx import Document

    doc = Document()
    doc.add_paragraph("Обыкновенные дроби. Числитель и знаменатель. Примеры задач.")
    p = tmp_path / "doc.docx"
    doc.save(str(p))
    parsed = parse_file(p, tmp_path / "media")
    assert "дроби" in parsed.text.lower()
    assert parsed.sentence_count >= 1


def test_txt(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("Текст материала для чтения. Второе предложение.", encoding="utf-8")
    parsed = parse_file(p, tmp_path / "media")
    assert parsed.word_count == 6
    assert parsed.sentence_count >= 2


def test_html(tmp_path):
    p = tmp_path / "h.html"
    p.write_text(
        "<html><head><title>Тема X</title></head><body><p>Абзац материала.</p></body></html>",
        encoding="utf-8",
    )
    parsed = parse_file(p, tmp_path / "media")
    assert "Абзац" in parsed.text
    assert parsed.title == "Тема X"


def test_md(tmp_path):
    p = tmp_path / "note.md"
    p.write_text("# Заголовок\n\nТекст заметки по теме.", encoding="utf-8")
    parsed = parse_file(p, tmp_path / "media")
    assert "заметки" in parsed.text
