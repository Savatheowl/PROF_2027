"""Извлечение текста и медиа из файлов различных форматов.

PDF: цифровой — PyMuPDF, скан — OCR (indra-ocr).
DOCX/PPTX/HTML/TXT/MD/RTF/EPUB/FB2 — текстовые парсеры.
Изображения — описание (OCR при наличии).
Аудио (MP3) — метаданные.
Видео — метаданные и, опционально, кадры.
"""

from __future__ import annotations

import logging
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from prof2027.core.magic import detect

logger = logging.getLogger(__name__)

# Минимальное число слов, чтобы считать PDF «цифровым»
_MIN_TEXT_WORDS = 40

_SENTENCE_END = re.compile(r"[.!?…]+(?:[)\"»'’]|$)")
_WHITESPACE = re.compile(r"\s+")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


@dataclass
class ParsedDocument:
    text: str = ""
    media: list[dict] = field(default_factory=list)
    file_type: str | None = None

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def sentence_count(self) -> int:
        return len(_SENTENCE_END.findall(self.text))

    @property
    def media_count(self) -> int:
        return len(self.media)


def _clean(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Парсеры по типам
# ---------------------------------------------------------------------------


def _parse_pdf(path: Path, ocr: Callable | None, max_pages: int) -> ParsedDocument:
    import fitz

    doc = fitz.open(path)
    chunks: list[str] = []
    media: list[dict] = []
    ordinal = 0
    try:
        for page in doc:
            chunks.append(page.get_text("text") or "")
            for img in page.get_images(full=True):
                ordinal += 1
                media.append({"type": "image", "ordinal": ordinal, "description": "Встроенное изображение"})
            if len(chunks) >= max_pages:
                break
    finally:
        doc.close()

    text = _clean("\n".join(chunks))
    # Если это скан — OCR
    if ocr is not None and len(text.split()) < _MIN_TEXT_WORDS:
        try:
            ocr_text = ocr(str(path), max_pages=max_pages)
            if ocr_text and len(ocr_text.strip()) > len(text.strip()):
                text = _clean(ocr_text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("OCR не сработал для %s: %s", path.name, exc)
    return ParsedDocument(text=text, media=media, file_type="pdf")


def _parse_docx(path: Path, _ocr, _max_pages) -> ParsedDocument:
    import docx

    d = docx.Document(str(path))
    chunks = [p.text for p in d.paragraphs if p.text.strip()]
    for table in d.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    chunks.append(cell.text)
    media: list[dict] = []
    try:
        for i, shape in enumerate(d.inline_shapes, start=1):
            media.append({"type": "image", "ordinal": i, "description": "Встроенное изображение"})
    except Exception:  # noqa: BLE001
        pass
    return ParsedDocument(text=_clean("\n".join(chunks)), media=media, file_type="docx")


def _parse_pptx(path: Path, _ocr, _max_pages) -> ParsedDocument:
    from pptx import Presentation

    prs = Presentation(str(path))
    chunks: list[str] = []
    media: list[dict] = []
    ordinal = 0
    for slide_index, slide in enumerate(prs.slides, start=1):
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    text = "".join(run.text for run in para.runs).strip()
                    if text:
                        chunks.append(text)
            if shape.shape_type == 13:  # PICTURE
                ordinal += 1
                media.append({"type": "image", "ordinal": ordinal,
                              "description": f"Изображение на слайде {slide_index}"})
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                chunks.append(f"Заметки: {notes}")
    return ParsedDocument(text=_clean("\n".join(chunks)), media=media, file_type="pptx")


def _parse_html(path: Path, _ocr, _max_pages) -> ParsedDocument:
    from bs4 import BeautifulSoup

    raw = _read_text_bytes(path)
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    text = _clean(soup.get_text(" ", strip=True))
    media = [{"type": "image", "ordinal": i, "description": img.get("alt") or "Изображение"}
             for i, img in enumerate(soup.find_all("img"), start=1)]
    return ParsedDocument(text=text, media=media, file_type="html")


def _parse_epub(path: Path, _ocr, _max_pages) -> ParsedDocument:
    from bs4 import BeautifulSoup

    chunks: list[str] = []
    media: list[dict] = []
    ordinal = 0

    pages: list[tuple[str, bytes]] = []
    try:
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if not name.lower().endswith((".xhtml", ".html", ".htm")):
                    continue
                try:
                    pages.append((name, zf.read(name)))
                except Exception:  # noqa: BLE001
                    continue
    except zipfile.BadZipFile:
        # Повреждённый/обрезанный epub: извлекаем локальные записи напрямую
        logger.warning("EPUB %s: повреждён, читаю локальные записи", path.name)
        for name, raw in _iter_local_zip_entries(path):
            if not name.lower().endswith((".xhtml", ".html", ".htm")) or raw is None:
                continue
            pages.append((name, raw))

    for name, raw in pages:
        soup = BeautifulSoup(raw, "lxml")
        for tag in soup(["script", "style"]):
            tag.decompose()
        chunks.append(soup.get_text(" ", strip=True))
        for img in soup.find_all("img"):
            ordinal += 1
            media.append({"type": "image", "ordinal": ordinal,
                          "description": img.get("alt") or "Иллюстрация книги"})
    return ParsedDocument(text=_clean("\n".join(chunks)), media=media, file_type="epub")


def _parse_fb2(path: Path, _ocr, _max_pages) -> ParsedDocument:
    from bs4 import BeautifulSoup

    raw = _read_text_bytes(path)
    soup = BeautifulSoup(raw, "lxml-xml")
    body = soup.find("body")
    text = _clean(body.get_text(" ", strip=True)) if body is not None else _clean(raw)
    media = [{"type": "image", "ordinal": i, "description": "Иллюстрация книги"}
             for i, _img in enumerate(soup.find_all("image"), start=1)]
    return ParsedDocument(text=text, media=media, file_type="fb2")


def _parse_txt(path: Path, _ocr, _max_pages) -> ParsedDocument:
    return ParsedDocument(text=_clean(_read_text_bytes(path)), file_type="txt")


def _parse_doc(path: Path, ocr: Callable | None, max_pages: int) -> ParsedDocument:
    """.doc: пробуем как DOCX; старые .doc часто хранят RTF — fallback."""
    try:
        return _parse_docx(path, ocr, max_pages)
    except Exception:  # noqa: BLE001
        return _parse_rtf(path, ocr, max_pages)


def _parse_rtf(path: Path, _ocr, _max_pages) -> ParsedDocument:
    raw = _read_text_bytes(path)
    text = _strip_rtf(raw)
    return ParsedDocument(text=_clean(text), file_type="rtf")


def _parse_image(path: Path, ocr: Callable | None, _max_pages) -> ParsedDocument:
    from PIL import Image

    media: list[dict] = []
    try:
        with Image.open(path) as img:
            width, height = img.size
            fmt = img.format
    except Exception:  # noqa: BLE001
        width = height = None
        fmt = None
    description = f"Изображение{f' ({fmt})' if fmt else ''}"
    media.append({"type": "image", "ordinal": 1, "description": description,
                  "width": width, "height": height})
    text = ""
    if ocr is not None:
        try:
            text = _clean(ocr(str(path), max_pages=1))
        except Exception as exc:  # noqa: BLE001
            logger.warning("OCR изображения не сработал для %s: %s", path.name, exc)
    return ParsedDocument(text=text, media=media, file_type=path.suffix.lstrip("."))


def _parse_audio(path: Path, _ocr, _max_pages) -> ParsedDocument:
    duration = None
    title = None
    try:
        from mutagen import File as MutagenFile

        audio = MutagenFile(str(path))
        if audio is not None:
            duration = float(audio.info.length) if getattr(audio.info, "length", None) else None
            tags = audio.tags
            if tags:
                title = _first_tag(tags, "TIT2", "title")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Аудио-метаданные не прочитаны для %s: %s", path.name, exc)
    description = f"Аудиозапись{': ' + title if title else ''}"
    media = [{"type": "audio", "ordinal": 1, "description": description,
              "duration_sec": round(duration, 1) if duration else None}]
    return ParsedDocument(text="", media=media, file_type="mp3")


def _parse_video(path: Path, _ocr, _max_pages) -> ParsedDocument:
    duration = None
    width = height = None
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
            if fps and frames:
                duration = frames / fps
        cap.release()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Видео-метаданные не прочитаны для %s: %s", path.name, exc)
    description = "Видеозапись учебного материала"
    media = [{"type": "video", "ordinal": 1, "description": description,
              "duration_sec": round(duration, 1) if duration else None,
              "width": width, "height": height}]
    return ParsedDocument(text="", media=media, file_type="video")


# ---------------------------------------------------------------------------
# Вспомогательное
# ---------------------------------------------------------------------------


def _read_text_bytes(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8", "cp1251", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _first_tag(tags, *names) -> str | None:
    for name in names:
        value = tags.get(name)
        if value:
            text = str(value)
            return text
    return None


def _iter_local_zip_entries(path: Path):
    """Обходит локальные заголовки zip (работает и для обрезанных файлов).

    Возвращает (имя, распакованное содержимое) для stored/deflated записей.
    """
    import zlib

    data = path.read_bytes()
    n = len(data)
    offset = 0
    while offset + 30 <= n:
        if data[offset : offset + 4] != b"PK\x03\x04":
            break
        method = int.from_bytes(data[offset + 8 : offset + 10], "little")
        comp_size = int.from_bytes(data[offset + 18 : offset + 22], "little")
        name_len = int.from_bytes(data[offset + 26 : offset + 28], "little")
        extra_len = int.from_bytes(data[offset + 28 : offset + 30], "little")
        name = data[offset + 30 : offset + 30 + name_len].decode("utf-8", errors="replace")
        content_start = offset + 30 + name_len + extra_len
        content_end = content_start + comp_size
        if content_end > n:
            break
        raw = data[content_start:content_end]
        if method == 0:
            content = raw
        elif method == 8:
            try:
                content = zlib.decompress(raw)
            except Exception:  # noqa: BLE001
                content = None
        else:
            content = None
        yield name, content
        next_offset = content_end
        if next_offset <= offset:
            break
        offset = next_offset


def _strip_rtf(raw: str) -> str:
    """Грубое извлечение текста из RTF: удаляет управляющие конструкции."""
    text = re.sub(r"\\[a-z]+\d?(?:-\d+)?\s?", "", raw)
    text = re.sub(r"\\'[0-9a-fA-F]{2}", " ", text)
    text = re.sub(r"\{|\}", "", text)
    return text


PARSERS: dict[str, Callable] = {
    ".pdf": _parse_pdf,
    ".docx": _parse_docx,
    ".doc": _parse_doc,
    ".pptx": _parse_pptx,
    ".ppt": _parse_pptx,
    ".html": _parse_html,
    ".htm": _parse_html,
    ".epub": _parse_epub,
    ".fb2": _parse_fb2,
    ".txt": _parse_txt,
    ".md": _parse_txt,
    ".rtf": _parse_rtf,
}


def parse_file(path: Path, ocr: Callable | None = None, *, max_pages: int = 20) -> ParsedDocument:
    """Разбирает файл по расширению (с OCR-fallback для сканов PDF и изображений)."""
    ext = path.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return _parse_image(path, ocr, max_pages)
    if ext in VIDEO_EXTENSIONS:
        return _parse_video(path, ocr, max_pages)
    if ext == ".mp3":
        return _parse_audio(path, ocr, max_pages)
    parser = PARSERS.get(ext)
    if parser is None:
        detected = detect(path.read_bytes()[:65536]) or ""
        raise ValueError(f"Формат {ext or '(без расширения)'} не поддерживается")
    return parser(path, ocr, max_pages)
