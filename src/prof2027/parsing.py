"""Парсинг файлов: извлечение текста, заголовка и медиа.

Поддерживаемые форматы: PDF (цифровой и скан), DOCX, DOC, PPTX, RTF, HTML, TXT/MD,
изображения, видео. OCR используется для сканов PDF, изображений и кадров видео.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

PDF_EXTS = {".pdf"}
DOCX_EXTS = {".docx"}
DOC_EXTS = {".doc"}
PPTX_EXTS = {".pptx"}
RTF_EXTS = {".rtf"}
HTML_EXTS = {".html", ".htm"}
TEXT_EXTS = {".txt", ".md"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}

_RTF_RE = re.compile(r"\\[a-z]+-?\d* ?|\\'[0-9a-fA-F]{2}|\\\*|\\\{|\\\}")


@dataclass
class ParsedFile:
    text: str = ""
    title: str = ""
    media: list[dict] = field(default_factory=list)
    char_count: int = 0
    word_count: int = 0
    sentence_count: int = 0

    def add_media(self, media_type: str, path: str = "", description: str = "", ocr_text: str = "") -> None:
        self.media.append(
            {"type": media_type, "ordinal": len(self.media) + 1, "path": path,
             "description": description, "ocr_text": ocr_text}
        )

    def finish_counts(self) -> "ParsedFile":
        self.char_count = len(self.text)
        self.word_count = len(self.text.split())
        self.sentence_count = max(0, len(re.findall(r"[.!?…]+(?:\s|$)", self.text)))
        return self


def _title_from_name(path: Path, doc_title: str = "") -> str:
    return (doc_title or "").strip() or path.stem


def extract_pdf(path: Path, media_dir: Path) -> ParsedFile:
    import pymupdf as fitz  # PyMuPDF

    doc = fitz.open(str(path))
    text = "".join(page.get_text("text") for page in doc)
    title = doc.metadata.get("title", "") or ""
    parsed = ParsedFile(text=text.strip(), title=_title_from_name(path, title))
    parsed.add_media("document", str(path), description=f"PDF, {len(doc)} страниц")
    doc.close()
    return parsed.finish_counts()


def extract_docx(path: Path, media_dir: Path) -> ParsedFile:
    import docx

    doc = docx.Document(str(path))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    text = "\n".join(parts)
    title = getattr(doc.core_properties, "title", "") or ""
    parsed = ParsedFile(text=text.strip(), title=_title_from_name(path, title))

    media_dir.mkdir(parents=True, exist_ok=True)
    for i, rel in enumerate(doc.part.rels.values(), 1):
        if "image" in getattr(rel, "reltype", ""):
            try:
                blob = rel.target_part.blob
                ext = Path(rel.target_part.partname).suffix or ".png"
                fp = media_dir / f"img_{i}{ext}"
                fp.write_bytes(blob)
                parsed.add_media("image", str(fp), description=f"Встроенное изображение {i}")
            except Exception:
                continue
    return parsed.finish_counts()


def extract_pptx(path: Path, media_dir: Path) -> ParsedFile:
    from pptx import Presentation

    prs = Presentation(str(path))
    parts = []
    media: list[dict] = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    t = "".join(run.text for run in para.runs).strip()
                    if t:
                        parts.append(t)
            if shape.shape_type == 13 and hasattr(shape, "image"):  # PICTURE
                try:
                    media_dir.mkdir(parents=True, exist_ok=True)
                    fname = Path(shape.image.filename or "").name or "img.png"
                    fp = media_dir / f"img_{len(media) + 1}_{fname}"
                    fp.write_bytes(shape.image.blob)
                    media.append({"type": "image", "ordinal": len(media) + 1,
                                  "path": str(fp), "description": "Слайд: изображение", "ocr_text": ""})
                except Exception:
                    continue
    text = "\n".join(parts)
    parsed = ParsedFile(text=text.strip(), title=_title_from_name(path))
    parsed.media = media
    return parsed.finish_counts()


def extract_html(path: Path) -> ParsedFile:
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
        text = soup.get_text(separator="\n")
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
    except Exception:
        text = re.sub(r"<[^>]+>", " ", path.read_text(encoding="utf-8", errors="ignore"))
        title = ""
    return ParsedFile(text=re.sub(r"\n{2,}", "\n", text).strip(), title=_title_from_name(path, title)).finish_counts()


def extract_rtf(path: Path) -> ParsedFile:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    text = _RTF_RE.sub(" ", raw).replace("\\par", "\n").replace("\\line", "\n")
    text = re.sub(r"\{\s*\{\s*", "", text).replace("{", "").replace("}", "")
    return ParsedFile(text=re.sub(r"\s+", " ", text).strip(), title=_title_from_name(path)).finish_counts()


def extract_txt(path: Path) -> ParsedFile:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return ParsedFile(text=text.strip(), title=_title_from_name(path)).finish_counts()


def extract_image(path: Path, media_dir: Path, ocr=None) -> ParsedFile:
    text = ""
    if ocr is not None and ocr.is_available():
        try:
            text = ocr.convert(path, max_pages=1)
        except Exception:
            text = ""
    parsed = ParsedFile(text=text.strip(), title=_title_from_name(path))
    parsed.add_media("image", str(path), description="Изображение", ocr_text=text[:500])
    return parsed.finish_counts()


def extract_video(path: Path, media_dir: Path, ocr=None, n_frames: int = 3) -> ParsedFile:
    import cv2

    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    duration = round(total / fps, 2) if fps else 0.0
    frames = sorted(set(int(i * max(total, 1) / (n_frames + 1)) for i in range(1, n_frames + 1)))
    text_parts = []
    media_dir.mkdir(parents=True, exist_ok=True)
    for idx, fr in enumerate(frames, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, img = cap.read()
        if not ok:
            continue
        fp = media_dir / f"frame_{idx:02d}.png"
        cv2.imwrite(str(fp), img)
        if ocr is not None and ocr.is_available():
            try:
                text_parts.append(ocr.convert(fp, max_pages=1))
            except Exception:
                pass
    cap.release()
    parsed = ParsedFile(text="\n".join(t for t in text_parts if t.strip()).strip(), title=_title_from_name(path))
    parsed.add_media("video", str(path), description=f"Видео, {duration:.1f} с, {total} кадров")
    return parsed.finish_counts()


def parse_file(path: Path, media_dir: Path, ocr=None) -> ParsedFile:
    """Основной диспетчер парсинга по расширению."""
    ext = path.suffix.lower()
    if ext in PDF_EXTS:
        parsed = extract_pdf(path, media_dir)
        if len(parsed.text) < 40 and ocr is not None and ocr.is_available():
            try:
                parsed.text = ocr.convert(path) or parsed.text
            except Exception:
                pass
        return parsed.finish_counts()
    if ext in DOCX_EXTS:
        return extract_docx(path, media_dir)
    if ext in DOC_EXTS:
        # legacy DOC: пробуем docx-конвертацию нет; берём OCR при доступности, иначе пусто
        text = ""
        if ocr is not None and ocr.is_available():
            try:
                text = ocr.convert(path)
            except Exception:
                text = ""
        return ParsedFile(text=text.strip(), title=_title_from_name(path)).finish_counts()
    if ext in PPTX_EXTS:
        return extract_pptx(path, media_dir)
    if ext in RTF_EXTS:
        return extract_rtf(path)
    if ext in HTML_EXTS:
        return extract_html(path)
    if ext in TEXT_EXTS:
        return extract_txt(path)
    if ext in IMAGE_EXTS:
        return extract_image(path, media_dir, ocr)
    if ext in VIDEO_EXTS:
        return extract_video(path, media_dir, ocr)
    raise ValueError(f"неподдерживаемый формат: {ext}")
