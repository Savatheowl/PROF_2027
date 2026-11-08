"""Определение типа файла по сигнатуре (magic bytes).

Чистый Python, без системных зависимостей — важно для офлайн-сборки.
Каждый тип проверяется заголовком; для контейнеров с расширением
внутри (zip: docx/pptx/epub, RIFF: avi/webp) — дополнительным признаком.

Функция `detect(data: bytes) -> str | None` возвращает расширение
(например "pdf"), если сигнатура распознана, иначе None.
"""

from __future__ import annotations


def detect(data: bytes) -> str | None:
    """Определяет расширение по первым байтам файла."""
    if not data:
        return None

    # UTF-8 BOM
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]

    # PDF
    if data.startswith(b"%PDF-"):
        return "pdf"

    # PNG
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"

    # JPEG
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"

    # BMP
    if data.startswith(b"BM"):
        return "bmp"

    # TIFF (little / big endian)
    if data.startswith(b"II\x2a\x00") or data.startswith(b"MM\x00\x2a"):
        return "tiff"

    # WEBP (RIFF....WEBP)
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"

    # GIF
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "gif"

    # AVI (RIFF....AVI )
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"AVI ":
        return "avi"

    # MP4 / MOV / M4V (ftyp box на байте 4)
    if len(data) >= 12 and data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in (b"isom", b"iso2", b"mp42", b"mp41", b"avc1", b"MSNV"):
            return "mp4"
        if brand in (b"qt  ", b"M4V "):
            return "mov"
        return "mp4"

    # MKV / WebM (EBML)
    if data.startswith(b"\x1a\x45\xdf\xa3"):
        return "mkv"

    # OLE2 (старые .doc / .ppt)
    if data.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        return _ole_kind(data)  # "doc" или "ppt"

    # ZIP-контейнеры: docx / pptx / epub
    if data.startswith(b"PK\x03\x04"):
        return _zip_kind(data)

    # MP3 (ID3v2 или MPEG frame sync)
    if data.startswith(b"ID3") or _looks_like_mp3(data):
        return "mp3"

    # FB2 / HTML / TXT / MD / RTF — текстовые
    if data.startswith(b"{\\rtf"):
        return "rtf"
    if data.lstrip().startswith(b"<") and _looks_like_text(data):
        head = data[:4000].lstrip().lower()
        if b"<fictionbook" in head or b"<fb2" in head:
            return "fb2"
        if b"<html" in head or b"<!doctype" in head:
            return "html"
        return "html"
    if _looks_like_text(data):
        return "txt"

    return None


def _looks_like_mp3(data: bytes) -> bool:
    if len(data) < 4:
        return False
    # MPEG Audio Layer I/II/III frame sync: 11 бит подряд
    b = data[:4]
    if b[0] == 0xFF and (b[1] & 0xE0) == 0xE0:
        version = (b[1] >> 3) & 0x03
        layer = (b[1] >> 1) & 0x03
        return version != 0x01 and layer != 0x00  # запрещённые комбинации
    return False


def _looks_like_text(data: bytes) -> bool:
    """Текст без NUL-байтов: допустимо для html/txt/fb2/rtf."""
    sample = data[:4096]
    if b"\x00" in sample:
        return False
    # Пытаемся декодировать как UTF-8 (или Latin-1) с заменой ошибок
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        try:
            sample.decode("cp1251")
        except UnicodeDecodeError:
            return False
    return True


def _zip_kind(data: bytes) -> str:
    """docx / pptx / epub / xlsx по внутренней структуре zip.

    Работает по фрагменту заголовка (первые десятки КБ), т.к. центральный
    каталог zip лежит в конце файла и в короткий header не попадает.
    """
    names, epub = _scan_zip_entries(data)
    if not names:
        return "zip"
    lowered = {n.lower() for n in names}
    if any(n.startswith("word/") for n in lowered):
        return "docx"
    if any(n.startswith("ppt/") for n in lowered):
        return "pptx"
    if any(n.startswith("xl/") for n in lowered):
        return "xlsx"
    if epub:
        return "epub"
    if "mimetype" in lowered:
        return "epub"
    return "zip"


def _scan_zip_entries(data: bytes) -> tuple[list[str], bool]:
    """Читает имена записей zip из локальных заголовков. Возвращает (имена, is_epub)."""
    names: list[str] = []
    epub = False
    offset = 0
    limit = min(len(data), 64 * 1024)
    while offset + 30 <= limit:
        if data[offset : offset + 4] != b"PK\x03\x04":
            break
        flags = int.from_bytes(data[offset + 6 : offset + 8], "little")
        method = int.from_bytes(data[offset + 8 : offset + 10], "little")
        comp_size = int.from_bytes(data[offset + 18 : offset + 22], "little")
        name_len = int.from_bytes(data[offset + 26 : offset + 28], "little")
        extra_len = int.from_bytes(data[offset + 28 : offset + 30], "little")
        name_start = offset + 30
        name_end = name_start + name_len
        if name_end > limit:
            break
        name = data[name_start:name_end].decode("utf-8", errors="replace")
        names.append(name)
        if name.lower() == "mimetype":
            content_start = name_end + extra_len
            content = data[content_start : content_start + 32]
            # mimetype хранится без сжатия (method 0), содержимое в начале
            if method == 0 and content.startswith(b"application/epub+zip"):
                epub = True
        next_offset = name_end + extra_len + comp_size
        if next_offset <= offset:  # защита от зацикливания
            break
        offset = next_offset
    return names, epub


def _ole_kind(data: bytes) -> str:
    """doc или ppt по сигнатуре OLE-потока."""
    try:
        import struct

        # Ищем имя первого потока в заголовке (приблизительно)
        text = data[:4096]
        if b"PowerPoint" in text or b"Ppt" in text:
            return "ppt"
        if b"WordDocument" in text or b"MSWordDoc" in text:
            return "doc"
    except Exception:  # pragma: no cover
        pass
    return "doc"
