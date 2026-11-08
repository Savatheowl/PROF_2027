"""OCR-адаптер: обёртка над indra_ocr (модель в models/ocr_model)."""

from __future__ import annotations

import logging
from pathlib import Path

from prof2027.ai.base import ModelAdapter

logger = logging.getLogger(__name__)


class OcrAdapter(ModelAdapter):
    """Распознавание текста в PDF-сканах и изображениях."""

    name = "indra-ocr"
    model_rel_path = "ocr_model"

    def __init__(self, models_dir=None, *, threads: int = 2, dpi: int = 300) -> None:
        super().__init__(models_dir)
        self.threads = threads
        self.dpi = dpi

    def load(self):
        import indra_ocr

        logger.info("Загрузка OCR-модели %s (threads=%d, dpi=%d)...",
                    self.model_path, self.threads, self.dpi)
        return indra_ocr.load(path=str(self.model_path), threads=self.threads, dpi=self.dpi)

    def extract_text(self, file_path: str | Path, *, max_pages: int = 20) -> str:
        """Текст из файла (PDF/изображение) через OCR."""
        conv = self.ensure_loaded()
        fp = str(file_path)
        pages: list[str] = []
        for _i, page in conv.convert(fp, max_pages=max_pages):
            try:
                markdown = getattr(page, "markdown", None) or ""
                if markdown.strip():
                    pages.append(markdown)
            except Exception:  # noqa: BLE001
                continue
        return "\n".join(pages)
