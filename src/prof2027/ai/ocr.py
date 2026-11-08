"""Адаптер OCR: indra-ocr (ONNX). Модель: models/ocr_model."""

from __future__ import annotations

from prof2027.ai.base import Adapter


class Ocr(Adapter):
    def __init__(self, model_dir, threads: int = 4, dpi: int = 300) -> None:
        super().__init__(model_dir)
        self.threads = threads
        self.dpi = dpi

    def is_available(self) -> bool:
        try:
            import indra_ocr  # noqa: F401
            return True
        except Exception:
            return False

    def load(self):
        import indra_ocr

        self._conv = indra_ocr.load(
            path=str(self.model_dir / "ocr_model"),
            threads=self.threads,
            dpi=self.dpi,
        )

    def convert(self, fp, max_pages: int = 20) -> str:
        conv = self._ensure()
        text: list[str] = []
        try:
            for page in conv.convert(fp, max_pages=max_pages):
                md = getattr(page, "markdown", None)
                if md:
                    text.append(md)
        except Exception:
            return ""
        return "\n".join(text)
