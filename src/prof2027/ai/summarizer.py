"""Адаптер суммаризатора: LaciaSUM. Модель: models/anotation-model.

Fallback — экстрактивное резюме (первые предложения, до 120 слов).
"""

from __future__ import annotations

import re

from prof2027.ai.base import Adapter

_SENT_RE = re.compile(r"(?<=[.!?…])\s+")


class Summarizer(Adapter):
    def is_available(self) -> bool:
        try:
            import transformers  # noqa: F401
            return True
        except Exception:
            return False

    def load(self):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        path = str(self.model_dir / "anotation-model")
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(path)
        self._obj = (self._tokenizer, self._model)

    def summarize(self, text: str, max_length: int = 150) -> str:
        text = (text or "").strip()
        if len(text) < 20:
            return ""
        try:
            tok = self._ensure()
            inputs = tok("summarize: " + text[:3000], return_tensors="pt",
                         max_length=512, truncation=True)
            ids = self._model.generate(inputs["input_ids"], max_length=max_length,
                                       num_beams=4, early_stopping=True)
            return tok.decode(ids[0], skip_special_tokens=True).strip()
        except Exception:
            return self._fallback(text)

    @staticmethod
    def _fallback(text: str) -> str:
        picked: list[str] = []
        words = 0
        for sent in _SENT_RE.split(text):
            w = len(sent.split())
            if words + w > 120:
                break
            picked.append(sent.strip())
            words += w
            if len(picked) >= 3:
                break
        return " ".join(picked)
