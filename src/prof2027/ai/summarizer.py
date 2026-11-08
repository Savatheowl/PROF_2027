"""Суммаризатор: LaciaSUM (RuT5) с экстрактивным fallback-ом."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from prof2027.ai.base import ModelAdapter

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")


class SummarizerAdapter(ModelAdapter):
    """Аннотации материалов."""

    name = "LaciaSUM"
    model_rel_path = "anotation-model"

    def load(self):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info("Загрузка суммаризатора LaciaSUM (%s)...", self.model_path)
        tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(self.model_path))
        model.eval()
        return tokenizer, model

    def summarize(self, text: str, *, max_length: int = 150) -> str:
        if not text or not text.strip():
            return ""
        try:
            tokenizer, model = self.ensure_loaded()
            input_text = "summarize: " + text
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            summary_ids = model.generate(
                inputs["input_ids"], max_length=max_length, num_beams=4, early_stopping=True
            )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
            return summary or self.extractive(text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LaciaSUM не сработал, экстрактивный fallback: %s", exc)
            return self.extractive(text)

    @staticmethod
    def extractive(text: str, *, max_sentences: int = 3, max_words: int = 120) -> str:
        """Fallback: первые содержательные предложения."""
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
        if not sentences:
            return " ".join(text.split()[: max_words])
        picked: list[str] = []
        words = 0
        for sentence in sentences:
            count = len(sentence.split())
            if words + count > max_words and picked:
                break
            picked.append(sentence)
            words += count
            if len(picked) >= max_sentences:
                break
        return " ".join(picked)
