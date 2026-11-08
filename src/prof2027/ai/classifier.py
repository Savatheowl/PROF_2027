"""Классификатор d1-2B: один вызов на материал для метаданных + вызов на соответствие
24 методическим требованиям. Вопросы строятся динамически из каталогов (YAML)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prof2027.ai.base import ModelAdapter
from prof2027.domain import catalogs as cat
from prof2027.domain.enums import (
    ComplexityLevel,
    EducationLevel,
    Language,
    MaterialType,
    ModerationVerdict,
)

logger = logging.getLogger(__name__)

COMPLEXITY_ORDER = [
    ComplexityLevel.VERY_LOW.value,
    ComplexityLevel.LOW.value,
    ComplexityLevel.MEDIUM.value,
    ComplexityLevel.HIGH.value,
    ComplexityLevel.VERY_HIGH.value,
]

COMPLEXITY_RU = {
    ComplexityLevel.VERY_LOW.value: "Очень низкая",
    ComplexityLevel.LOW.value: "Низкая",
    ComplexityLevel.MEDIUM.value: "Средняя",
    ComplexityLevel.HIGH.value: "Высокая",
    ComplexityLevel.VERY_HIGH.value: "Очень высокая",
}

EDUCATION_LEVEL_RU = {
    EducationLevel.SCHOOL.value: "Школа (школьное образование)",
    EducationLevel.SPO.value: "СПО (колледж, техникум)",
    EducationLevel.UNIVERSITY.value: "Вуз (высшее образование)",
    EducationLevel.UNKNOWN.value: "Не удалось определить",
}

LANGUAGE_RU = {
    Language.RU.value: "Русский",
    Language.EN.value: "Английский",
    Language.OTHER.value: "Другой язык",
}

MAX_STATE_CHARS = 12_000

# Пороги уверенности: ниже значения считаем атрибут неопределённым
MIN_CONFIDENCE = {"category": 0.4, "subject": 0.3}

# Маркеры служебного текста, которые модель не должна повторять в теме
_TOPIC_MARKERS = ("УЧЕБНЫЙ МАТЕРИАЛ", "НАЗВАНИЕ", "МЕТОДИЧЕСКИЕ", "ПОДСКАЗКА")


@dataclass
class ClassificationResult:
    category: str | None
    subject: str | None
    topic: str | None
    education_level: str
    language: str
    material_type: str
    class_type: str
    complexity_score: float  # 1..5
    complexity_level: str
    confidences: dict[str, float] = field(default_factory=dict)


@dataclass
class ComplianceResult:
    scores: dict[str, float]  # требование -> вероятность «да»
    compliant: dict[str, bool]  # требование -> 0/1
    methodology_score: float  # 0..1
    verdict: str
    confidence: float = 0.0


class ClassifierAdapter(ModelAdapter):
    """d1-2B: метаданные и методическое соответствие одним вызовом на задачу."""

    name = "d1-2B"
    model_rel_path = "model_classifier"

    def __init__(self, models_dir: Path | str | None = None, *, root: Path | None = None) -> None:
        super().__init__(models_dir)
        self._root = root
        self._cached_catalog: dict | None = None

    # ------------------------------------------------------------------
    # Каталоги
    # ------------------------------------------------------------------

    def _catalogs(self) -> dict:
        if self._cached_catalog is None:
            self._cached_catalog = cat.all_catalogs(self._root)
        return self._cached_catalog

    def _categories(self) -> list[str]:
        return [c.name for c in self._catalogs()["categories"]]

    def _subjects(self) -> list[tuple[str, str]]:
        """Список (subject_name, category_name)."""
        out: list[tuple[str, str]] = []
        for c in self._catalogs()["categories"]:
            for s in c.subjects:
                out.append((s.name, c.name))
        return out

    def _material_types(self) -> list[tuple[str, str]]:
        return [(v.code, v.name) for v in self._catalogs()["material_types"]]

    def _class_types(self) -> list[tuple[str, str]]:
        return [(v.code, v.name) for v in self._catalogs()["class_types"]]

    def _requirements(self) -> list[cat.Requirement]:
        return self._catalogs()["methodology"]

    # ------------------------------------------------------------------
    # Вопросы
    # ------------------------------------------------------------------

    @staticmethod
    def _choice(instructions: str, criteria: dict) -> dict:
        return {"type": "choice", "instructions": instructions, "criteria": criteria}

    def _questions_metadata(self) -> dict[str, dict]:
        categories = self._categories()
        category_criteria = {name: None for name in categories}
        category_criteria["cannot_determine"] = "Не удалось определить категорию"

        return {
            "category": self._choice(
                "Определи категорию учебных дисциплин, к которой относится материал. "
                "Выбирай по содержанию и по названию материала.",
                category_criteria,
            ),
            "education_level": self._choice(
                "Определи уровень образования, для которого предназначен материал: школа, СПО или вуз.",
                EDUCATION_LEVEL_RU,
            ),
            "language": self._choice(
                "Определи язык, на котором написан материал: русский, английский или другой.",
                LANGUAGE_RU,
            ),
            "material_type": self._choice(
                "Определи основной тип учебного материала по его назначению и аудитории.",
                {code: name for code, name in self._material_types()},
            ),
            "class_type": self._choice(
                "Определи тип занятия, для которого предназначен материал.",
                {code: name for code, name in self._class_types()},
            ),
            "complexity": {
                "type": "score",
                "instructions": (
                    "Оцени сложность учебного материала по содержанию: понятия, терминология, "
                    "количество новых связей, сложность заданий. Не завышай из-за объёма текста."
                ),
                "criteria": [COMPLEXITY_RU[l] for l in COMPLEXITY_ORDER],
            },
            "topic": {
                "type": "text",
                "instructions": (
                    "Извлеки из материала короткое название темы (одно словосочетание, 2-5 слов), "
                    "как оно написано в тексте. Если темы нет, ничего не отвечай."
                ),
            },
        }

    def _subject_question(self, category: str | None, title: str | None = None) -> dict:
        """Вопрос о дисциплине: только предметы выбранной категории (точнее)."""
        subjects = [
            (name, cat_name)
            for name, cat_name in self._subjects()
            if category is None or cat_name == category
        ]
        if not subjects:
            subjects = self._subjects()
        criteria = {name: cat_name for name, cat_name in subjects}
        criteria["cannot_determine"] = "Не удалось определить дисциплину"
        instructions = (
            "Определи учебную дисциплину (предмет), к которой относится материал. "
            "Учитывай содержание, термины, учебные цели и название материала"
        )
        if title:
            instructions += f": «{title[:150]}»"
        instructions += "."
        return self._choice(instructions, criteria)

    def _questions_compliance(self) -> dict[str, dict]:
        requirements = self._requirements()
        criteria = {req.text: None for req in requirements}
        return {
            "compliance": {
                "type": "choice",
                "multiple": True,
                "instructions": (
                    "Оцени материал по каждому методическому требованию из списка. "
                    "Для каждого требования ответь да или нет — соответствует ли материал этому требованию."
                ),
                "criteria": criteria,
            }
        }

    # ------------------------------------------------------------------
    # Загрузка модели
    # ------------------------------------------------------------------

    def load(self):
        import torch
        from transformers import AutoModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        logger.info("Загрузка классификатора d1-2B (%s, %s)...", self.model_path, device)
        model = AutoModel.from_pretrained(str(self.model_path), trust_remote_code=True, dtype=dtype).to(device)
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Промпт-состояние
    # ------------------------------------------------------------------

    def _state(self, text: str, title: str, hint: str | None) -> str:
        criteria_text = self._criteria_text()
        head = (
            "МЕТОДИЧЕСКИЕ РЕКОМЕНДАЦИИ:\n"
            f"{criteria_text}\n\n"
        )
        if hint:
            head += f"ПОДСКАЗКА ОТ ПОЛЬЗОВАТЕЛЯ:\n{hint}\n\n"
        head += f"НАЗВАНИЕ УЧЕБНОГО МАТЕРИАЛА:\n{title}\n\nУЧЕБНЫЙ МАТЕРИАЛ:\n{text}"
        return head[:MAX_STATE_CHARS]

    def _criteria_text(self) -> str:
        lines: list[str] = []
        current = None
        for req in self._requirements():
            if req.category != current:
                lines.append(req.category)
                current = req.category
            lines.append(f"- {req.text}.")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Вызовы
    # ------------------------------------------------------------------

    def _call(self, state: str, questions: dict[str, dict]) -> dict[str, Any]:
        model = self.ensure_loaded()
        import torch

        with torch.inference_mode():
            out = model.system_one(state, questions)
        return out["answers"]

    def classify(self, text: str, title: str, hint: str | None = None) -> ClassificationResult:
        state = self._state(text, title, hint)

        # 1-й проход: категория и прочие метаданные
        answers = self._call(state, self._questions_metadata())

        confidences: dict[str, float] = {}

        category = self._answer_choice(answers.get("category"))
        confidences["category"] = self._answer_confidence(answers.get("category"))
        if category == "cannot_determine" or confidences["category"] < MIN_CONFIDENCE["category"]:
            category = None
            confidences["category"] = 0.0

        # 2-й проход: дисциплина только среди предметов выбранной категории
        subject = None
        confidences["subject"] = 0.0
        subject_answer = self._call(state, {"subject": self._subject_question(category, title)})["subject"]
        subject = self._answer_choice(subject_answer)
        confidences["subject"] = self._answer_confidence(subject_answer)
        if subject == "cannot_determine" or confidences["subject"] < MIN_CONFIDENCE["subject"]:
            subject = None
            confidences["subject"] = 0.0

        education_level = self._answer_choice(answers.get("education_level"), EducationLevel.UNKNOWN.value)
        language = self._answer_choice(answers.get("language"), Language.OTHER.value)
        material_type = self._answer_choice(answers.get("material_type"), MaterialType.CANNOT_DETERMINE.value)
        class_type = self._answer_choice(answers.get("class_type"), "unknown")
        confidences["education_level"] = self._answer_confidence(answers.get("education_level"))
        confidences["language"] = self._answer_confidence(answers.get("language"))
        confidences["material_type"] = self._answer_confidence(answers.get("material_type"))
        confidences["class_type"] = self._answer_confidence(answers.get("class_type"))

        complexity_index = self._answer_score_index(answers.get("complexity"))
        complexity_index = max(0, min(len(COMPLEXITY_ORDER) - 1, complexity_index))
        complexity_score = float(complexity_index + 1)
        complexity_level = COMPLEXITY_ORDER[complexity_index]
        confidences["complexity"] = self._answer_confidence(answers.get("complexity"))

        topic = self._answer_topic(answers.get("topic"), title)

        return ClassificationResult(
            category=category,
            subject=subject,
            topic=topic,
            education_level=education_level,
            language=language,
            material_type=material_type,
            class_type=class_type,
            complexity_score=complexity_score,
            complexity_level=complexity_level,
            confidences=confidences,
        )

    def moderate(self, text: str, title: str, hint: str | None = None,
                 thresholds: dict | None = None) -> ComplianceResult:
        state = self._state(text, title, hint)
        answers = self._call(state, self._questions_compliance())
        raw = answers.get("compliance", {})
        probabilities: dict[str, float] = raw.get("probabilities", {})
        chosen: list[str] = raw.get("choices", [])

        requirements = self._requirements()
        scores: dict[str, float] = {}
        compliant: dict[str, bool] = {}
        for req in requirements:
            p = probabilities.get(req.text, 0.0)
            scores[req.text] = float(p)
            compliant[req.text] = req.text in chosen or p >= 0.5

        methodology_score = sum(scores.values()) / len(scores) if scores else 0.0
        verdict = self._verdict(methodology_score, thresholds or {})
        return ComplianceResult(
            scores=scores,
            compliant=compliant,
            methodology_score=round(methodology_score, 3),
            verdict=verdict,
            confidence=float(raw.get("confidence", 0.0)),
        )

    # ------------------------------------------------------------------
    # Разбор ответов
    # ------------------------------------------------------------------

    @staticmethod
    def _answer_choice(answer: dict | None, default: str | None = None) -> str | None:
        if not isinstance(answer, dict):
            return default
        choice = answer.get("choice")
        return choice if isinstance(choice, str) and choice else default

    @staticmethod
    def _answer_confidence(answer: dict | None) -> float:
        if not isinstance(answer, dict):
            return 0.0
        return round(float(answer.get("confidence", 0.0)), 3)

    @staticmethod
    def _answer_score_index(answer: dict | None) -> int:
        if not isinstance(answer, dict):
            return 0
        value = answer.get("score")
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _answer_topic(answer: dict | None, fallback: str) -> str | None:
        if not isinstance(answer, dict):
            return fallback or None
        text = answer.get("text")
        found = answer.get("found", 0.0)
        if not text or float(found) < 0.5:
            return fallback or None
        clean = " ".join(str(text).split())
        upper = clean.upper()
        if any(marker in upper for marker in _TOPIC_MARKERS):
            return fallback or None
        if len(clean) > 80 or "\n" in clean:
            return fallback or None
        if len(clean.split()) > 8:  # тема — короткое словосочетание, не абзац
            return fallback or None
        return clean or (fallback or None)

    @staticmethod
    def _verdict(score: float, thresholds: dict) -> str:
        approved = float(thresholds.get("approved_threshold", 0.7))
        needs_review = float(thresholds.get("needs_review_threshold", 0.4))
        if score >= approved:
            return ModerationVerdict.APPROVED.value
        if score >= needs_review:
            return ModerationVerdict.NEEDS_REVIEW.value
        return ModerationVerdict.REJECTED.value
