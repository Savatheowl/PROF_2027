"""Адаптер классификатора: d1-2B (LiquidAI). Модель: models/model_classifier.

Один вызов system_one определяет: дисциплину, категорию, уровень образования, язык,
тип материала, тип занятия, тему, сложность и соответствие каждому методическому
требованию (24 noul-вопроса). Вопросы строятся динамически из каталогов.
"""

from __future__ import annotations

from prof2027.ai.base import Adapter
from prof2027.catalogs import Catalogs

STATE_MAX_CHARS = 30000


def criteria_text(catalogs: Catalogs) -> str:
    lines = []
    for cat in catalogs.methodology:
        lines.append(cat["category"])
        for req in cat["requirements"]:
            lines.append(f"- {req}.")
    return "\n".join(lines)


class Classifier(Adapter):
    def __init__(self, model_dir, catalogs: Catalogs) -> None:
        super().__init__(model_dir)
        self.catalogs = catalogs

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            return True
        except Exception:
            return False

    def load(self):
        import torch
        from transformers import AutoModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        self._model = AutoModel.from_pretrained(
            str(self.model_dir / "model_classifier"),
            trust_remote_code=True,
            dtype=dtype,
        ).to(self.device)
        self._model.eval()
        self._obj = self._model

    # ------------------------------------------------------------------ #
    def build_questions(self) -> dict:
        cat = self.catalogs
        subjects = {name: name for name in cat.subject_names}
        subjects["cannot_determine"] = "Не удалось определить"

        questions = {
            "subject": {
                "type": "choice",
                "instructions": (
                    "Определи дисциплину (учебный предмет), к которой относится материал. "
                    "Анализируй основное содержание, а не заголовок. Выбирай одну дисциплину. "
                    "Если содержание недостаточно, выбирай cannot_determine."
                ),
                "criteria": subjects,
            },
            "education_level": {
                "type": "choice",
                "instructions": "Определи уровень образования, для которого предназначен материал.",
                "criteria": {
                    "school": "Школа (школьное образование)",
                    "spo": "Колледж/техникум (среднее профессиональное образование)",
                    "university": "Вуз (высшее образование)",
                    "unknown": "Не удалось определить",
                },
            },
            "language": {
                "type": "choice",
                "instructions": "Определи язык, на котором написан материал.",
                "criteria": {"ru": "Русский", "en": "Английский", "other": "Другой язык"},
            },
            "material_type": {
                "type": "choice",
                "instructions": "Определи тип учебного материала по его назначению и аудитории.",
                "criteria": {
                    "student_material": "Учебный материал для обучающегося",
                    "teacher_material": "Материал для педагога (методическое пособие)",
                    "practice": "Упражнения и задачи",
                    "assessment": "Контрольные и проверочные работы",
                    "practical_lab": "Практические и лабораторные работы",
                    "project": "Проектные и исследовательские задания",
                    "reference": "Справочные и наглядные материалы",
                    "mixed": "Смешанный тип",
                    "other": "Другой тип",
                    "cannot_determine": "Не удалось определить",
                },
            },
            "class_type": {
                "type": "choice",
                "instructions": "Определи форму занятия, для которой предназначен материал.",
                "criteria": {
                    "lesson": "Урок", "lecture": "Лекция", "seminar": "Семинар",
                    "practical": "Практическое занятие", "lab": "Лабораторное занятие",
                    "control": "Контрольное занятие", "project": "Проектное занятие",
                    "independent": "Самостоятельная работа", "elective": "Электив/факультатив",
                    "coursework": "Курсовая работа", "practice": "Учебная/производственная практика",
                    "unknown": "Не определено",
                },
            },
            "topic": {
                "type": "text",
                "instructions": "Определи тему учебного материала коротким словосочетанием (3-6 слов). Если тема неясна, ничего не пиши.",
            },
            "complexity": {
                "type": "score",
                "instructions": (
                    "Оцени сложность учебного материала по содержанию, а не по объёму. "
                    "Учитывай сложность понятий, терминологию, необходимость анализа и самостоятельности."
                ),
                "criteria": self._complexity_levels(),
            },
        }
        for i, req in enumerate(cat.requirements, 1):
            questions[f"compliance_{i}"] = {
                "type": "noul",
                "instructions": f"Соответствует ли материал методическому требованию: «{req}»?",
            }
        return questions

    def _complexity_levels(self) -> list[str]:
        return list(self.catalogs.complexity_levels)

    # ------------------------------------------------------------------ #
    def classify(self, text: str, title: str) -> dict:
        model = self._ensure()
        import torch

        state = (
            "МЕТОДИЧЕСКИЕ ТРЕБОВАНИЯ:\n"
            f"{criteria_text(self.catalogs)}\n\n"
            f"НАЗВАНИЕ УЧЕБНОГО МАТЕРИАЛА:\n{title or ''}\n\n"
            f"УЧЕБНЫЙ МАТЕРИАЛ:\n{text[:STATE_MAX_CHARS]}"
        )
        questions = self.build_questions()
        with torch.inference_mode():
            out = model.system_one(state, questions)
        return self._parse(out["answers"])

    def _parse(self, a: dict) -> dict:
        cat = self.catalogs

        subject = a["subject"]["choice"]
        category = "Междисциплинарные"
        subject_is_auto = False
        for c in cat.subjects:
            if subject in c["subjects"]:
                category = c["name"]
                break
        else:
            if subject != "cannot_determine":
                subject_is_auto = True

        level_idx = int(a["complexity"]["score"])
        levels = self._complexity_levels()
        complexity_level = levels[level_idx] if 0 <= level_idx < len(levels) else "unknown"
        complexity_score = level_idx + 1 if 0 <= level_idx < len(levels) else 0.0

        compliance = []
        for i, req in enumerate(cat.requirements, 1):
            p = float(a[f"compliance_{i}"]["noul"])
            compliance.append({"requirement": req, "score": round(p, 3), "is_compliant": p >= 0.5})

        topic = (a["topic"].get("text") or "").strip() if a["topic"].get("found") else ""

        return {
            "subject": subject,
            "subject_confidence": round(float(a["subject"]["confidence"]), 3),
            "subject_is_auto": subject_is_auto,
            "category": category,
            "education_level": a["education_level"]["choice"],
            "language": a["language"]["choice"],
            "material_type": a["material_type"]["choice"],
            "class_type": a["class_type"]["choice"],
            "topic": topic,
            "complexity_score": float(complexity_score),
            "complexity_level": complexity_level,
            "compliance": compliance,
        }
