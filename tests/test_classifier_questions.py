"""Тесты классификатора d1-2B: построение вопросов и разбор ответов."""

from prof2027.ai.classifier import Classifier


def _fake_answers(cat):
    answers = {
        "subject": {"type": "choice", "choice": "Алгебра", "confidence": 0.9, "probabilities": {}},
        "education_level": {"type": "choice", "choice": "school", "confidence": 0.9, "probabilities": {}},
        "language": {"type": "choice", "choice": "ru", "confidence": 0.9, "probabilities": {}},
        "material_type": {"type": "choice", "choice": "student_material", "confidence": 0.9, "probabilities": {}},
        "class_type": {"type": "choice", "choice": "lesson", "confidence": 0.9, "probabilities": {}},
        "topic": {"type": "text", "text": "Обыкновенные дроби", "found": True},
        "complexity": {"type": "score", "score": 1, "confidence": 0.7, "probabilities": {}, "legend": {}},
    }
    for i in range(1, len(cat.requirements) + 1):
        answers[f"compliance_{i}"] = {"type": "noul", "noul": 0.8 if i % 2 else 0.2}
    return answers


def test_build_questions(cat):
    clf = Classifier(model_dir=None, catalogs=cat)
    q = clf.build_questions()
    assert q["subject"]["type"] == "choice"
    assert "cannot_determine" in q["subject"]["criteria"]
    assert len(q["subject"]["criteria"]) == len(cat.subject_names) + 1
    assert set(q["education_level"]["criteria"]) == {"school", "spo", "university", "unknown"}
    assert set(q["language"]["criteria"]) == {"ru", "en", "other"}
    assert q["topic"]["type"] == "text"
    assert len(q["complexity"]["criteria"]) == 5
    compliance = [k for k in q if k.startswith("compliance_")]
    assert len(compliance) == 24


def test_parse_answers(cat):
    clf = Classifier(model_dir=None, catalogs=cat)
    parsed = clf._parse(_fake_answers(cat))
    assert parsed["subject"] == "Алгебра"
    assert parsed["category"] == "Математика и статистика"
    assert parsed["subject_is_auto"] is False
    assert parsed["education_level"] == "school"
    assert parsed["language"] == "ru"
    assert parsed["topic"] == "Обыкновенные дроби"
    assert parsed["complexity_score"] == 2.0
    assert parsed["complexity_level"] == "низкая"
    assert len(parsed["compliance"]) == 24
    assert sum(1 for c in parsed["compliance"] if c["is_compliant"]) == 12


def test_auto_subject(cat):
    clf = Classifier(model_dir=None, catalogs=cat)
    answers = _fake_answers(cat)
    answers["subject"] = {"type": "choice", "choice": "Новая дисциплина", "confidence": 0.9, "probabilities": {}}
    parsed = clf._parse(answers)
    assert parsed["subject_is_auto"] is True
    assert parsed["category"] == "Междисциплинарные"


def test_unknown_complexity(cat):
    clf = Classifier(model_dir=None, catalogs=cat)
    answers = _fake_answers(cat)
    answers["complexity"] = {"type": "score", "score": 99, "confidence": 0.5, "probabilities": {}, "legend": {}}
    parsed = clf._parse(answers)
    assert parsed["complexity_level"] == "unknown"
    assert parsed["complexity_score"] == 0.0
