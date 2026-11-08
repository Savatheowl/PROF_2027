"""Тесты каталогов: предметы, методические требования, типы, языки."""

from __future__ import annotations

from prof2027.domain import catalogs


def test_subjects_catalog() -> None:
    categories = catalogs.load_subjects()
    assert len(categories) >= 8
    names = {cat.name for cat in categories}
    assert "Программирование и компьютеры" in names
    programming = next(c for c in categories if c.name == "Программирование и компьютеры")
    subject_names = {s.name for s in programming.subjects}
    assert "Разработка искусственного интеллекта" in subject_names
    assert "Информационные системы" in subject_names


def test_methodology_24_requirements() -> None:
    reqs = catalogs.load_methodology()
    assert len(reqs) == 24
    by_category = {}
    for r in reqs:
        by_category[r.category] = by_category.get(r.category, 0) + 1
    assert all(count == 3 for count in by_category.values())
    texts = {r.text for r in reqs}
    assert len(texts) == 24  # уникальны


def test_value_catalogs() -> None:
    material_types = catalogs.load_material_types()
    codes = {v.code for v in material_types}
    assert "student_material" in codes
    assert "cannot_determine" in codes

    class_types = catalogs.load_class_types()
    class_codes = {v.code for v in class_types}
    assert "lesson" in class_codes and "lecture" in class_codes

    languages = catalogs.load_languages()
    lang_codes = {v.code for v in languages}
    assert lang_codes == {"ru", "en", "other"}


def test_methodology_statistics() -> None:
    stats = catalogs.methodology_statistics()
    assert stats["total"] == 24
    assert stats["categories"] == 8
