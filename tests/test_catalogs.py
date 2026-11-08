"""Тесты каталогов YAML."""

from prof2027.catalogs import Catalogs


def test_subjects(cat: Catalogs):
    assert len(cat.categories) >= 9
    assert "Программирование и компьютеры" in cat.categories
    assert "Разработка искусственного интеллекта" in cat.subject_names
    assert "Информационные системы" in cat.subjects_of("Программирование и компьютеры")
    assert len(cat.subject_names) >= 30


def test_methodology(cat: Catalogs):
    assert len(cat.methodology) == 8
    assert len(cat.requirements) == 24


def test_lists(cat: Catalogs):
    assert cat.languages == ["ru", "en", "other"]
    assert len(cat.material_types) >= 5
    assert "student_material" in cat.material_types
    assert len(cat.class_types) >= 5
    assert "lecture" in cat.class_types
