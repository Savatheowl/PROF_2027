"""Загрузка каталогов из YAML: предметы, требования, типы, языки."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from prof2027.core.paths import config_dir


def _read_yaml(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Каталог не найден: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Категории и дисциплины
# ---------------------------------------------------------------------------

@dataclass
class Subject:
    name: str
    aliases: list[str] = field(default_factory=list)


@dataclass
class Category:
    name: str
    subjects: list[Subject]


def load_subjects(root: Path | None = None) -> list[Category]:
    """Читает config/subjects.yaml -> список категорий с дисциплинами."""
    path = config_dir(root) / "subjects.yaml"
    data = _read_yaml(path)
    categories: list[Category] = []
    for cat in data.get("categories", []):
        subjects = []
        for subj in cat.get("subjects", []):
            if isinstance(subj, str):
                subjects.append(Subject(name=subj))
            elif isinstance(subj, dict):
                subjects.append(Subject(name=subj["name"], aliases=list(subj.get("aliases", []))))
        categories.append(Category(name=cat["name"], subjects=subjects))
    return categories


# ---------------------------------------------------------------------------
# Методические требования
# ---------------------------------------------------------------------------

@dataclass
class Requirement:
    category: str
    text: str
    weight: float = 1.0


def load_methodology(root: Path | None = None) -> list[Requirement]:
    """Читает config/methodology.yaml -> плоский список требований."""
    path = config_dir(root) / "methodology.yaml"
    data = _read_yaml(path)
    requirements: list[Requirement] = []
    for block in data:
        category = block["category"]
        for text in block.get("requirements", []):
            requirements.append(Requirement(category=category, text=text))
    return requirements


def methodology_statistics(root: Path | None = None) -> dict:
    """Краткая статистика методических требований."""
    reqs = load_methodology(root)
    by_category: dict[str, int] = {}
    for r in reqs:
        by_category[r.category] = by_category.get(r.category, 0) + 1
    return {"total": len(reqs), "categories": len(by_category), "by_category": by_category}


# ---------------------------------------------------------------------------
# Значения-каталоги (типы, занятия, языки)
# ---------------------------------------------------------------------------

@dataclass
class CatalogValue:
    code: str
    name: str


def _load_value_catalog(root: Path | None, filename: str) -> list[CatalogValue]:
    path = config_dir(root) / filename
    data = _read_yaml(path)
    values = data.get("values", [])
    return [CatalogValue(code=v["code"], name=v["name"]) for v in values]


def load_material_types(root: Path | None = None) -> list[CatalogValue]:
    return _load_value_catalog(root, "material_types.yaml")


def load_class_types(root: Path | None = None) -> list[CatalogValue]:
    return _load_value_catalog(root, "class_types.yaml")


def load_languages(root: Path | None = None) -> list[CatalogValue]:
    return _load_value_catalog(root, "languages.yaml")


def all_catalogs(root: Path | None = None) -> dict:
    """Все каталоги одним словарём (для отчётов и дашборда)."""
    return {
        "categories": load_subjects(root),
        "material_types": load_material_types(root),
        "class_types": load_class_types(root),
        "languages": load_languages(root),
        "methodology": load_methodology(root),
    }
