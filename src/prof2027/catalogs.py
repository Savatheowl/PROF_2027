"""Каталоги предметной области: предметы, требования, типы, языки.

Данные живут в YAML; замена файла = новая конфигурация без правки кода.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from prof2027.config import Config


class Catalogs:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._data: dict[str, dict] = {}

    def load(self) -> "Catalogs":
        for key, path in self.cfg.catalogs.items():
            self._data[key] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return self

    # --- subjects.yaml ---
    @property
    def subjects(self) -> list[dict]:
        return self._data["subjects"].get("categories", [])

    @property
    def categories(self) -> list[str]:
        return [c["name"] for c in self.subjects]

    @property
    def subject_names(self) -> list[str]:
        return [s for c in self.subjects for s in c["subjects"]]

    def subjects_of(self, category: str) -> list[str]:
        for c in self.subjects:
            if c["name"] == category:
                return c["subjects"]
        return []

    # --- methodology.yaml ---
    @property
    def methodology(self) -> list[dict]:
        return self._data["methodology"]

    @property
    def requirements(self) -> list[str]:
        return [r for cat in self.methodology for r in cat["requirements"]]

    # --- material_types / class_types / languages ---
    @property
    def material_types(self) -> list[str]:
        return self._data["material_types"]["material_types"]

    @property
    def class_types(self) -> list[str]:
        return self._data["class_types"]["class_types"]

    @property
    def languages(self) -> list[str]:
        return self._data["languages"]["languages"]
