"""Базовый адаптер модели: ленивая загрузка, офлайн-флаги, единый интерфейс."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def set_offline_flags() -> None:
    """Офлайн-режим HuggingFace: не обращаемся в сеть за весами."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class ModelAdapter:
    """Общий контракт для моделей из `models/`."""

    name: str = "base"
    model_rel_path: str = ""

    def __init__(self, models_dir: Path | str | None = None) -> None:
        self.models_dir = Path(models_dir) if models_dir else _default_models_dir()
        self._loaded: Any = None
        self._failed: Exception | None = None

    @property
    def model_path(self) -> Path:
        return self.models_dir / self.model_rel_path

    def is_available(self) -> bool:
        return self.model_path.is_dir()

    def is_loaded(self) -> bool:
        return self._loaded is not None

    def load(self) -> None:
        raise NotImplementedError

    def ensure_loaded(self) -> Any:
        if self._loaded is None:
            if self._failed is not None:
                raise self._failed
            if not self.is_available():
                raise FileNotFoundError(f"Модель не найдена: {self.model_path}")
            try:
                set_offline_flags()
                self._loaded = self.load()
            except Exception as exc:  # noqa: BLE001
                self._failed = exc
                raise
        return self._loaded

    def describe_missing(self) -> str:
        return f"папка модели {self.model_path.name} не найдена в {self.models_dir}"


def _default_models_dir() -> Path:
    from prof2027.core.paths import project_root

    return project_root() / "models"
