"""Базовый класс AI-адаптеров: ленивая загрузка моделей и проверка доступности."""

from __future__ import annotations

from pathlib import Path


class Adapter:
    def __init__(self, model_dir: Path | str | None = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else Path(".")
        self._obj = None

    def is_available(self) -> bool:
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def _ensure(self):
        if self._obj is None:
            if not self.is_available():
                raise RuntimeError(f"{self.__class__.__name__}: зависимости не установлены")
            self.load()
        return self._obj
