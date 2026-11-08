"""Конфигурация системы.

Источники в порядке приоритета:
1. переменные окружения;
2. `.env` в корне проекта;
3. `config/config.yaml` (с интерполяцией ${VAR:default}).

Конфиг доступен как словарь через `load_config()`, а типизированные
значения — через `AppConfig.load()` (Pydantic).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")


def _interpolate(value: Any) -> Any:
    if isinstance(value, str):

        def repl(m: re.Match) -> str:
            name, default = m.group(1), m.group(2)
            return os.environ.get(name, default if default is not None else "")

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


def _load_env(root: Path) -> None:
    load_dotenv(root / ".env", override=False)


def load_config(root: Path | None = None) -> dict:
    """Загружает и интерполирует config/config.yaml."""
    from prof2027.core.paths import config_dir  # локальный импорт против цикла

    root = root or config_dir().parent
    _load_env(root)
    path = config_dir(root) / "config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Конфиг не найден: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError("config.yaml должен содержать словарь верхнего уровня")
    return _interpolate(raw)


def get(root: Path | None = None, *keys: str, default: Any = None) -> Any:
    """Достаёт значение из конфига по цепочке ключей."""
    cfg = load_config(root)
    node: Any = cfg
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node
