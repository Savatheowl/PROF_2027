"""Пути проекта и создание служебных папок."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_MARKERS = ("pyproject.toml",)


def project_root() -> Path:
    """Корень проекта PROF_2027.

    Приоритет: переменная окружения PROF2027_ROOT, затем поиск вверх
    от каталога пакета до папки, содержащей pyproject.toml и config/.
    """
    env_root = os.environ.get("PROF2027_ROOT")
    if env_root:
        return Path(env_root).resolve()

    current = Path(__file__).resolve()
    for parent in current.parents:
        has_marker = any((parent / marker).is_file() for marker in PROJECT_MARKERS)
        if has_marker and (parent / "config").is_dir():
            return parent
    # Fallback: src/prof2027/core -> PROF_2027
    return current.parents[3]


def config_dir(root: Path | None = None) -> Path:
    root = root or project_root()
    return root / "config"


def ensure_dirs(root: Path | None = None) -> dict[str, Path]:
    """Создаёт служебные папки и возвращает их пути."""
    root = root or project_root()
    cfg = load_config_dict(root)
    paths: dict[str, Path] = {}
    for key in (
        "data_dir",
        "raw_dir",
        "processed_dir",
        "incoming_dir",
        "artifacts_dir",
        "quarantine_dir",
        "models_dir",
        "reports_dir",
        "logs_dir",
    ):
        rel = cfg["paths"][key]
        p = root / rel
        p.mkdir(parents=True, exist_ok=True)
        paths[key] = p
    return paths


def load_config_dict(root: Path | None = None) -> dict:
    """Загружает config/config.yaml с интерполяцией ${ENV:default}."""
    from prof2027.core.config import load_config  # локальный импорт против цикла

    root = root or project_root()
    return load_config(root)
