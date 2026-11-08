"""Manifest загрузки: список источников (URL или локальные пути) с подсказками."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_manifest(path: Path | str) -> list[dict]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    sources = data.get("sources") or []
    if not isinstance(sources, list):
        raise ValueError("manifest: 'sources' должен быть списком")
    return [dict(s) for s in sources]


def sources_from_dir(path: Path | str, allowed_extensions: set[str]) -> list[dict]:
    """Собирает список источников из папки (рекурсивно) по разрешённым расширениям.

    Пропускает подпапку 'cached' (скачанные копии) и не-файлы.
    Возвращает абсолютные пути — загрузчик их нормализует через cfg.root.
    """
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"не папка: {root}")
    allowed = {e.lower() for e in allowed_extensions}
    sources: list[dict] = []
    for fp in sorted(root.rglob("*")):
        if not fp.is_file():
            continue
        if "cached" in fp.parts:
            continue
        if fp.suffix.lower() in allowed:
            sources.append({"path": str(fp.resolve())})
    return sources


def hint(source: dict, key: str, default: str = "") -> str:
    value = source.get(key)
    return default if value is None else str(value)
