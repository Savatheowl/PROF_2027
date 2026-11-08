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


def hint(source: dict, key: str, default: str = "") -> str:
    value = source.get(key)
    return default if value is None else str(value)
