"""Общие фикстуры тестов."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """Конфиг с БД во временной папке (конфиг-файлы — из репозитория)."""
    monkeypatch.setenv("PROF2027_DB_PATH", str(tmp_path / "test.db"))
    from prof2027.config import Config

    return Config()


@pytest.fixture
def cat(cfg):
    from prof2027.catalogs import Catalogs

    return Catalogs(cfg).load()
