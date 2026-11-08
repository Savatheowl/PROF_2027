"""Общие фикстуры: изолированный корень проекта и БД для тестов."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

REAL_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def proj_root(tmp_path: Path) -> Path:
    """Временный корень проекта: свои config/ и data/, без записи в реальный проект."""
    root = tmp_path / "proj"
    root.mkdir()
    shutil.copytree(REAL_ROOT / "config", root / "config")
    os.environ["PROF2027_ROOT"] = str(root)
    os.environ["PROF2027_DB_PATH"] = str(root / "data" / "test.db")
    os.environ.pop("PROF2027_ENV", None)
    return root


@pytest.fixture()
def cfg(proj_root: Path) -> dict:
    from prof2027.core.config import load_config

    return load_config(proj_root)


@pytest.fixture()
def db_path(cfg: dict) -> Path:
    from prof2027.core.paths import project_root

    return project_root() / cfg["db"]["path"]
