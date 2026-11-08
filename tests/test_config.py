"""Тесты конфигурации: загрузка YAML, интерполяция ${VAR:default}, доступ по ключам."""

from __future__ import annotations

import os

from prof2027.core import config as cfg_mod


def test_load_config_basic(cfg: dict) -> None:
    assert cfg["app"]["name"] == "PROF_2027"
    assert cfg["dashboard"]["port"] == 8050
    assert "allowed_extensions" in cfg["ingestion"]


def test_extension_allowlist(cfg: dict) -> None:
    exts = cfg["ingestion"]["allowed_extensions"]
    assert ".pdf" in exts
    assert ".docx" in exts
    assert ".pptx" in exts
    assert ".epub" in exts
    assert ".fb2" in exts
    assert ".mp3" in exts
    assert ".exe" not in exts
    assert ".tmp" not in exts


def test_env_interpolation(proj_root, monkeypatch) -> None:
    monkeypatch.setenv("PROF2027_ADMIN_PASSWORD", "secret")
    cfg = cfg_mod.load_config(proj_root)
    assert cfg["users"]["admin"]["password"] == "secret"


def test_get_default(proj_root) -> None:
    assert cfg_mod.get(proj_root, "nope", "key", default=42) == 42
    assert cfg_mod.get(proj_root, "app", "name") == "PROF_2027"
