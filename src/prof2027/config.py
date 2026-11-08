"""Конфигурация PROF_2027.

Читает config/config.yaml и .env, разрешает пути относительно корня проекта.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Config:
    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root else PROJECT_ROOT
        load_dotenv(self.root / ".env")

        cfg = yaml.safe_load((self.root / "config" / "config.yaml").read_text(encoding="utf-8")) or {}
        self.project_name = cfg["project"]["name"]

        db_env = os.getenv("PROF2027_DB_PATH")
        self.db_path = (self.root / db_env) if db_env else self.root / cfg["project"]["db_path"]

        self.catalogs = {key: self.root / value for key, value in cfg["catalogs"].items()}
        self.ingestion = cfg["ingestion"]
        self.moderation = cfg["moderation"]
        self.complexity_levels = cfg["complexity"]["levels"]
        self.env = os.getenv("PROF2027_ENV", "prod")

    @property
    def allowed_extensions(self) -> set[str]:
        return set(self.ingestion["allowed_extensions"])

    def user_defaults(self) -> dict[str, tuple[str, str, str]]:
        """Логин / пароль / роль по умолчанию (можно переопределить в .env)."""
        return {
            "admin": (
                os.getenv("PROF2027_ADMIN_USER", "admin"),
                os.getenv("PROF2027_ADMIN_PASSWORD", "admin"),
                "admin",
            ),
            "analyst": (
                os.getenv("PROF2027_ANALYST_USER", "analyst"),
                os.getenv("PROF2027_ANALYST_PASSWORD", "analyst"),
                "analyst",
            ),
            "viewer": (
                os.getenv("PROF2027_VIEWER_USER", "viewer"),
                os.getenv("PROF2027_VIEWER_PASSWORD", "viewer"),
                "viewer",
            ),
        }
