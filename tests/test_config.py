"""Тесты конфигурации."""

from prof2027.config import Config


def test_basics(cfg: Config):
    assert cfg.project_name == "PROF_2027"
    assert cfg.db_path.name == "test.db"


def test_allowed_extensions(cfg: Config):
    assert ".pdf" in cfg.allowed_extensions
    assert ".docx" in cfg.allowed_extensions
    assert ".mp4" in cfg.allowed_extensions
    assert ".exe" not in cfg.allowed_extensions
    assert ".zip" not in cfg.allowed_extensions


def test_moderation_thresholds(cfg: Config):
    assert cfg.moderation["approved_threshold"] == 0.7
    assert cfg.moderation["review_threshold"] == 0.4


def test_complexity_levels(cfg: Config):
    assert len(cfg.complexity_levels) == 5


def test_user_defaults(cfg: Config):
    u = cfg.user_defaults()
    assert u["admin"] == ("admin", "admin", "admin")
    assert u["analyst"][:2] == ("analyst", "analyst")
    assert u["viewer"][2] == "viewer"
