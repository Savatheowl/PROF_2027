"""Тесты Telegram-бота: тексты ответов, траектория, сборка без токена."""

import os

import pytest

from prof2027.bot import handlers
from prof2027.db import (
    Material,
    MaterialCompliance,
    MethodologicalRequirement,
    get_engine,
    init_db,
    make_session,
    seed_catalogs,
)


def _seed(tmp_path, cfg, cat):
    engine = get_engine(tmp_path / "bot.db")
    init_db(engine)
    seed_catalogs(engine, cat)
    Session = make_session(engine)
    with Session() as s:
        reqs = {r.id: r for r in s.query(MethodologicalRequirement).all()}
        m = Material(title="Дроби", source_hash="bot_h1", category="Математика и статистика",
                     subject="Алгебра", education_level="school", language="ru",
                     material_type="student_material", text_content="Текст. " * 30,
                     word_count=300, complexity_score=2.0, complexity_level="низкая",
                     methodology_score=0.8, moderation_verdict="approved",
                     estimated_time_hours=1.0)
        s.add(m)
        s.flush()
        for rid in reqs:
            s.add(MaterialCompliance(material_id=m.id, requirement_id=rid, score=0.8,
                                     is_compliant=True))
        s.commit()
    return engine


@pytest.fixture
def bot_cfg(tmp_path, cfg, cat, monkeypatch):
    monkeypatch.setenv("PROF2027_DB_PATH", str(tmp_path / "bot.db"))
    from prof2027.config import Config

    _seed(tmp_path, cfg, cat)
    return Config()


def test_start_and_help_text():
    assert "/trajectory" in handlers.build_help_text()


def test_start_text_mentions_commands():
    t = handlers.build_start_text(None)
    assert "Привет" in t
    assert "/status" in t


def test_status_text(bot_cfg):
    t = handlers.build_status_text(bot_cfg)
    assert "Материалов: 1" in t
    assert "Алгебра" in t or "Предметов: 1" in t


def test_materials_text(bot_cfg):
    t = handlers.build_materials_text(bot_cfg)
    assert "#1" in t
    assert "Дроби" in t


def test_subjects_text(bot_cfg):
    t = handlers.build_subjects_text(bot_cfg)
    assert "Алгебра: 1" in t


def test_trajectory_text(bot_cfg):
    from prof2027.catalogs import Catalogs

    cat = Catalogs(bot_cfg).load()
    text, result = handlers.build_trajectory_text(bot_cfg, cat, 2.0, 10, with_gantt=True)
    assert result["total_days"] >= 1
    assert "День 1" in text
    assert result.get("gantt") and os.path.exists(result["gantt"])


def test_loader_result_text():
    summary = {"stats": {"ok": 1, "error": 0},
               "report": "reports/loader_report_x.md",
               "results": [{"source": "a.pdf", "status": "ok", "reason": "id=1"}]}
    t = handlers.build_loader_result_text(summary)
    assert "принято 1" in t
    assert "reports/loader_report_x.md" in t


def test_make_bot_empty_token(bot_cfg, monkeypatch):
    monkeypatch.setenv("PROF2027_TELEGRAM_TOKEN", "")
    from prof2027.bot.app import make_bot

    assert make_bot(bot_cfg) is None


def test_make_bot_with_token(bot_cfg, monkeypatch):
    monkeypatch.setenv("PROF2027_TELEGRAM_TOKEN", "123:fake-token")
    from prof2027.bot.app import make_bot

    built = make_bot(bot_cfg)
    assert built is not None
    bot, dp = built
    assert bot is not None and dp is not None


def test_cli_bot_empty_token_runs(tmp_path, bot_cfg, monkeypatch):
    """CLI 'bot' с пустым токеном не падает, выводит подсказку."""
    import sys
    from prof2027.cli import main

    monkeypatch.setenv("PROF2027_TELEGRAM_TOKEN", "")
    rc = main(["bot", "--db", str(tmp_path / "bot.db")])
    assert rc == 0
