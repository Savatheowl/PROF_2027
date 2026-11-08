"""Тесты CLI."""

import pytest

from prof2027.cli import main


def test_init_prints_credentials(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PROF2027_DB_PATH", str(tmp_path / "cli.db"))
    rc = main(["init"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "PROF_2027 инициализирован" in out
    assert "admin" in out and "analyst" in out and "viewer" in out
    assert (tmp_path / "cli.db").exists()


def test_config_show(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PROF2027_DB_PATH", str(tmp_path / "x.db"))
    rc = main(["config", "show"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "db_path" in out


def test_users_show(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PROF2027_DB_PATH", str(tmp_path / "x.db"))
    rc = main(["users", "show"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "admin" in out


def test_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
