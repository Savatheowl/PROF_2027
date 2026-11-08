"""Тесты CLI: init, status, requirements reload."""

from __future__ import annotations

from pathlib import Path

from prof2027.cli import main


def test_init_command(proj_root: Path, capsys) -> None:
    code = main(["init"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Инициализация завершена" in out
    assert (proj_root / "data" / "test.db").is_file()
    # учётные записи печатаются при запуске
    assert "admin" in out and "analyst" in out and "viewer" in out


def test_status_command(proj_root: Path, capsys) -> None:
    main(["init"])
    code = main(["status"])
    assert code == 0
    out = capsys.readouterr().out
    assert "requirements" in out
    assert "users" in out


def test_status_without_db(proj_root: Path, capsys) -> None:
    code = main(["status"])
    assert code == 0
    assert "не создана" in capsys.readouterr().out


def test_requirements_reload(proj_root: Path, capsys) -> None:
    main(["init"])
    code = main(["requirements", "reload"])
    assert code == 0
    assert "24" in capsys.readouterr().out


def test_force_init_rebuilds(proj_root: Path) -> None:
    main(["init"])
    main(["init", "--force"])
    assert (proj_root / "data" / "test.db").is_file()
