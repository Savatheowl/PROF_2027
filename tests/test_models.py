"""Тесты БД: инициализация, 8 таблиц, сиды, ограничения уникальности."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError

from prof2027.core.security import verify_password
from prof2027.db.base import SessionManager, make_engine
from prof2027.db.init_db import initialize, seed_status
from prof2027.db.models import (
    Material,
    MethodologicalRequirement,
    Subject,
    User,
)

EXPECTED_TABLES = {
    "subjects",
    "materials",
    "methodological_requirements",
    "material_compliance",
    "cluster_runs",
    "model_versions",
    "users",
    "trajectories",
}


@pytest.fixture()
def initialized(cfg: dict, proj_root: Path):
    return initialize(cfg)


def test_initialize_creates_eight_tables(initialized, db_path: Path) -> None:
    assert db_path.is_file()
    engine = make_engine(db_path)
    tables = set(inspect(engine).get_table_names())
    assert EXPECTED_TABLES <= tables


def test_seeds(initialized, db_path: Path) -> None:
    manager = SessionManager(make_engine(db_path))
    with manager.session() as session:
        counts = seed_status(session)
        assert counts["subjects"] >= 25
        assert counts["requirements"] == 24
        assert counts["users"] == 3
        assert counts["materials"] == 0


def test_default_credentials(initialized, db_path: Path) -> None:
    manager = SessionManager(make_engine(db_path))
    with manager.session() as session:
        admin = session.scalar(select(User).where(User.username == "admin"))
        assert admin is not None
        assert admin.role == "admin"
        assert verify_password("admin", admin.password_hash)
        assert not verify_password("wrong", admin.password_hash)


def test_subject_unique_constraint(initialized, db_path: Path) -> None:
    manager = SessionManager(make_engine(db_path))
    with manager.session() as session:
        session.add(Subject(category="Cat", name="Subj"))
        session.add(Subject(category="Cat", name="Subj"))
        with pytest.raises(IntegrityError):
            session.commit()


def test_requirement_unique_text(initialized, db_path: Path) -> None:
    manager = SessionManager(make_engine(db_path))
    with manager.session() as session:
        session.add(MethodologicalRequirement(category="C", text="Уникальный текст требования"))
        session.add(MethodologicalRequirement(category="C", text="Уникальный текст требования"))
        with pytest.raises(IntegrityError):
            session.commit()


def test_material_hash_unique(initialized, db_path: Path) -> None:
    manager = SessionManager(make_engine(db_path))
    with manager.session() as session:
        session.add(Material(source_hash="abc", title="m1"))
        session.add(Material(source_hash="abc", title="m2"))
        with pytest.raises(IntegrityError):
            session.commit()


def test_init_is_idempotent(cfg: dict, proj_root: Path, db_path: Path) -> None:
    initialize(cfg)
    initialize(cfg)  # повторный запуск не должен падать
    manager = SessionManager(make_engine(db_path))
    with manager.session() as session:
        assert session.scalar(select(Subject.id).limit(1)) is not None
        assert seed_status(session)["requirements"] == 24  # без дублей
