"""Инициализация системы: папки, БД, каталоги, требования, пользователи."""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from prof2027.core.paths import ensure_dirs, project_root
from prof2027.core.security import hash_password
from prof2027.db.base import SessionManager, make_engine
from prof2027.db.models import (
    Base,
    MethodologicalRequirement,
    Subject,
    User,
)
from prof2027.domain import catalogs as cat
from prof2027.domain.enums import UserRole

logger = logging.getLogger(__name__)

CATALOG_TABLES = (
    "subjects",
    "materials",
    "methodological_requirements",
    "material_compliance",
    "cluster_runs",
    "model_versions",
    "users",
    "trajectories",
)


def create_database(cfg: dict, *, force: bool = False) -> Path:
    """Создаёт SQLite-файл и таблицы. Возвращает путь к БД."""
    db_path = project_root() / cfg["db"]["path"]
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = make_engine(db_path)
    if force:
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    logger.info("База данных: %s (таблиц: %d)", db_path, len(CATALOG_TABLES))
    return db_path


def seed_subjects(session: Session) -> None:
    """Заполняет subjects из config/subjects.yaml. Обновляет существующие."""
    categories = cat.load_subjects()
    for category in categories:
        for subject in category.subjects:
            existing = session.scalar(
                select(Subject).where(
                    Subject.category == category.name,
                    Subject.name == subject.name,
                )
            )
            if existing is None:
                session.add(
                    Subject(
                        category=category.name,
                        name=subject.name,
                        aliases=repr(subject.aliases),
                        source="seed",
                    )
                )
    logger.info("Каталог предметов: %d категорий, %d дисциплин",
                len(categories), sum(len(c.subjects) for c in categories))


def seed_requirements(session: Session) -> None:
    """Заполняет methodological_requirements из config/methodology.yaml."""
    requirements = cat.load_methodology()
    existing_texts = {r.text for r in session.scalars(select(MethodologicalRequirement)).all()}
    added = 0
    for req in requirements:
        if req.text in existing_texts:
            continue
        session.add(
            MethodologicalRequirement(
                category=req.category,
                text=req.text,
                weight=req.weight,
                is_active=True,
            )
        )
        added += 1
    logger.info("Методические требования: %d загружено, %d уже было",
                added, len(requirements) - added)


def seed_users(session: Session, cfg: dict) -> None:
    """Создаёт пользователей дашборда из config.yaml (.env)."""
    users = cfg.get("users", {})
    for key, item in users.items():
        username = item["username"]
        password = item["password"]
        role = item.get("role", UserRole.VIEWER.value)
        existing = session.scalar(select(User).where(User.username == username))
        if existing is None:
            session.add(User(username=username, password_hash=hash_password(password), role=role))
        else:
            existing.password_hash = hash_password(password)
            existing.role = role
    logger.info("Пользователи дашборда: %s", ", ".join(u["username"] for u in users.values()))


def initialize(cfg: dict, *, force: bool = False) -> dict:
    """Полная инициализация: папки -> БД -> каталоги -> пользователи."""
    paths = ensure_dirs()
    db_path = create_database(cfg, force=force)
    manager = SessionManager(make_engine(db_path))
    with manager.transaction() as session:
        seed_subjects(session)
        seed_requirements(session)
        seed_users(session, cfg)
    return {"db_path": str(db_path), "paths": {k: str(v) for k, v in paths.items()}}


def seed_status(session: Session) -> dict:
    """Сводка содержимого БД."""
    from sqlalchemy import func

    from prof2027.db.models import Material

    def count(model) -> int:
        return session.scalar(select(func.count()).select_from(model)) or 0

    return {
        "subjects": count(Subject),
        "requirements": count(MethodologicalRequirement),
        "users": count(User),
        "materials": count(Material),
    }
