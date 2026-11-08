"""Безопасность: хеширование паролей и пользователи по умолчанию."""

from __future__ import annotations

import hashlib

from sqlalchemy.orm import Session

from prof2027.config import Config
from prof2027.db import User


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_password(password: str, digest: str) -> bool:
    return hash_password(password) == digest


def create_default_users(session: Session, cfg: Config) -> list[tuple[str, str, str]]:
    """Создаёт пользователей по умолчанию, если их нет. Возвращает созданных."""
    created: list[tuple[str, str, str]] = []
    for username, password, role in cfg.user_defaults().values():
        if session.query(User).filter_by(username=username).first() is None:
            session.add(User(username=username, password_hash=hash_password(password), role=role))
            created.append((username, password, role))
    session.commit()
    return created
