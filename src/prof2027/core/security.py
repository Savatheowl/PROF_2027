"""Хеширование паролей (SHA-256 + соль на уровне приложения).

Для демонстрационного проекта достаточно SHA-256; при необходимости
заменяется на bcrypt/argon2 без изменения интерфейса.
"""

from __future__ import annotations

import hashlib

_SALT = "PROF2027::"


def hash_password(password: str) -> str:
    return hashlib.sha256((_SALT + password).encode("utf-8")).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    return hash_password(password) == password_hash
