"""Подключение к SQLite: engine, фабрика сессий, вспомогательные функции."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


def make_engine(db_path: str | Path, *, echo: bool = False) -> Engine:
    """Создаёт engine для SQLite по пути из конфига."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(
        f"sqlite:///{path.as_posix()}",
        echo=echo,
        connect_args={"check_same_thread": False},
    )


class SessionManager:
    """Обёртка engine + sessionmaker, создаётся один раз при старте."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._factory = sessionmaker(bind=engine, expire_on_commit=False)

    @classmethod
    def from_config(cls, cfg: dict) -> "SessionManager":
        db_path = cfg["db"]["path"]
        return cls(make_engine(db_path))

    def session(self) -> Session:
        return self._factory()

    @contextmanager
    def transaction(self) -> Iterator[Session]:
        session = self._factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
