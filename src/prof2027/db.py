"""База данных: 8 таблиц, engine/session, инициализация и начальное наполнение."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.engine import Engine, create_engine


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# 1. subjects — категории и дисциплины
# ---------------------------------------------------------------------------
class Subject(Base):
    __tablename__ = "subjects"
    __table_args__ = (UniqueConstraint("category", "name", name="uq_subject"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    category: Mapped[str] = mapped_column(String(120), index=True)
    name: Mapped[str] = mapped_column(String(200))
    aliases: Mapped[str] = mapped_column(Text, default="[]")
    source: Mapped[str] = mapped_column(String(20), default="seed")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


# ---------------------------------------------------------------------------
# 2. materials — материалы и их атрибуты (медиа — JSON в media_descriptions)
# ---------------------------------------------------------------------------
class Material(Base):
    __tablename__ = "materials"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_url: Mapped[str] = mapped_column(String(2000), nullable=True)
    source_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    file_path: Mapped[str] = mapped_column(String(500), nullable=True)
    file_type: Mapped[str] = mapped_column(String(20), nullable=True)
    title: Mapped[str] = mapped_column(String(300))
    category: Mapped[str] = mapped_column(String(120), default="")
    subject: Mapped[str] = mapped_column(String(200), default="", index=True)
    topic: Mapped[str] = mapped_column(String(200), default="")
    education_level: Mapped[str] = mapped_column(String(20), default="unknown")
    language: Mapped[str] = mapped_column(String(10), default="ru")
    material_type: Mapped[str] = mapped_column(String(40), default="cannot_determine")
    class_type: Mapped[str] = mapped_column(String(40), default="unknown")

    text_content: Mapped[str] = mapped_column(Text, default="")
    annotation: Mapped[str] = mapped_column(Text, default="")
    media_descriptions: Mapped[str] = mapped_column(Text, default="[]")  # JSON-массив

    word_count: Mapped[int] = mapped_column(Integer, default=0)
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    sentence_count: Mapped[int] = mapped_column(Integer, default=0)
    media_count: Mapped[int] = mapped_column(Integer, default=0)

    complexity_score: Mapped[float] = mapped_column(Float, default=0.0)
    complexity_level: Mapped[str] = mapped_column(String(20), default="unknown")
    methodology_score: Mapped[float] = mapped_column(Float, default=0.0)
    moderation_verdict: Mapped[str] = mapped_column(String(20), default="pending")

    is_generated: Mapped[int] = mapped_column(Integer, default=0)
    generation_parent_id: Mapped[int] = mapped_column(Integer, nullable=True)

    cluster_parallel: Mapped[int] = mapped_column(Integer, nullable=True)
    cluster_sequential: Mapped[int] = mapped_column(Integer, nullable=True)

    has_previous: Mapped[int] = mapped_column(Integer, default=0)
    has_next: Mapped[int] = mapped_column(Integer, default=0)
    previous_material_id: Mapped[int] = mapped_column(Integer, nullable=True)
    next_material_id: Mapped[int] = mapped_column(Integer, nullable=True)

    estimated_time_hours: Mapped[float] = mapped_column(Float, default=0.0)
    features_json: Mapped[str] = mapped_column(Text, default="{}")

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)


# ---------------------------------------------------------------------------
# 3. methodological_requirements — методические требования
# ---------------------------------------------------------------------------
class MethodologicalRequirement(Base):
    __tablename__ = "methodological_requirements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    category: Mapped[str] = mapped_column(String(120))
    text: Mapped[str] = mapped_column(String(300), unique=True)
    question: Mapped[str] = mapped_column(Text, default="")
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


# ---------------------------------------------------------------------------
# 4. material_compliance — соответствие материалов требованиям
# ---------------------------------------------------------------------------
class MaterialCompliance(Base):
    __tablename__ = "material_compliance"
    __table_args__ = (UniqueConstraint("material_id", "requirement_id", name="uq_compliance"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    material_id: Mapped[int] = mapped_column(Integer, ForeignKey("materials.id"), index=True)
    requirement_id: Mapped[int] = mapped_column(Integer, ForeignKey("methodological_requirements.id"))
    score: Mapped[float] = mapped_column(Float, default=0.0)
    is_compliant: Mapped[bool] = mapped_column(Boolean, default=False)
    details: Mapped[str] = mapped_column(Text, default="")


# ---------------------------------------------------------------------------
# 5. cluster_runs — запуски кластеризации и их качество
# ---------------------------------------------------------------------------
class ClusterRun(Base):
    __tablename__ = "cluster_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    strategy: Mapped[str] = mapped_column(String(30), index=True)  # parallel/sequential/complexity
    algorithm: Mapped[str] = mapped_column(String(30))
    params_json: Mapped[str] = mapped_column(Text, default="{}")
    metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    n_clusters: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


# ---------------------------------------------------------------------------
# 6. model_versions — модели, метрики, дрейф, версии
# ---------------------------------------------------------------------------
class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    task: Mapped[str] = mapped_column(String(30), index=True)  # parallel/sequential/complexity
    algorithm: Mapped[str] = mapped_column(String(50))
    version: Mapped[int] = mapped_column(Integer, default=1)
    file_path: Mapped[str] = mapped_column(String(300), default="")
    feature_schema_json: Mapped[str] = mapped_column(Text, default="[]")
    metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    drift_json: Mapped[str] = mapped_column(Text, default="{}")
    data_hash: Mapped[str] = mapped_column(String(64), default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


# ---------------------------------------------------------------------------
# 7. users — доступы к дашборду
# ---------------------------------------------------------------------------
class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(60), unique=True)
    password_hash: Mapped[str] = mapped_column(String(64))
    role: Mapped[str] = mapped_column(String(20), default="viewer")  # admin/analyst/viewer
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


# ---------------------------------------------------------------------------
# 8. trajectories — вход/выход траекторий
# ---------------------------------------------------------------------------
class Trajectory(Base):
    __tablename__ = "trajectories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_ref: Mapped[str] = mapped_column(String(120), default="")
    input_json: Mapped[str] = mapped_column(Text, default="{}")
    output_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


# ---------------------------------------------------------------------------
# Engine / session / init
# ---------------------------------------------------------------------------
def get_engine(db_path: Path | str) -> Engine:
    return create_engine(f"sqlite:///{db_path}", echo=False)


def make_session(engine: Engine):
    return sessionmaker(bind=engine, expire_on_commit=False)


def init_db(engine: Engine) -> None:
    """Создаёт все таблицы, если их нет."""
    Base.metadata.create_all(engine)


def seed_catalogs(engine: Engine, catalogs) -> None:
    """Наполняет subjects и methodological_requirements из каталогов."""
    Session = make_session(engine)
    with Session() as session:
        for cat in catalogs.subjects:
            for name in cat["subjects"]:
                if not session.query(Subject).filter_by(category=cat["name"], name=name).first():
                    session.add(Subject(category=cat["name"], name=name))
        for cat in catalogs.methodology:
            for text in cat["requirements"]:
                if not session.query(MethodologicalRequirement).filter_by(text=text).first():
                    session.add(MethodologicalRequirement(category=cat["category"], text=text))
        session.commit()
