"""SQLAlchemy-модели: 8 таблиц (архитектура v0.2, раздел 6)."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from prof2027.domain.enums import (
    EducationLevel,
    Language,
    MaterialType,
    ModerationVerdict,
    UserRole,
)


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# 1. subjects
# ---------------------------------------------------------------------------

class Subject(Base):
    __tablename__ = "subjects"
    __table_args__ = (UniqueConstraint("category", "name", name="uq_subject_category_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    category: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    aliases: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    source: Mapped[str] = mapped_column(String(20), nullable=False, default="seed")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Subject {self.category} / {self.name}>"


# ---------------------------------------------------------------------------
# 2. materials
# ---------------------------------------------------------------------------

class Material(Base):
    __tablename__ = "materials"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_url: Mapped[str | None] = mapped_column(Text)
    source_hash: Mapped[str | None] = mapped_column(String(64), unique=True, index=True)
    file_path: Mapped[str | None] = mapped_column(Text)
    file_type: Mapped[str | None] = mapped_column(String(20))

    title: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    category: Mapped[str | None] = mapped_column(String(120))
    subject: Mapped[str | None] = mapped_column(String(200))
    topic: Mapped[str | None] = mapped_column(String(300))
    education_level: Mapped[str] = mapped_column(String(20), nullable=False, default=EducationLevel.UNKNOWN.value)
    language: Mapped[str] = mapped_column(String(10), nullable=False, default=Language.OTHER.value)
    material_type: Mapped[str] = mapped_column(String(30), nullable=False, default=MaterialType.CANNOT_DETERMINE.value)
    class_type: Mapped[str] = mapped_column(String(30), nullable=False, default="unknown")

    text_content: Mapped[str | None] = mapped_column(Text)
    annotation: Mapped[str | None] = mapped_column(Text)
    media_descriptions: Mapped[list | None] = mapped_column(JSON)

    word_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sentence_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    media_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    complexity_score: Mapped[float | None] = mapped_column(Float)
    complexity_level: Mapped[str | None] = mapped_column(String(20))
    methodology_score: Mapped[float | None] = mapped_column(Float)
    moderation_verdict: Mapped[str] = mapped_column(String(20), nullable=False, default=ModerationVerdict.PENDING.value)

    is_generated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    generation_parent_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("materials.id"))

    cluster_parallel: Mapped[int | None] = mapped_column(Integer)
    cluster_sequential: Mapped[int | None] = mapped_column(Integer)

    has_previous: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    has_next: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    previous_material_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("materials.id"))
    next_material_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("materials.id"))

    estimated_time_hours: Mapped[float | None] = mapped_column(Float)
    features_json: Mapped[dict | None] = mapped_column(JSON)

    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow, onupdate=_utcnow)

    __table_args__ = (
        Index("ix_materials_subject", "subject"),
        Index("ix_materials_category", "category"),
        Index("ix_materials_level", "education_level"),
    )


# ---------------------------------------------------------------------------
# 3. methodological_requirements
# ---------------------------------------------------------------------------

class MethodologicalRequirement(Base):
    __tablename__ = "methodological_requirements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    category: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    question: Mapped[str | None] = mapped_column(Text)
    weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


# ---------------------------------------------------------------------------
# 4. material_compliance
# ---------------------------------------------------------------------------

class MaterialCompliance(Base):
    __tablename__ = "material_compliance"
    __table_args__ = (
        UniqueConstraint("material_id", "requirement_id", name="uq_compliance_material_requirement"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    material_id: Mapped[int] = mapped_column(Integer, ForeignKey("materials.id"), nullable=False, index=True)
    requirement_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("methodological_requirements.id"), nullable=False, index=True
    )
    score: Mapped[float | None] = mapped_column(Float)
    is_compliant: Mapped[bool | None] = mapped_column(Boolean)
    details: Mapped[str | None] = mapped_column(Text)


# ---------------------------------------------------------------------------
# 5. cluster_runs
# ---------------------------------------------------------------------------

class ClusterRun(Base):
    __tablename__ = "cluster_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    strategy: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    algorithm: Mapped[str] = mapped_column(String(30), nullable=False)
    params_json: Mapped[dict | None] = mapped_column(JSON)
    metrics_json: Mapped[dict | None] = mapped_column(JSON)
    n_clusters: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# 6. model_versions
# ---------------------------------------------------------------------------

class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("task", "version", name="uq_model_task_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    task: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    algorithm: Mapped[str] = mapped_column(String(50), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    file_path: Mapped[str | None] = mapped_column(Text)
    feature_schema_json: Mapped[list | None] = mapped_column(JSON)
    metrics_json: Mapped[dict | None] = mapped_column(JSON)
    drift_json: Mapped[dict | None] = mapped_column(JSON)
    data_hash: Mapped[str | None] = mapped_column(String(64))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# 7. users
# ---------------------------------------------------------------------------

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default=UserRole.VIEWER.value)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# 8. trajectories
# ---------------------------------------------------------------------------

class Trajectory(Base):
    __tablename__ = "trajectories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_ref: Mapped[str | None] = mapped_column(String(200))
    input_json: Mapped[dict | None] = mapped_column(JSON)
    output_json: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)
