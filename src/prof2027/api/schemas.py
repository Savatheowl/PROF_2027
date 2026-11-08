"""Схемы запросов/ответов REST API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LoaderRunRequest(BaseModel):
    manifest: str = Field(..., description="путь к manifest.yaml относительно корня проекта")
    db: str | None = Field(None, description="переопределить путь к БД")


class ClusterRunRequest(BaseModel):
    strategy: str = Field("all", description="all / parallel / sequential / complexity")


class TrainRunRequest(BaseModel):
    task: str = Field("all", description="all / complexity / methodology / cluster")


class TimeEstimateRequest(BaseModel):
    word_count: int = Field(400, ge=0, description="объём текста в словах")
    media_count: int = Field(0, ge=0, description="количество медиафайлов")
    complexity_level: str = Field("средняя", description="уровень сложности материала")
    complexity_score: float = Field(3.0, ge=1.0, le=5.0)


class TrajectoryRequest(BaseModel):
    user: str = "demo"
    hours_per_day: float = Field(2.0, gt=0.0)
    deadline_days: int = Field(14, gt=0)
    subject: str | None = None
    category: str | None = None
    level: str | None = None
    gantt: bool = True
