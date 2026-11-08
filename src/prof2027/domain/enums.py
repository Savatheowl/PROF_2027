"""Константы предметной области.

Значения хранятся в БД как TEXT, поэтому используются строковые константы
(а не Enum с числами). Enum — для валидации и подсказок IDE.
"""

from __future__ import annotations

from enum import Enum


class EducationLevel(str, Enum):
    SCHOOL = "school"
    SPO = "spo"
    UNIVERSITY = "university"
    UNKNOWN = "unknown"


class Language(str, Enum):
    RU = "ru"
    EN = "en"
    OTHER = "other"


class MaterialType(str, Enum):
    STUDENT_MATERIAL = "student_material"
    TEACHER_MATERIAL = "teacher_material"
    PRACTICE = "practice"
    ASSESSMENT = "assessment"
    PRACTICAL_LAB = "practical_lab"
    PROJECT = "project"
    REFERENCE = "reference"
    MIXED = "mixed"
    OTHER = "other"
    CANNOT_DETERMINE = "cannot_determine"


class ClassType(str, Enum):
    LESSON = "lesson"
    LECTURE = "lecture"
    SEMINAR = "seminar"
    PRACTICAL = "practical"
    LAB = "lab"
    CONTROL = "control"
    PROJECT = "project"
    INDEPENDENT = "independent"
    ELECTIVE = "elective"
    COURSEWORK = "coursework"
    PRACTICE = "practice"
    UNKNOWN = "unknown"


class ComplexityLevel(str, Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ModerationVerdict(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"


class UserRole(str, Enum):
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"


class ClusterStrategy(str, Enum):
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    COMPLEXITY = "complexity"


class ClusterAlgorithm(str, Enum):
    KMEANS = "kmeans"
    AGGLOMERATIVE = "agglomerative"
    GMM = "gmm"
    DBSCAN = "dbscan"


class ModelTask(str, Enum):
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    COMPLEXITY = "complexity"
