"""Сбор данных для отчётов по модулям конкурсного задания.

А. Сбор и обработка — материалы, форматы, вердикты, соответствие требованиям.
Б. Анализ и визуализация — распределения, кластеризация, дашборд, графики.
В. Моделирование — задачи × алгоритмы, метрики, дрейф, время, траектории.
Г. Интеграция — API, бот, роли дашборда, документация.
Д. Демонстрация и развёртывание — структура, CLI, офлайн-сборка.
"""

from __future__ import annotations

import json
from pathlib import Path

from prof2027.db import (
    ClusterRun,
    Material,
    MaterialCompliance,
    MethodologicalRequirement,
    ModelVersion,
    Trajectory,
    get_engine,
    init_db,
    make_session,
)


def _session(cfg):
    engine = get_engine(cfg.db_path)
    init_db(engine)
    return make_session(engine)


def collect(cfg) -> dict:
    """Собирает все данные в один словарь."""
    with _session(cfg)() as s:
        mats = s.query(Material).order_by(Material.id).all()
        reqs = {r.id: r.text for r in s.query(MethodologicalRequirement).all()}
        comp: dict[int, list[float]] = {}
        for row in s.query(MaterialCompliance).all():
            comp.setdefault(row.requirement_id, []).append(row.score)
        clusters = [
            {"id": r.id, "strategy": r.strategy, "algorithm": r.algorithm,
             "n_clusters": r.n_clusters,
             "metrics": json.loads(r.metrics_json or "{}")}
            for r in s.query(ClusterRun).order_by(ClusterRun.id).all()
        ]
        models = [
            {"id": v.id, "task": v.task, "algorithm": v.algorithm, "version": v.version,
             "is_active": v.is_active,
             "metrics": json.loads(v.metrics_json or "{}"),
             "drift": json.loads(v.drift_json or "{}").get("level", "—")}
            for v in s.query(ModelVersion).order_by(ModelVersion.id).all()
        ]
        trajs = [
            {"id": t.id, "user": t.user_ref, **json.loads(t.output_json or "{}")}
            for t in s.query(Trajectory).order_by(Trajectory.id).all()
        ]

    data = {
        "materials": mats,
        "requirements": reqs,
        "compliance": comp,
        "clusters": clusters,
        "models": models,
        "trajectories": trajs,
    }

    # вердикты, типы, предметы, уровни
    data["by_verdict"] = _count(mats, "moderation_verdict")
    data["by_file_type"] = _count(mats, "file_type")
    data["by_subject"] = _count(mats, "subject")
    data["by_level"] = _count(mats, "education_level")
    data["by_language"] = _count(mats, "language")
    data["compliance_means"] = [
        {"requirement": reqs.get(rid, f"#{rid}"),
         "mean": round(sum(scores) / len(scores), 3) if scores else 0.0}
        for rid, scores in sorted(comp.items())
    ]
    data["total_hours"] = round(sum(float(m.estimated_time_hours or 0.0) for m in mats), 2)
    data["methodology_mean"] = round(
        sum(float(m.methodology_score or 0.0) for m in mats) / len(mats), 3) if mats else 0.0
    data["complexity_mean"] = round(
        sum(float(m.complexity_score or 0.0) for m in mats) / len(mats), 3) if mats else 0.0

    # ссылки на артефакты отчётов
    root = cfg.root / "reports"
    data["loader_reports"] = _md_files(root, "loader_report_*.md")
    data["dataset_files"] = _md_files(root / "dataset", "*.md")
    data["charts"] = sorted(str(p.relative_to(root))
                            for p in list((root / "dataset" / "charts").glob("*.png"))
                            + list((root / "cluster" / "charts").glob("*.png")))
    data["gantts"] = _md_files(root / "predict", "gantt_*.png")
    return data


def _count(mats, attr: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for m in mats:
        v = getattr(m, attr) or "—"
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items(), key=lambda x: -x[1]))


def _md_files(root: Path, pattern: str) -> list[str]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(str(p.relative_to(root.parent)) for p in root.glob(pattern))
