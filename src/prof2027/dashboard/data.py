"""Чтение данных для дашборда: KPI, таблицы, статистика из БД и отчётов."""

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
    User,
)


def kpis(session) -> dict:
    mats = session.query(Material).all()
    n = len(mats)
    by_verdict: dict[str, int] = {}
    for m in mats:
        by_verdict[m.moderation_verdict] = by_verdict.get(m.moderation_verdict, 0) + 1
    return {
        "materials": n,
        "subjects": len({m.subject for m in mats}),
        "categories": len({m.category for m in mats}),
        "approved": by_verdict.get("approved", 0),
        "needs_review": by_verdict.get("needs_review", 0),
        "rejected": by_verdict.get("rejected", 0),
        "total_hours": round(sum(float(m.estimated_time_hours or 0.0) for m in mats), 2),
    }


def materials_rows(session) -> list[dict]:
    rows = []
    for m in session.query(Material).order_by(Material.id).all():
        rows.append({
            "id": m.id, "title": m.title, "subject": m.subject, "category": m.category,
            "level": m.education_level, "language": m.language,
            "type": m.material_type, "class": m.class_type,
            "words": m.word_count, "complexity": m.complexity_score,
            "methodology": m.methodology_score, "verdict": m.moderation_verdict,
            "hours": round(float(m.estimated_time_hours or 0.0), 2),
        })
    return rows


def subject_counts(session) -> list[dict]:
    counts: dict[str, int] = {}
    for m in session.query(Material).all():
        counts[m.subject or "—"] = counts.get(m.subject or "—", 0) + 1
    return [{"subject": k, "count": v} for k, v in sorted(counts.items(), key=lambda x: -x[1])]


def compliance_means(session) -> list[dict]:
    comp = {}
    for row in session.query(MaterialCompliance).all():
        comp.setdefault(row.requirement_id, []).append(row.score)
    out = []
    for rid, scores in comp.items():
        req = session.get(MethodologicalRequirement, rid)
        out.append({"requirement": req.text if req else f"#{rid}",
                    "mean": round(sum(scores) / len(scores), 3),
                    "n": len(scores)})
    return sorted(out, key=lambda x: -x["mean"])


def cluster_runs(session) -> list[dict]:
    rows = []
    for r in session.query(ClusterRun).order_by(ClusterRun.id.desc()).all():
        m = json.loads(r.metrics_json or "{}")
        rows.append({
            "id": r.id, "strategy": r.strategy, "algorithm": r.algorithm,
            "n_clusters": r.n_clusters,
            "silhouette": m.get("silhouette"),
            "created": r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else "",
        })
    return rows


def model_versions(session) -> list[dict]:
    rows = []
    for v in session.query(ModelVersion).order_by(ModelVersion.id.desc()).all():
        m = json.loads(v.metrics_json or "{}")
        d = json.loads(v.drift_json or "{}")
        row = {"id": v.id, "task": v.task, "algorithm": v.algorithm, "version": v.version,
               "active": v.is_active, "created": v.created_at.strftime("%Y-%m-%d %H:%M") if v.created_at else ""}
        for key in ("roc_auc", "accuracy", "r2", "rmse"):
            val = m.get(key)
            if isinstance(val, dict):
                row[key] = val.get("mean")
        row["drift"] = d.get("level", "—")
        rows.append(row)
    return rows


def trajectories(session) -> list[dict]:
    rows = []
    for t in session.query(Trajectory).order_by(Trajectory.id.desc()).limit(20).all():
        out = json.loads(t.output_json or "{}")
        rows.append({
            "id": t.id, "user": t.user_ref, "days": out.get("total_days"),
            "hours": out.get("total_hours"), "deadline": out.get("deadline_days"),
            "in_time": bool(out.get("meets_deadline")),
            "created": t.created_at.strftime("%Y-%m-%d %H:%M") if t.created_at else "",
        })
    return rows


def users(session) -> list[dict]:
    return [{"id": u.id, "username": u.username, "role": u.role} for u in session.query(User).all()]


def report_files(cfg) -> list[dict]:
    out = []
    root = cfg.root / "reports"
    if root.exists():
        for fp in sorted(root.rglob("*.md"), key=lambda p: -p.stat().st_mtime)[:40]:
            out.append({"name": fp.name, "path": str(fp.relative_to(root))})
    return out


def chart_paths(cfg) -> list[str]:
    out = []
    root = cfg.root / "reports"
    for sub in ("cluster/charts", "dataset/charts"):
        d = root / sub
        if d.exists():
            for fp in sorted(d.glob("*.png")):
                out.append(str(fp.relative_to(cfg.root)))
    return out
