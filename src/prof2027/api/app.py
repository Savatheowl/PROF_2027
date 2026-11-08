"""FastAPI-приложение PROF_2027: 20+ эндпоинтов, Swagger /docs, API-ключ.

Покрывает данные и запуски всех агентов: каталоги, материалы, загрузчик,
датасет, кластеризация, обучение, дрейф, прогноз, дашборд-метрики, отчёты.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import PlainTextResponse

from prof2027.agents.cluster.pipeline import run_cluster
from prof2027.agents.dataset.describe import describe
from prof2027.agents.dataset.features import build_frame, numeric_columns
from prof2027.agents.dataset.importance import compute_importance
from prof2027.agents.loader.pipeline import run_loader
from prof2027.agents.predict.time import estimate_hours, run_time_estimate
from prof2027.agents.predict.trajectory import build_trajectory
from prof2027.agents.train.continuous import run_continuous
from prof2027.agents.train.pipeline import run_train
from prof2027.api.schemas import (
    ClusterRunRequest,
    LoaderRunRequest,
    TimeEstimateRequest,
    TrainRunRequest,
    TrajectoryRequest,
)
from prof2027.catalogs import Catalogs
from prof2027.config import Config
from prof2027.db import (
    Material,
    MaterialCompliance,
    MethodologicalRequirement,
    ModelVersion,
    Trajectory,
    get_engine,
    init_db,
    make_session,
)

DEFAULT_API_KEY = "prof2027-demo-key"


@lru_cache(maxsize=1)
def _config() -> Config:
    return Config()


def get_config() -> Config:
    return _config()


def api_key_value(cfg: Config) -> str:
    return os.getenv("PROF2027_API_KEY", "") or DEFAULT_API_KEY


def require_key(
    x_api_key: str | None = Header(default=None),
    cfg: Config = Depends(get_config),
) -> None:
    if x_api_key != api_key_value(cfg):
        raise HTTPException(status_code=401, detail="неверный или отсутствующий API-ключ")


def _session(cfg: Config, db: str | None = None):
    engine = get_engine(db or cfg.db_path)
    init_db(engine)
    return make_session(engine)


def _material_detail(s, m: Material, with_text: bool = False) -> dict:
    out = {
        "id": m.id,
        "title": m.title,
        "subject": m.subject,
        "category": m.category,
        "topic": m.topic,
        "education_level": m.education_level,
        "language": m.language,
        "material_type": m.material_type,
        "class_type": m.class_type,
        "annotation": m.annotation,
        "word_count": m.word_count,
        "media_count": m.media_count,
        "complexity_score": m.complexity_score,
        "complexity_level": m.complexity_level,
        "methodology_score": m.methodology_score,
        "moderation_verdict": m.moderation_verdict,
        "estimated_time_hours": m.estimated_time_hours,
        "cluster_parallel": m.cluster_parallel,
        "cluster_sequential": m.cluster_sequential,
        "has_previous": m.has_previous,
        "has_next": m.has_next,
        "previous_material_id": m.previous_material_id,
        "next_material_id": m.next_material_id,
        "created_at": m.created_at.strftime("%Y-%m-%d %H:%M:%S") if m.created_at else "",
    }
    if with_text:
        out["text_content"] = m.text_content
        out["media_descriptions"] = m.media_descriptions
    out["compliance"] = [
        {"requirement_id": c.requirement_id, "score": c.score, "is_compliant": c.is_compliant}
        for c in s.query(MaterialCompliance).filter_by(material_id=m.id).order_by(
            MaterialCompliance.requirement_id
        ).all()
    ]
    out["features"] = json.loads(m.features_json or "{}")
    return out


def make_api(cfg: Config | None = None):
    cfg = cfg or Config()
    app = FastAPI(
        title="PROF_2027 API",
        description="REST API универсальной системы учебных материалов. "
                    "Авторизация: заголовок X-API-Key.",
        version="1.0",
    )

    @app.get("/health", tags=["system"])
    def health(cfg: Config = Depends(get_config)):
        with _session(cfg)() as s:
            n = s.query(Material).count()
        return {"status": "ok", "project": cfg.project_name,
                "materials": n, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    @app.get("/", include_in_schema=False)
    def root():
        return PlainTextResponse("PROF_2027 API. Документация: /docs. Авторизация: X-API-Key.")

    # ---------------------------------------------------------------- catalogs
    @app.get("/catalogs/subjects", tags=["catalogs"],
             dependencies=[Depends(require_key)])
    def catalogs_subjects(cfg: Config = Depends(get_config)):
        return Catalogs(cfg).load().subjects

    @app.get("/catalogs/methodology", tags=["catalogs"],
             dependencies=[Depends(require_key)])
    def catalogs_methodology(cfg: Config = Depends(get_config)):
        return Catalogs(cfg).load().methodology

    @app.get("/catalogs/languages", tags=["catalogs"],
             dependencies=[Depends(require_key)])
    def catalogs_languages(cfg: Config = Depends(get_config)):
        return Catalogs(cfg).load().languages

    @app.get("/catalogs/material_types", tags=["catalogs"],
             dependencies=[Depends(require_key)])
    def catalogs_material_types(cfg: Config = Depends(get_config)):
        return Catalogs(cfg).load().material_types

    @app.get("/catalogs/class_types", tags=["catalogs"],
             dependencies=[Depends(require_key)])
    def catalogs_class_types(cfg: Config = Depends(get_config)):
        return Catalogs(cfg).load().class_types

    # --------------------------------------------------------------- materials
    @app.get("/materials", tags=["materials"], dependencies=[Depends(require_key)])
    def list_materials(
        subject: str | None = Query(None),
        category: str | None = Query(None),
        level: str | None = Query(None),
        verdict: str | None = Query(None),
        limit: int = Query(100, ge=1, le=1000),
        cfg: Config = Depends(get_config),
    ):
        with _session(cfg)() as s:
            q = s.query(Material)
            if subject:
                q = q.filter(Material.subject == subject)
            if category:
                q = q.filter(Material.category == category)
            if level:
                q = q.filter(Material.education_level == level)
            if verdict:
                q = q.filter(Material.moderation_verdict == verdict)
            rows = [_material_detail(s, m) for m in q.order_by(Material.id).limit(limit).all()]
        return {"count": len(rows), "items": rows}

    @app.get("/materials/{material_id}", tags=["materials"],
             dependencies=[Depends(require_key)])
    def get_material(material_id: int, cfg: Config = Depends(get_config)):
        with _session(cfg)() as s:
            m = s.get(Material, material_id)
            if m is None:
                raise HTTPException(404, f"материал {material_id} не найден")
            return _material_detail(s, m, with_text=True)

    # ------------------------------------------------------------------ loader
    @app.post("/loader/run", tags=["loader"], dependencies=[Depends(require_key)])
    def loader_run(req: LoaderRunRequest, cfg: Config = Depends(get_config)):
        catalogs = Catalogs(cfg).load()
        manifest = (cfg.root / req.manifest) if not Path(req.manifest).is_absolute() else Path(req.manifest)
        if not manifest.exists():
            raise HTTPException(404, f"manifest не найден: {manifest}")
        try:
            result = run_loader(cfg, catalogs, manifest, db=req.db)
        except Exception as e:
            raise HTTPException(500, f"загрузка не удалась: {e}")
        return {"accepted": result["stats"]["ok"], "skipped_or_rejected": result["stats"].get("skipped_duplicate", 0)
                + result["stats"].get("skipped_unsupported", 0) + result["stats"].get("skipped_too_large", 0)
                + result["stats"].get("rejected_signature", 0) + result["stats"].get("rejected_path", 0)
                + result["stats"].get("error", 0),
                "report": result["report"], "results": result["results"]}

    # ----------------------------------------------------------------- dataset
    @app.get("/dataset/stats", tags=["dataset"], dependencies=[Depends(require_key)])
    def dataset_stats(cfg: Config = Depends(get_config)):
        cat = Catalogs(cfg).load()
        with _session(cfg)() as s:
            frame = build_frame(s, cat)
            stats = describe(frame)
        return stats

    @app.get("/dataset/importance", tags=["dataset"], dependencies=[Depends(require_key)])
    def dataset_importance(cfg: Config = Depends(get_config)):
        cat = Catalogs(cfg).load()
        with _session(cfg)() as s:
            frame = build_frame(s, cat)
            imp = compute_importance(frame)
        return imp

    @app.get("/dataset/features/{material_id}", tags=["dataset"],
             dependencies=[Depends(require_key)])
    def material_features(material_id: int, cfg: Config = Depends(get_config)):
        with _session(cfg)() as s:
            m = s.get(Material, material_id)
            if m is None:
                raise HTTPException(404, f"материал {material_id} не найден")
            features = json.loads(m.features_json or "{}")
            cols = numeric_columns(build_frame(s, Catalogs(cfg).load()))
            return {"material_id": m.id, "columns": cols, "features": features}

    # ---------------------------------------------------------------- cluster
    @app.post("/cluster/run", tags=["cluster"], dependencies=[Depends(require_key)])
    def cluster_run(req: ClusterRunRequest, cfg: Config = Depends(get_config)):
        cat = Catalogs(cfg).load()
        try:
            result = run_cluster(cfg, cat, strategy=req.strategy)
        except Exception as e:
            raise HTTPException(500, f"кластеризация не удалась: {e}")
        return {"materials": result["materials"], "strategies": result["strategies"],
                "report_dir": result["report_dir"]}

    @app.get("/cluster/runs", tags=["cluster"], dependencies=[Depends(require_key)])
    def cluster_runs(cfg: Config = Depends(get_config)):
        with _session(cfg)() as s:
            rows = []
            from prof2027.db import ClusterRun
            for r in s.query(ClusterRun).order_by(ClusterRun.id.desc()).limit(50).all():
                m = json.loads(r.metrics_json or "{}")
                rows.append({"id": r.id, "strategy": r.strategy, "algorithm": r.algorithm,
                             "n_clusters": r.n_clusters, "silhouette": m.get("silhouette"),
                             "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S")
                             if r.created_at else ""})
        return rows

    # ------------------------------------------------------------------ train
    @app.post("/train/run", tags=["train"], dependencies=[Depends(require_key)])
    def train_run(req: TrainRunRequest, cfg: Config = Depends(get_config)):
        cat = Catalogs(cfg).load()
        try:
            summary = run_train(cfg, cat, task=req.task)
        except Exception as e:
            raise HTTPException(500, f"обучение не удалось: {e}")
        return summary

    @app.get("/models", tags=["train"], dependencies=[Depends(require_key)])
    def list_models(cfg: Config = Depends(get_config)):
        with _session(cfg)() as s:
            rows = []
            for v in s.query(ModelVersion).order_by(ModelVersion.id.desc()).limit(100).all():
                m = json.loads(v.metrics_json or "{}")
                d = json.loads(v.drift_json or "{}")
                row = {"id": v.id, "task": v.task, "algorithm": v.algorithm,
                       "version": v.version, "is_active": v.is_active,
                       "file_path": v.file_path, "drift": d.get("level"),
                       "created_at": v.created_at.strftime("%Y-%m-%d %H:%M:%S")
                       if v.created_at else ""}
                for key in ("roc_auc", "accuracy", "r2", "rmse"):
                    val = m.get(key)
                    row[key] = val.get("mean") if isinstance(val, dict) else val
                rows.append(row)
        return rows

    @app.get("/models/best", tags=["train"], dependencies=[Depends(require_key)])
    def best_models(cfg: Config = Depends(get_config)):
        """Лучшая активная версия по каждой задаче (по максимальной ключевой метрике)."""
        from prof2027.agents.train.pipeline import _metric_value

        cat = Catalogs(cfg).load()
        with _session(cfg)() as s:
            best: dict[str, tuple[float, ModelVersion]] = {}
            for v in s.query(ModelVersion).filter_by(is_active=True).all():
                m = json.loads(v.metrics_json or "{}")
                kind = "classification" if v.task in ("methodology", "cluster") else "regression"
                score = _metric_value(m, kind)
                if v.task not in best or score > best[v.task][0]:
                    best[v.task] = (score, v)
        return {t: {"version": v.version, "algorithm": v.algorithm,
                    "score": round(score, 3), "file_path": v.file_path}
                for t, (score, v) in best.items()}

    @app.post("/continuous/check", tags=["train"], dependencies=[Depends(require_key)])
    def continuous_check(cfg: Config = Depends(get_config)):
        cat = Catalogs(cfg).load()
        try:
            result = run_continuous(cfg, cat)
        except Exception as e:
            raise HTTPException(500, f"проверка дрейфа не удалась: {e}")
        return result

    # ---------------------------------------------------------------- predict
    @app.post("/predict/time", tags=["predict"], dependencies=[Depends(require_key)])
    def predict_time(req: TimeEstimateRequest):
        return {"hours": estimate_hours(req.word_count, req.media_count,
                                        req.complexity_level, req.complexity_score)}

    @app.post("/predict/time/all", tags=["predict"], dependencies=[Depends(require_key)])
    def predict_time_all(cfg: Config = Depends(get_config)):
        with _session(cfg)() as s:
            result = run_time_estimate(s)
            s.commit()
        return result

    @app.post("/predict/trajectory", tags=["predict"], dependencies=[Depends(require_key)])
    def predict_trajectory(req: TrajectoryRequest, cfg: Config = Depends(get_config)):
        cat = Catalogs(cfg).load()
        params = {"user": req.user, "hours_per_day": req.hours_per_day,
                  "deadline_days": req.deadline_days}
        for key, val in (("subject", req.subject), ("category", req.category),
                         ("level", req.level)):
            if val:
                params[key] = val
        gantt_out = None
        if req.gantt:
            gantt_out = cfg.root / "reports" / "predict" / \
                f"api_gantt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        try:
            result = build_trajectory(cfg, cat, params, gantt_out=gantt_out)
        except Exception as e:
            raise HTTPException(500, f"траектория не построена: {e}")
        return result

    @app.get("/trajectories", tags=["predict"], dependencies=[Depends(require_key)])
    def list_trajectories(cfg: Config = Depends(get_config)):
        with _session(cfg)() as s:
            rows = []
            for t in s.query(Trajectory).order_by(Trajectory.id.desc()).limit(50).all():
                out = json.loads(t.output_json or "{}")
                rows.append({"id": t.id, "user": t.user_ref,
                             "total_days": out.get("total_days"),
                             "total_hours": out.get("total_hours"),
                             "deadline_days": out.get("deadline_days"),
                             "meets_deadline": out.get("meets_deadline"),
                             "created_at": t.created_at.strftime("%Y-%m-%d %H:%M:%S")
                             if t.created_at else ""})
        return rows

    # ------------------------------------------------------------------ misc
    @app.get("/dashboard/kpis", tags=["dashboard"], dependencies=[Depends(require_key)])
    def dashboard_kpis(cfg: Config = Depends(get_config)):
        from prof2027.dashboard import data as ddata

        with _session(cfg)() as s:
            return ddata.kpis(s)

    @app.get("/reports", tags=["system"], dependencies=[Depends(require_key)])
    def list_reports(cfg: Config = Depends(get_config)):
        from prof2027.dashboard import data as ddata

        return ddata.report_files(cfg)

    return app
