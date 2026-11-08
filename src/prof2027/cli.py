"""CLI PROF_2027.

Команды:
  prof2027 init                 — инициализация: папки, БД, каталоги, пользователи
  prof2027 config show          — показать конфигурацию
  prof2027 users show           — показать пользователей дашборда
  prof2027 loader run --manifest PATH — загрузка материалов (или --folder DIR)
  prof2027 dataset build        — датасет: признаки, prev/next, важность, описание, графики
  prof2027 cluster run --strategy all — кластеризация: parallel/sequential/complexity, t-SNE
  prof2027 train run --task all   — обучение: 3 задачи × ≥3 алгоритма, GroupKFold, версии
  prof2027 continuous check       — проверка дрейфа данных против обученных версий
  prof2027 predict time           — оценка времени изучения материалов
  prof2027 predict trajectory --input user.yaml — траектория с дедлайном и Gantt
  prof2027 dashboard              — веб-дашборд (Dash): логин, роли, 10 вкладок
  prof2027 api                    — REST API (FastAPI): Swagger /docs, X-API-Key
  prof2027 bot                    — Telegram-бот (aiogram polling); пустой токен не роняет
  prof2027 report all             — отчёты по модулям А–Д и комплексная документация
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from prof2027.catalogs import Catalogs
from prof2027.config import Config
from prof2027.db import MethodologicalRequirement, Subject, get_engine, init_db, make_session, seed_catalogs
from prof2027.agents.cluster.pipeline import run_cluster
from prof2027.agents.dataset.pipeline import run_dataset
from prof2027.agents.loader.pipeline import run_loader
from prof2027.agents.predict.time import run_time_estimate
from prof2027.agents.predict.trajectory import build_trajectory
from prof2027.agents.train.continuous import run_continuous
from prof2027.agents.train.pipeline import run_train
from prof2027.security import create_default_users


def cmd_init(args: argparse.Namespace) -> int:
    cfg = Config()

    for folder in ("data/incoming", "data/raw", "data/processed", "data/artifacts", "reports"):
        (cfg.root / folder).mkdir(parents=True, exist_ok=True)
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)

    catalogs = Catalogs(cfg).load()
    engine = get_engine(cfg.db_path)
    init_db(engine)
    seed_catalogs(engine, catalogs)

    Session = make_session(engine)
    with Session() as session:
        created = create_default_users(session, cfg)
        n_subjects = session.query(Subject).count()
        n_reqs = session.query(MethodologicalRequirement).count()

    print("=" * 48)
    print("PROF_2027 инициализирован")
    print(f"  БД:          {cfg.db_path}")
    print(f"  Предметы:    {n_subjects}")
    print(f"  Требования:  {n_reqs}")
    if created:
        print("-" * 48)
        print("Доступы дашборда:")
        for username, password, role in created:
            print(f"  {username:10} {password:10} [{role}]")
    print("=" * 48)
    return 0


def cmd_config_show(args: argparse.Namespace) -> int:
    cfg = Config()
    print(f"root:         {cfg.root}")
    print(f"env:          {cfg.env}")
    print(f"db_path:      {cfg.db_path}")
    print(f"extensions:   {len(cfg.allowed_extensions)} шт.")
    print(f"max_file_mb:  {cfg.ingestion['max_file_mb']}")
    print(f"complexity:   {', '.join(cfg.complexity_levels)}")
    return 0


def cmd_users_show(args: argparse.Namespace) -> int:
    cfg = Config()
    for username, password, role in cfg.user_defaults().values():
        print(f"{username:10} {role:10} пароль по умолчанию: {password}")
    return 0


def cmd_loader_run(args: argparse.Namespace) -> int:
    cfg = Config()
    catalogs = Catalogs(cfg).load()
    if args.manifest and args.folder:
        print("укажите только один из --manifest / --folder")
        return 2
    source = args.folder or args.manifest
    if not source:
        print("нужен --manifest PATH или --folder DIR")
        return 2
    result = run_loader(cfg, catalogs, source, db=args.db)
    stats = result["stats"]
    print(f"Загрузка завершена: принято {stats['ok']}, "
          f"пропущено/отклонено {sum(v for k, v in stats.items() if k != 'ok')}")
    print(f"Отчёт: {result['report']}")
    return 0


def cmd_dataset_build(args: argparse.Namespace) -> int:
    cfg = Config()
    catalogs = Catalogs(cfg).load()
    summary = run_dataset(cfg, catalogs, db=args.db)
    print(f"Датасет: материалов {summary['materials']}, предметов {summary['subjects']}")
    print(f"  признаки: {summary['features_updated']}, prev/next: {summary['prev_next_updated']}")
    print(f"  важность:  {summary['importance_file']}")
    print(f"  описание:  {summary['description_file']}")
    print(f"  графики:   {len(summary['charts'])} шт.")
    return 0


def cmd_cluster_run(args: argparse.Namespace) -> int:
    cfg = Config()
    catalogs = Catalogs(cfg).load()
    result = run_cluster(cfg, catalogs, strategy=args.strategy, db=args.db)
    print(f"Кластеризация: материалов {result['materials']}")
    for st, info in result["strategies"].items():
        m = info["metrics"]
        sil = m.get("silhouette")
        print(f"  {st:11} кластеров {m['n_clusters']:3}  силуэт {sil if sil is not None else '-'}  {info['chart']}")
    return 0


def cmd_train_run(args: argparse.Namespace) -> int:
    cfg = Config()
    catalogs = Catalogs(cfg).load()
    summary = run_train(cfg, catalogs, task=args.task, db=args.db)
    for tname, entry in summary["tasks"].items():
        if entry.get("note"):
            print(f"  {tname:12} {entry['note']}")
            continue
        best = entry["best"]
        if best is None:
            print(f"  {tname:12} моделей не обучено (мало данных)")
            continue
        print(f"  {tname:12} лучшая {best['algorithm']:16} "
              f"({entry['best_key']}={_fmt_best(best['metrics'], entry['kind'])}) v{best['version']}")
    return 0


def _fmt_best(metrics: dict, kind: str) -> str:
    if not metrics:
        return "—"
    key = "roc_auc" if kind == "classification" and metrics.get("roc_auc") else (
        "accuracy" if kind == "classification" else "r2")
    m = metrics.get(key)
    if isinstance(m, dict) and m.get("mean") is not None:
        return str(m["mean"])
    return str(m)


def cmd_continuous_check(args: argparse.Namespace) -> int:
    cfg = Config()
    catalogs = Catalogs(cfg).load()
    result = run_continuous(cfg, catalogs, db=args.db)
    n = len(result["versions"])
    levels = {}
    for v in result["versions"].values():
        levels[v["level"]] = levels.get(v["level"], 0) + 1
    print(f"Дрейф: версий проверено {n}: " + ", ".join(f"{k}={v}" for k, v in levels.items()))
    return 0


def cmd_predict_time(args: argparse.Namespace) -> int:
    cfg = Config()
    engine = get_engine(args.db or cfg.db_path)
    init_db(engine)
    with make_session(engine)() as s:
        result = run_time_estimate(s)
        s.commit()
    print(f"Время изучения: материалов {result['materials']}, всего {result['total_hours']} ч.")
    return 0


def cmd_predict_trajectory(args: argparse.Namespace) -> int:
    cfg = Config()
    catalogs = Catalogs(cfg).load()
    params = yaml.safe_load(Path(args.input).read_text(encoding="utf-8")) or {}
    from datetime import datetime

    gantt = cfg.root / "reports" / "predict" / f"gantt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    result = build_trajectory(cfg, catalogs, params, db=args.db, gantt_out=gantt)
    status = "в срок" if result["meets_deadline"] else "НЕ в срок"
    print(f"Траектория: {result['total_days']} дн. из дедлайна {result['deadline_days']} дн., "
          f"{result['total_hours']} ч. ({status})")
    print(f"Gantt: {result['gantt']}")
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    cfg = Config()
    from prof2027.dashboard.app import make_dashboard

    app = make_dashboard(cfg)
    print(f"Дашборд: http://127.0.0.1:{args.port}")
    print("Доступы: admin/admin · analyst/analyst · viewer/viewer")
    app.run(debug=False, host=args.host, port=args.port)
    return 0


def cmd_api(args: argparse.Namespace) -> int:
    if args.db:
        os.environ["PROF2027_DB_PATH"] = args.db
    cfg = Config()
    from prof2027.api.app import make_api, api_key_value
    from prof2027.api import app as api_mod

    api_mod._config.cache_clear()
    app = make_api(cfg)
    print(f"API: http://127.0.0.1:{args.port}")
    print(f"Swagger: http://127.0.0.1:{args.port}/docs")
    print(f"API-ключ (X-API-Key): {api_key_value(cfg)}")
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_bot(args: argparse.Namespace) -> int:
    if args.db:
        os.environ["PROF2027_DB_PATH"] = args.db
    cfg = Config()
    from prof2027.bot.app import make_bot

    built = make_bot(cfg)
    if built is None:
        print("Бот не запущен: PROF2027_TELEGRAM_TOKEN пуст. Задайте токен в .env")
        print("(демо работает офлайн — бот можно включить позже).")
        return 0
    bot, dp = built
    print(f"Бот запущен (polling). Остановка: Ctrl+C.")
    import asyncio

    asyncio.run(dp.start_polling(bot))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    if args.db:
        os.environ["PROF2027_DB_PATH"] = args.db
    cfg = Config()
    catalogs = Catalogs(cfg).load()
    from prof2027.agents.report.pipeline import run_report

    result = run_report(cfg, catalogs, sections=args.sections)
    print(f"Отчёт: {result['report']}")
    print(f"Модули: {len(result['modules'])} файлов в {Path(result['report']).parent}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prof2027",
        description="Универсальная мультиагентная система для учебных материалов",
    )
    sub = parser.add_subparsers(dest="command")

    p_init = sub.add_parser("init", help="инициализация: папки, БД, каталоги, пользователи")
    p_init.set_defaults(func=cmd_init)

    p_cfg = sub.add_parser("config", help="работа с конфигурацией")
    p_cfg.add_argument("action", choices=["show"], default="show", nargs="?")
    p_cfg.set_defaults(func=cmd_config_show)

    p_users = sub.add_parser("users", help="пользователи дашборда")
    p_users.add_argument("action", choices=["show"], default="show", nargs="?")
    p_users.set_defaults(func=cmd_users_show)

    p_loader = sub.add_parser("loader", help="загрузка материалов")
    p_loader.add_argument("action", choices=["run"], default="run", nargs="?")
    p_loader.add_argument("--manifest", default=None, help="путь к manifest.yaml (список источников)")
    p_loader.add_argument("--folder", default=None, help="папка с файлами — сканируется автоматически")
    p_loader.add_argument("--db", default=None, help="переопределить путь к БД")
    p_loader.set_defaults(func=cmd_loader_run)

    p_dataset = sub.add_parser("dataset", help="датасет: признаки, prev/next, важность, описание, графики")
    p_dataset.add_argument("action", choices=["build"], default="build", nargs="?")
    p_dataset.add_argument("--db", default=None, help="переопределить путь к БД")
    p_dataset.set_defaults(func=cmd_dataset_build)

    p_cluster = sub.add_parser("cluster", help="кластеризация материалов")
    p_cluster.add_argument("action", choices=["run"], default="run", nargs="?")
    p_cluster.add_argument("--strategy", default="all", choices=["all", "parallel", "sequential", "complexity"])
    p_cluster.add_argument("--db", default=None, help="переопределить путь к БД")
    p_cluster.set_defaults(func=cmd_cluster_run)

    p_train = sub.add_parser("train", help="обучение моделей")
    p_train.add_argument("action", choices=["run"], default="run", nargs="?")
    p_train.add_argument("--task", default="all",
                         choices=["all", "complexity", "methodology", "cluster"])
    p_train.add_argument("--db", default=None, help="переопределить путь к БД")
    p_train.set_defaults(func=cmd_train_run)

    p_cont = sub.add_parser("continuous", help="проверка дрейфа данных")
    p_cont.add_argument("action", choices=["check"], default="check", nargs="?")
    p_cont.add_argument("--db", default=None, help="переопределить путь к БД")
    p_cont.set_defaults(func=cmd_continuous_check)

    p_predict = sub.add_parser("predict", help="прогноз времени и траекторий")
    p_predict.add_argument("action", choices=["time", "trajectory"], default="time")
    p_predict.add_argument("--input", default=None, help="user.yaml для траектории")
    p_predict.add_argument("--db", default=None, help="переопределить путь к БД")
    p_predict.set_defaults(func=_dispatch_predict)

    p_dash = sub.add_parser("dashboard", help="веб-дашборд (Dash)")
    p_dash.add_argument("--host", default="127.0.0.1")
    p_dash.add_argument("--port", type=int, default=8050)
    p_dash.set_defaults(func=cmd_dashboard)

    p_api = sub.add_parser("api", help="REST API (FastAPI)")
    p_api.add_argument("--host", default="127.0.0.1")
    p_api.add_argument("--port", type=int, default=8080)
    p_api.add_argument("--db", default=None, help="переопределить путь к БД")
    p_api.set_defaults(func=cmd_api)

    p_bot = sub.add_parser("bot", help="Telegram-бот (aiogram polling)")
    p_bot.add_argument("--db", default=None, help="переопределить путь к БД")
    p_bot.set_defaults(func=cmd_bot)

    p_report = sub.add_parser("report", help="отчёты по модулям А–Д")
    p_report.add_argument("action", choices=["all"], default="all", nargs="?")
    p_report.add_argument("--sections", default="A B C D E",
                          help="пробелы: A B C D E (по умолчанию все)")
    p_report.add_argument("--db", default=None, help="переопределить путь к БД")
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        _print_menu()
        return 0
    return args.func(args)


def _print_menu() -> None:
    print("PROF_2027 — универсальная система учебных материалов")
    print("")
    print("Команды:")
    print("  init                  инициализация (папки, БД, каталоги, пользователи)")
    print("  loader run --manifest ...   загрузка (список источников) или --folder DIR (сканирование папки)")
    print("  dataset build         признаки, prev/next, важность, описание, графики")
    print("  cluster run           кластеризация (parallel/sequential/complexity, t-SNE)")
    print("  train run             обучение 3 задач x >=3 алгоритма (GroupKFold)")
    print("  continuous check      проверка дрейфа данных")
    print("  predict time          оценка времени изучения")
    print("  predict trajectory --input user.yaml   план обучения + Gantt")
    print("  dashboard             веб-дашборд http://127.0.0.1:8050 (admin/admin)")
    print("  api                   REST API http://127.0.0.1:8080/docs")
    print("  bot                   Telegram-бот (нужен токен в .env)")
    print("  report all            отчёты по модулям А-Д")
    print("")
    print("Подробнее: prof2027 --help. Данные офлайн: models/ рядом с программой.")


def _dispatch_predict(args: argparse.Namespace) -> int:
    if args.action == "time":
        return cmd_predict_time(args)
    return cmd_predict_trajectory(args)


if __name__ == "__main__":
    raise SystemExit(main())
