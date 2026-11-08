"""CLI PROF_2027.

Команды:
  prof2027 init                 — инициализация: папки, БД, каталоги, пользователи
  prof2027 config show          — показать конфигурацию
  prof2027 users show           — показать пользователей дашборда
  prof2027 loader run --manifest PATH — загрузка материалов
  prof2027 dataset build        — датасет: признаки, prev/next, важность, описание, графики
  prof2027 cluster run --strategy all — кластеризация: parallel/sequential/complexity, t-SNE
  prof2027 train run --task all   — обучение: 3 задачи × ≥3 алгоритма, GroupKFold, версии
  prof2027 continuous check       — проверка дрейфа данных против обученных версий
"""

from __future__ import annotations

import argparse

from prof2027.catalogs import Catalogs
from prof2027.config import Config
from prof2027.db import MethodologicalRequirement, Subject, get_engine, init_db, make_session, seed_catalogs
from prof2027.agents.cluster.pipeline import run_cluster
from prof2027.agents.dataset.pipeline import run_dataset
from prof2027.agents.loader.pipeline import run_loader
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
    result = run_loader(cfg, catalogs, args.manifest, db=args.db)
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
        print(f"  {st:11} кластеров {m['n_clusters']:3}  силуэт {sil if sil is not None else '—':>6}  {info['chart']}")
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prof2027",
        description="Универсальная мультиагентная система для учебных материалов",
    )
    sub = parser.add_subparsers(dest="command", required=True)

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
    p_loader.add_argument("--manifest", required=True, help="путь к manifest.yaml")
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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
