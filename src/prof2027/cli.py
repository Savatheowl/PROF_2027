"""CLI PROF_2027.

Команды:
  prof2027 init                 — инициализация: папки, БД, каталоги, пользователи
  prof2027 config show          — показать конфигурацию
  prof2027 users show           — показать пользователей дашборда
  prof2027 loader run --manifest PATH — загрузка материалов
"""

from __future__ import annotations

import argparse

from prof2027.catalogs import Catalogs
from prof2027.config import Config
from prof2027.db import MethodologicalRequirement, Subject, get_engine, init_db, make_session, seed_catalogs
from prof2027.agents.loader.pipeline import run_loader
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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
