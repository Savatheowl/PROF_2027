"""CLI PROF_2027.

Команды этапа 0:
- `prof2027 init [--force]` — инициализация папок, БД, каталогов, пользователей;
- `prof2027 requirements reload` — перезагрузка методических требований из YAML;
- `prof2027 status` — сводка системы;
- `prof2027 --help`.

Команды этапа 1:
- `prof2027 loader run --source <папка|manifest> [--limit N] [--max-pages N]
   [--skip-ai] [--no-ocr] [--hint TEXT]` — загрузка материалов.

Команды этапа 2:
- `prof2027 dataset build [--limit N] [--svd-components K] [--out DIR]`
   — сборка датасета признаков (csv/parquet/description/PNG + prev/next в БД).
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from prof2027 import __version__
from prof2027.core.config import load_config
from prof2027.core.logging import setup_logging
from prof2027.core.paths import project_root
from prof2027.db.base import SessionManager, make_engine
from prof2027.db.init_db import initialize, seed_requirements, seed_status


def _print_credentials(cfg: dict) -> None:
    """Печать логинов и паролей при запуске (требование КЗ)."""
    print("\n" + "=" * 60)
    print("  ДАШБОРД — учётные записи")
    print("=" * 60)
    for key, item in cfg.get("users", {}).items():
        print(f"  {item['role'].upper():<9}  {item['username']:<20} / {item['password']}")
    print("=" * 60 + "\n")


def _command_init(cfg: dict, force: bool) -> None:
    result = initialize(cfg, force=force)
    print(f"Инициализация завершена.\nБаза данных: {result['db_path']}")
    _print_credentials(cfg)


def _command_requirements_reload(cfg: dict) -> None:
    from prof2027.domain.catalogs import methodology_statistics

    db_path = project_root() / cfg["db"]["path"]
    manager = SessionManager(make_engine(db_path))
    with manager.transaction() as session:
        seed_requirements(session)
    stats = methodology_statistics()
    print(f"Требования перезагружены: {stats['total']} в {stats['categories']} категориях")


def _command_status(cfg: dict) -> None:
    db_path = project_root() / cfg["db"]["path"]
    if not db_path.is_file():
        print("База данных не создана. Выполните: prof2027 init")
        return
    manager = SessionManager(make_engine(db_path))
    with manager.session() as session:
        counts = seed_status(session)
    print(f"База данных: {db_path}")
    for table, count in counts.items():
        print(f"  {table:<12} {count}")
    _print_credentials(cfg)


def _command_loader_run(cfg: dict, args) -> int:
    from prof2027.agents.loader.pipeline import LoaderPipeline
    from prof2027.agents.loader.report import write_report
    from prof2027.ai.classifier import ClassifierAdapter
    from prof2027.ai.ocr import OcrAdapter
    from prof2027.ai.summarizer import SummarizerAdapter
    from prof2027.core.paths import ensure_dirs

    db_path = project_root() / cfg["db"]["path"]
    if not db_path.is_file():
        print("База данных не создана. Выполните: prof2027 init")
        return 1
    manager = SessionManager(make_engine(db_path))

    ocr = None if args.no_ocr else OcrAdapter()
    classifier = None if args.skip_ai else ClassifierAdapter()
    summarizer = None if args.skip_ai else SummarizerAdapter()

    pipeline = LoaderPipeline(
        cfg, manager,
        ocr=ocr, classifier=classifier, summarizer=summarizer,
    )
    report = pipeline.run(
        args.source,
        limit=args.limit,
        max_pages=args.max_pages,
        skip_ai=args.skip_ai,
        hint=args.hint,
    )

    report_path = write_report(report, ensure_dirs()["reports_dir"])
    counts = report.counts

    print(f"\nЗагрузка завершена: {report.total} файлов, "
          f"принято {counts.get('ok', 0)}, "
          f"дубликаты {counts.get('duplicate', 0)}, "
          f"ошибки {counts.get('failed', 0)}")
    print(f"Отчёт: {report_path}\n")
    return 0


def _command_dataset_build(cfg: dict, args) -> int:
    from pathlib import Path

    from prof2027.dataset.build import DatasetBuilder
    from prof2027.db.base import SessionManager, make_engine

    db_path = project_root() / cfg["db"]["path"]
    if not db_path.is_file():
        print("База данных не создана. Выполните: prof2027 init")
        return 1
    manager = SessionManager(make_engine(db_path))

    out_dir = Path(args.out).resolve() if args.out else None
    builder = DatasetBuilder(
        cfg, manager,
        out_dir=out_dir,
        svd_components=args.svd_components,
    )
    report = builder.run(limit=args.limit)

    if report.materials == 0:
        print("В базе нет материалов. Сначала выполните: prof2027 loader run")
        return 0

    print(f"\nДатасет собран: {report.rows} материалов, {report.columns} атрибутов, "
          f"SVD k={report.svd_components}, связей prev/next записано: {report.links_written}")
    print("Файлы:")
    for name, path in report.files.items():
        if path is None:
            print(f"  {name:<20} — не создан")
        else:
            print(f"  {name:<20} {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prof2027",
        description="Универсальная мультиагентная система для учебных материалов",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    init_p = sub.add_parser("init", help="инициализация системы")
    init_p.add_argument("--force", action="store_true", help="пересоздать таблицы БД")

    req_p = sub.add_parser("requirements", help="методические требования")
    req_sub = req_p.add_subparsers(dest="requirements_command", required=True)
    req_sub.add_parser("reload", help="перезагрузить требования из config/methodology.yaml")

    sub.add_parser("status", help="сводка системы")

    loader_p = sub.add_parser("loader", help="загрузка учебных материалов")
    loader_sub = loader_p.add_subparsers(dest="loader_command", required=True)
    run_p = loader_sub.add_parser("run", help="запустить загрузку из папки или manifest'а")
    run_p.add_argument("--source", required=True,
                       help="папка с материалами или manifest (.json / .txt)")
    run_p.add_argument("--limit", type=int, default=None,
                       help="ограничить число обрабатываемых файлов")
    run_p.add_argument("--max-pages", type=int, default=20,
                       help="максимум страниц/кадров для извлечения текста")
    run_p.add_argument("--skip-ai", action="store_true",
                       help="без классификации, модерации и аннотации (только парсинг)")
    run_p.add_argument("--no-ocr", action="store_true",
                       help="не загружать OCR-модель для сканов и изображений")
    run_p.add_argument("--hint", default=None,
                       help="подсказка для классификатора (например, тип источника)")

    ds_p = sub.add_parser("dataset", help="датасет и признаки (этап 2)")
    ds_sub = ds_p.add_subparsers(dest="dataset_command", required=True)
    build_p = ds_sub.add_parser("build", help="собрать датасет из БД")
    build_p.add_argument("--limit", type=int, default=None,
                         help="ограничить число материалов")
    build_p.add_argument("--svd-components", type=int, default=25,
                         help="число компонент SVD (по умолчанию 25)")
    build_p.add_argument("--out", default=None,
                         help="папка для файлов датасета (по умолчанию data/artifacts/dataset)")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # Корректная кириллица в консоли Windows (в т.ч. при перенаправлении вывода)
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")

    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config()

    if args.command == "init":
        _command_init(cfg, force=args.force)
    elif args.command == "requirements":
        _command_requirements_reload(cfg)
    elif args.command == "status":
        _command_status(cfg)
    elif args.command == "loader":
        return _command_loader_run(cfg, args)
    elif args.command == "dataset":
        return _command_dataset_build(cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
