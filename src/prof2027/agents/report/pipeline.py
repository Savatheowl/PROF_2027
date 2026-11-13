"""Pipeline отчётов: сборка отчётов по модулям А–Д и комплексной документации."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from prof2027.agents.report.modules import collect

SECTIONS = ["A", "B", "C", "D", "E"]

TITLES = {
    "A": "А. Сбор и обработка данных",
    "B": "Б. Анализ и визуализация",
    "C": "В. Моделирование",
    "D": "Г. Интеграция",
    "E": "Д. Демонстрация и развёртывание",
}


def run_report(cfg, catalogs, *, sections=None, db=None, out_dir=None) -> dict:
    if isinstance(sections, str):
        sections = sections.split()
    sections = sections or SECTIONS
    out_dir = Path(out_dir) if out_dir else cfg.root / "reports" / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = collect(cfg)

    builders = {"A": _section_a, "B": _section_b, "C": _section_c,
                "D": _section_d, "E": _section_e}
    parts = [
        f"# Отчёт PROF_2027",
        "",
        f"- Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Материалов в базе: **{len(data['materials'])}**",
        "",
    ]
    for s in sections:
        title = TITLES.get(s, s)
        parts.append(f"## {title}")
        parts.append("")
        parts.append(builders[s](cfg, data))
        parts.append("")

    report = "\n".join(parts)
    fp = out_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    fp.write_text(report, encoding="utf-8")

    # отдельные файлы по модулям
    made = [str(fp)]
    for s in sections:
        mfile = out_dir / f"module_{s}.md"
        mfile.write_text(f"# {TITLES.get(s, s)}\n\n{builders[s](cfg, data)}\n",
                         encoding="utf-8")
        made.append(str(mfile))

    return {"report": str(fp), "modules": made, "sections": sections,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


# --------------------------------------------------------------------------- #
def _table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def _bullet(items) -> str:
    return "\n".join(f"- {i}" for i in items)


def _metric(model: dict, key: str) -> str:
    m = model["metrics"].get(key)
    if isinstance(m, dict) and m.get("mean") is not None:
        return str(m["mean"])
    return "—"


def _section_a(cfg, data) -> str:
    mats = data["materials"]
    lines = []
    lines.append(f"**Загружено материалов:** {len(mats)}")
    lines.append("")
    lines.append("### Вердикты модерации (approved ≥ 0.7, needs_review ≥ 0.4, rejected < 0.4)")
    lines.append(_table(["Вердикт", "Кол-во"], [[k, v] for k, v in data["by_verdict"].items()]))
    lines.append("")
    lines.append("### Форматы файлов")
    lines.append(_table(["Тип", "Кол-во"], [[k, v] for k, v in data["by_file_type"].items()]))
    lines.append("")
    lines.append(f"Средняя методическая оценка: **{data['methodology_mean']}** · "
                 f"средняя сложность: **{data['complexity_mean']}**")
    lines.append("")
    lines.append("### Соответствие методическим требованиям (24)")
    rows = [[r["requirement"], r["mean"]] for r in data["compliance_means"]]
    lines.append(_table(["Требование", "Средний балл"], rows))
    lines.append("")
    if data["loader_reports"]:
        lines.append("### Отчёты загрузчика")
        lines.append(_bullet(data["loader_reports"]))
    return "\n".join(lines)


def _section_b(cfg, data) -> str:
    lines = []
    lines.append("### Предметы")
    lines.append(_table(["Предмет", "Кол-во"], [[k, v] for k, v in data["by_subject"].items()]))
    lines.append("")
    lines.append("### Уровни образования")
    lines.append(_table(["Уровень", "Кол-во"], [[k, v] for k, v in data["by_level"].items()]))
    lines.append("")
    lines.append("### Кластеризация (TF-IDF + SVD, 3 стратегии)")
    if data["clusters"]:
        rows = [[c["id"], c["strategy"], c["algorithm"], c["n_clusters"],
                 c["metrics"].get("silhouette") or "—"] for c in data["clusters"]]
        lines.append(_table(["Запуск", "Стратегия", "Алгоритм", "Кластеров", "Силуэт"], rows))
    else:
        lines.append("Запусков нет — выполните `prof2027 cluster run`.")
    lines.append("")
    lines.append("### Дашборд")
    lines.append("- Роли: `admin/admin`, `analyst/analyst`, `viewer/viewer`")
    lines.append("- Вкладки: Обзор, Материалы, Предметы, Датасет, Кластеризация, "
                 "Обучение, Дрейф, Траектории, Отчёты, Администрирование")
    lines.append("")
    if data["charts"]:
        lines.append("### Графики")
        lines.append(_bullet(data["charts"]))
    return "\n".join(lines)


def _section_c(cfg, data) -> str:
    lines = []
    lines.append("### Задачи обучения (GroupKFold по предметам, без утечки)")
    lines.append(_bullet([
        "complexity — регрессия сложности (1–5)",
        "methodology — классификация вердикта модерации",
        "cluster — классификация шага последовательного изучения",
    ]))
    lines.append("")
    if data["models"]:
        lines.append("### Версии моделей и метрики")
        rows = []
        for v in data["models"]:
            kind = "class" if v["task"] in ("methodology", "cluster") else "reg"
            if kind == "class":
                row = [v["id"], v["task"], v["algorithm"], f"v{v['version']}",
                       _metric(v, "accuracy"), _metric(v, "roc_auc"), "—", "—", v["drift"]]
            else:
                row = [v["id"], v["task"], v["algorithm"], f"v{v['version']}",
                       "—", "—", _metric(v, "r2"), _metric(v, "rmse"), v["drift"]]
            rows.append(row)
        headers = ["id", "Задача", "Алгоритм", "Версия", "accuracy", "roc_auc", "r2", "rmse", "Дрейф"]
        lines.append(_table(headers, rows))
    else:
        lines.append("Моделей нет — выполните `prof2027 train run`.")
    lines.append("")
    lines.append(f"### Время изучения")
    lines.append(f"Всего по базе: **{data['total_hours']} ч.**")
    lines.append("")
    lines.append("### Траектории обучения")
    if data["trajectories"]:
        rows = [[t["id"], t["user"], t.get("total_days", "—"), t.get("total_hours", "—"),
                 t.get("deadline_days", "—"), bool(t.get("meets_deadline"))]
                for t in data["trajectories"]]
        lines.append(_table(["id", "Пользователь", "Дней", "Часов", "Дедлайн", "В срок"], rows))
    else:
        lines.append("Траекторий нет — выполните `prof2027 predict trajectory --input user.yaml`.")
    return "\n".join(lines)


def _section_d(cfg, data) -> str:
    lines = []
    lines.append("### REST API (FastAPI)")
    lines.append("- Swagger: `http://127.0.0.1:8080/docs`")
    lines.append("- Авторизация: заголовок `X-API-Key` (переменная `PROF2027_API_KEY` в `.env`; "
                 "демо-ключ печатается при запуске)")
    lines.append("- Эндпоинты: каталоги, материалы, загрузчик, датасет, кластеризация, "
                 "обучение, дрейф, прогноз, дашборд-метрики, отчёты")
    lines.append("")
    lines.append("### Telegram-бот (aiogram 3, polling)")
    lines.append(_bullet([
        "/start, /help — приветствие и справка",
        "/status — статистика базы",
        "/materials — список материалов",
        "/subjects — предметы",
        "/trajectory 2 10 — план обучения (часы/день, дедлайн) с Gantt",
        "приём файлов: PDF/DOCX/PPTX/... → загрузчик с реальными моделями",
    ]))
    lines.append("")
    lines.append("### Документация")
    lines.append(_bullet([
        "README.md — установка, архитектура, pipeline, сервисы",
    ]))
    return "\n".join(lines)


def _section_e(cfg, data) -> str:
    lines = []
    lines.append("### Структура проекта")
    lines.append(_bullet([
        "`src/prof2027/` — ядро (config, catalogs, db, validation, parsing)",
        "`src/prof2027/agents/` — loader, dataset, cluster, train, predict, report",
        "`src/prof2027/ai/` — адаптеры моделей sava (OCR, d1-2B, LaciaSUM)",
        "`src/prof2027/dashboard/`, `api/`, `bot/` — сервисы",
        "`models/` — копии моделей (~4,6 ГБ, офлайн)",
        "`data/` — входящие, БД, артефакты; `reports/` — отчёты и графики",
    ]))
    lines.append("")
    lines.append("### CLI")
    lines.append("```bash")
    lines.append("prof2027 init")
    lines.append("prof2027 loader run --manifest data/incoming/manifest.yaml")
    lines.append("prof2027 dataset build")
    lines.append("prof2027 cluster run --strategy all")
    lines.append("prof2027 train run --task all")
    lines.append("prof2027 continuous check")
    lines.append("prof2027 predict time")
    lines.append("prof2027 predict trajectory --input data/incoming/user_demo.yaml")
    lines.append("prof2027 dashboard   # 127.0.0.1:8050")
    lines.append("prof2027 api         # 127.0.0.1:8080/docs")
    lines.append("prof2027 bot         # требует токен в .env")
    lines.append("prof2027 report all  # этот отчёт")
    lines.append("```")
    lines.append("")
    lines.append("### Офлайн-сборка (PyInstaller --onedir)")
    lines.append(_bullet([
        "каждый агент собирается в отдельную папку `--onedir`",
        "папка `models/` копируется в комплект",
        "`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` — без интернета",
        "демонстрация ≤ 10 минут",
    ]))
    lines.append("")
    lines.append("### Проверка конкурсного задания")
    lines.append(_bullet([
        "А (сбор/обработка): загрузчик, валидатор, OCR, классификация, модерация, датасет, prev/next",
        "Б (анализ): дашборд с ролями, кластеризация, силуэт, t-SNE, отчёты",
        "В (моделирование): 3 задачи × 3 алгоритма, GroupKFold, метрики, дрейф, время, траектории",
        "Г (интеграция): REST API, Telegram-бот, документация",
    ]))
    return "\n".join(lines)
