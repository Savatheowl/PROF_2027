"""Markdown-отчёт о запуске загрузчика.

Формируется по `RunReport`: принято / пропущено / отклонено / дубликаты,
с причинами и результатами классификации. Сохраняется в reports/.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from prof2027.agents.loader.pipeline import (
    ST_DUPLICATE,
    ST_FAILED,
    ST_OK,
    ST_REJECTED_PATH,
    ST_REJECTED_SIGNATURE,
    ST_SKIPPED_TOO_LARGE,
    ST_SKIPPED_UNSUPPORTED,
    RunReport,
    SourceItem,
)

_HEADER = """# Отчёт о загрузке материалов — PROF_2027

- **Источник:** `{source}`
- **Запуск:** {started}
- **Завершено:** {finished}
- **Продолжительность:** {duration:.1f} с
- **Режим:** {ai_mode}
- **Лимит файлов:** {limit}

## Итоги

| Показатель | Значение |
|---|---:|
| Всего файлов | {total} |
| Принято | {accepted} |
| Пропущено (расширение) | {skipped_unsupported} |
| Отклонено (сигнатура) | {rejected_signature} |
| Пропущено (размер) | {skipped_too_large} |
| Отклонено (имя) | {rejected_path} |
| Дубликаты | {duplicate} |
| Ошибки обработки | {failed} |

"""


def render_markdown(report: RunReport) -> str:
    counts = report.counts
    duration = 0.0
    if report.finished_at is not None:
        duration = (report.finished_at - report.started_at).total_seconds()

    ai_mode = "AI отключён (skip-ai)" if report.skip_ai else "полный (классификация + модерация + аннотация)"
    limit = str(report.limit) if report.limit else "—"

    lines = [
        _HEADER.format(
            source=report.source,
            started=report.started_at.strftime("%Y-%m-%d %H:%M:%S"),
            finished=report.finished_at.strftime("%Y-%m-%d %H:%M:%S")
            if report.finished_at
            else "—",
            duration=duration,
            ai_mode=ai_mode,
            limit=limit,
            total=report.total,
            accepted=counts.get(ST_OK, 0),
            skipped_unsupported=counts.get(ST_SKIPPED_UNSUPPORTED, 0),
            rejected_signature=counts.get(ST_REJECTED_SIGNATURE, 0),
            skipped_too_large=counts.get(ST_SKIPPED_TOO_LARGE, 0),
            rejected_path=counts.get(ST_REJECTED_PATH, 0),
            duplicate=counts.get(ST_DUPLICATE, 0),
            failed=counts.get(ST_FAILED, 0),
        )
    ]

    accepted = [i for i in report.items if i.status == ST_OK]
    lines.append(f"## Принято ({len(accepted)})\n")
    if accepted:
        lines.append("| # | Файл | Тип | Категория | Дисциплина | Сложность | Методика | Вердикт |")
        lines.append("|---|------|-----|-----------|------------|-----------|----------|---------|")
        for idx, item in enumerate(accepted, start=1):
            lines.append(
                "| {n} | {name} | {ft} | {cat} | {subj} | {cx} | {ms} | {vd} |".format(
                    n=idx,
                    name=_escape(item.path.name),
                    ft=item.file_type or "—",
                    cat=item.category or "—",
                    subj=item.subject or "—",
                    cx=item.complexity_level or "—",
                    ms=f"{item.methodology_score:.2f}" if item.methodology_score is not None else "—",
                    vd=item.verdict or "pending",
                )
            )
    else:
        lines.append("_Материалы не загружены._")
    lines.append("")

    skipped = [
        i
        for i in report.items
        if i.status in (ST_SKIPPED_UNSUPPORTED, ST_REJECTED_SIGNATURE, ST_SKIPPED_TOO_LARGE, ST_REJECTED_PATH)
    ]
    lines.append(f"## Пропущено и отклонено ({len(skipped)})\n")
    if skipped:
        lines.append("| Файл | Статус | Причина |")
        lines.append("|------|--------|---------|")
        for item in skipped:
            lines.append(f"| {_escape(item.path.name)} | {item.status} | {_escape(item.reason or '—')} |")
    else:
        lines.append("_Нет._")
    lines.append("")

    duplicates = [i for i in report.items if i.status == ST_DUPLICATE]
    lines.append(f"## Дубликаты ({len(duplicates)})\n")
    if duplicates:
        for item in duplicates:
            lines.append(f"- `{_escape(item.path.name)}`")
    else:
        lines.append("_Нет._")
    lines.append("")

    failed = [i for i in report.items if i.status == ST_FAILED]
    lines.append(f"## Ошибки обработки ({len(failed)})\n")
    if failed:
        for item in failed:
            lines.append(f"- `{_escape(item.path.name)}` — {_escape(item.reason or '—')}")
    else:
        lines.append("_Нет._")
    lines.append("")

    return "\n".join(lines)


def write_report(report: RunReport, reports_dir: Path) -> Path:
    """Сохраняет отчёт в reports/ с временным именем. Возвращает путь."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = report.started_at.strftime("%Y%m%d_%H%M%S")
    path = reports_dir / f"loader_report_{stamp}.md"
    path.write_text(render_markdown(report), encoding="utf-8")
    return path


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
