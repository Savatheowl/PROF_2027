"""Отчёт о запуске загрузчика (Markdown)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

STATUS_NAMES = {
    "ok": "принят",
    "skipped_duplicate": "дубль",
    "skipped_unsupported": "неподдерживаемый формат",
    "skipped_too_large": "слишком большой",
    "rejected_signature": "сигнатура не совпадает",
    "rejected_path": "небезопасное имя",
    "error": "ошибка",
}


def write_report(path: Path | str, manifest_path: Path | str, results: list[dict], stats: dict, cfg) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    total = len(results)
    lines = [
        "# Отчёт загрузчика PROF_2027",
        "",
        f"- Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Manifest: `{manifest_path}`",
        f"- Источников: {total}",
        f"- Принято: {stats.get('ok', 0)}",
        f"- Пропущено/отклонено: {total - stats.get('ok', 0)}",
        "",
        "## Результаты",
        "",
        "| # | Источник | Статус | Причина |",
        "|---:|---|---|---|",
    ]
    for r in results:
        status = STATUS_NAMES.get(r.get("status", ""), r.get("status", ""))
        reason = (r.get("reason") or "").replace("|", "/")
        source = (r.get("source") or "").replace("|", "/")
        lines.append(f"| {r['index']} | {source} | {status} | {reason} |")

    lines += ["", "## Итог", ""]
    lines.append(f"Материалов добавлено: **{stats.get('ok', 0)}** из {total}.")
    lines.append("")
    for key, name in STATUS_NAMES.items():
        if stats.get(key):
            lines.append(f"- {name}: {stats[key]}")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path
