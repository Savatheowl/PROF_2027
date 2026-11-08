"""Gantt-диаграмма плана обучения (matplotlib, Agg)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["figure.dpi"] = 110


def gantt_chart(plan: list[dict], out_path: Path) -> Path:
    """Горизонтальные полосы по материалам на оси часов. Возвращает путь PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, float, float]] = []  # (метка, start_h, end_h)
    cursor = 0.0
    for d in plan:
        day_start = cursor
        for it in d["items"]:
            rows.append((f"{it['subject']}: {it['title']}", cursor, cursor + it["hours"]))
            cursor += it["hours"]
        if d["items"]:
            # небольшой разрыв между днями на диаграмме
            cursor += 0.5
        else:
            cursor = day_start + 1.0

    fig, ax = plt.subplots(figsize=(10, max(3, len(rows) * 0.35 + 1)))
    colors = plt.get_cmap("tab20")
    for i, (label, start, end) in enumerate(rows):
        ax.barh(i, end - start, left=start, height=0.6,
                color=colors(i % 20), edgecolor="white")
        ax.text(start + 0.1, i + 0.35, label, fontsize=7, va="center")
    ax.set_yticks([])
    ax.set_xlabel("Время, часы")
    ax.set_title(f"План обучения: {len(rows)} материалов")
    ax.set_xlim(0, cursor)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
