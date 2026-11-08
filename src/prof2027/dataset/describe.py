"""Описание атрибутов датасета → `dataset_description.md`.

Для каждой колонки таблицы признаков формируется блок:
тип, смысл, число непустых значений, уникальных, частоты (топ-5),
числовые статистики. Дополнительно печатается таблица важности признаков.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

# Смысл каждой колонки (для читаемого описания в markdown)
COLUMN_MEANINGS: dict[str, str] = {
    "material_id": "Внутренний идентификатор материала в БД",
    "title": "Название материала",
    "category": "Категория (каталог config/catalogs): предметная область",
    "subject": "Дисциплина (предмет) внутри категории",
    "topic": "Тема материала",
    "education_level": "Уровень образования: школа / СПО / вуз / unknown",
    "language": "Язык материала",
    "material_type": "Тип материала (учебник, конспект, практикум, …)",
    "class_type": "Тип занятия (лекция, семинар, практика, …)",
    "file_type": "Тип исходного файла (pdf, docx, epub, …)",
    "complexity_level": "Уровень сложности (5 уровней: очень низкая … очень высокая)",
    "moderation_verdict": "Вердикт модерации: approved / needs_review / rejected / pending",
    "word_count": "Число слов в тексте материала",
    "char_count": "Число символов в тексте материала",
    "sentence_count": "Число предложений в тексте",
    "media_count": "Общее число медиа-объектов (иллюстрации, аудио, видео)",
    "complexity_score": "Числовая оценка сложности (классификатор d1-2B)",
    "methodology_score": "Средний балл методических требований (0..1)",
    "estimated_time_hours": "Расчётное время изучения, часов",
    "avg_word_len": "Средняя длина слова, символов",
    "avg_sentence_len_words": "Средняя длина предложения, слов",
    "unique_words": "Число уникальных словоформ",
    "ttr": "Type-token ratio: отношение уникальных слов к общему числу",
    "uppercase_ratio": "Доля букв в верхнем регистре (заголовки)",
    "media_image": "Число изображений в материале",
    "media_audio": "Число аудио-объектов",
    "media_video": "Число видео-объектов",
    "has_media": "Есть ли в материале медиа",
    "previous_material_id": "Предыдущий материал по порядку изучения внутри предмета",
    "next_material_id": "Следующий материал по порядку изучения внутри предмета",
    "has_previous": "Есть предыдущий материал (не первый в последовательности)",
    "has_next": "Есть следующий материал (не последний в последовательности)",
    "semantic_neighbor_id": "Ближайший материал по семантической близости (SVD)",
    "_svd_k": "Число компонент SVD, добавленных в строку",
}


def _is_numeric(col: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(col)


def _stats_line(col: pd.Series) -> str:
    """Числовые статистики одной строкой."""
    vals = col.dropna()
    if vals.empty:
        return "  - Числовые статистики: нет данных\n"
    return (
        f"  - min = {vals.min():.4g}  |  median = {vals.median():.4g}  |  "
        f"mean = {vals.mean():.4g}  |  max = {vals.max():.4g}  |  "
        f"std = {vals.std():.4g}\n"
    )


def _stringify_unhashable(value):
    """Списки/словари → строковое представление (для частот и уникальных)."""
    if isinstance(value, (list, dict)):
        return repr(value)
    return value


def _freq_lines(col: pd.Series, top: int = 5) -> list[str]:
    """Частоты значений (топ) в формате markdown."""
    total = int(col.notna().sum())
    if total == 0:
        return ["  - Частоты: нет данных\n"]
    counts = col.value_counts(dropna=True).head(top)
    lines = []
    for value, count in counts.items():
        pct = 100.0 * count / total
        text = str(value)
        if len(text) > 48:
            text = text[:45] + "…"
        lines.append(f"  - `{text}` — {int(count)} ({pct:.1f}%)\n")
    return lines


def _attr_block(name: str, col: pd.Series, meanings: Mapping[str, str]) -> str:
    """Markdown-блок описания одного атрибута."""
    meaning = meanings.get(name, "—")
    n_total = len(col)
    n_notna = int(col.notna().sum())

    # Нехэшируемые значения (списки/словари) приводим к строке для статистик
    if not _is_numeric(col):
        col = col.map(_stringify_unhashable)

    n_unique = int(col.nunique(dropna=True))

    if _is_numeric(col):
        kind = "числовой"
    elif pd.api.types.is_bool_dtype(col):
        kind = "логический"
    else:
        kind = "категориальный/текстовый"

    lines = [f"### `{name}`\n"]
    lines.append(f"- Смысл: {meaning}\n")
    lines.append(f"- Тип: {kind} (dtype: {col.dtype})\n")
    lines.append(f"- Непустых значений: {n_notna} из {n_total} (пропущено {n_total - n_notna})\n")
    lines.append(f"- Уникальных значений: {n_unique}\n")
    if _is_numeric(col):
        lines.append(_stats_line(col))
    else:
        lines.extend(_freq_lines(col))
    return "\n".join(lines) + "\n"


def _render_importance(imp: pd.DataFrame, target: str) -> str:
    """Markdown-таблица важности признаков по целевой переменной."""
    lines = [f"### Важность признаков → `{target}`\n"]
    if imp.empty:
        lines.append("Методы не применимы (мало данных).\n")
        return "\n".join(lines) + "\n"

    is_note = (imp["type"] == "note").any()
    if is_note:
        for _i, row in imp.iterrows():
            method = row.get("feature", "").strip("[]")
            reason = row.get(method, "")
            lines.append(f"- **{method}**: {reason}")
        lines.append("")
        return "\n".join(lines) + "\n"

    cols = [c for c in imp.columns if c not in ("feature", "type")]
    lines.append("| Признак | " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * (len(cols) + 1))
    for _i, row in imp.iterrows():
        cells = [str(row["feature"])]
        for c in cols:
            v = row.get(c)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                cells.append("—")
            else:
                cells.append(f"{v:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines) + "\n"


def build_description(
    df: pd.DataFrame,
    *,
    title: str | None = None,
    meanings: Mapping[str, str] | None = None,
    importance: Mapping[str, pd.DataFrame] | None = None,
) -> str:
    """Полный markdown-отчёт `dataset_description.md`."""
    meanings = meanings or COLUMN_MEANINGS

    head = [f"# Датасет материалов — {title or 'описание'}\n"]
    head.append("")
    head.append(f"- Материалов в датасете: **{len(df)}**")
    head.append(f"- Атрибутов (колонок): **{len(df.columns)}**")
    head.append("")
    head.append("## Атрибуты\n")

    blocks = [_attr_block(name, df[name], meanings) for name in df.columns]
    body = "\n".join(head) + "\n" + "\n".join(blocks)

    if importance:
        sections = ["## Важность признаков\n"]
        sections.append(
            "Методы: Pearson/Spearman корреляция, mutual information, "
            "permutation importance (Ridge MSE). «—» означает неприменимость на данной выборке.\n"
        )
        for target, imp in importance.items():
            sections.append(_render_importance(imp, target))
        body += "\n" + "\n".join(sections)

    body += "\n*Отчёт сформирован автоматически модулем `dataset build`.*\n"
    return body
