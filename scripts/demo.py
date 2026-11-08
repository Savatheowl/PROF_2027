"""Демо-прогон PROF_2027: весь pipeline офлайн с замером времени.

Запуск:  python scripts/demo.py
Проверка: каждая стадия печатает время; итог <= 10 минут (допустимо на демо).

Стадии: PDF-генерация -> loader (реальные модели) -> dataset -> cluster ->
train -> continuous -> predict time -> trajectory (+Gantt) -> report all.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
DB = ROOT / "data" / "demo.db"
MANIFEST = ROOT / "data" / "incoming" / "manifest_demo.yaml"
USER_YAML = ROOT / "data" / "incoming" / "user_demo.yaml"
LIMIT_MINUTES = 10.0

STAGES = [
    ("make_smoke_pdf", lambda: subprocess.run([PY, str(ROOT / "scripts" / "make_smoke_pdf.py")],
                                              cwd=ROOT, check=True)),
    ("loader run", lambda: subprocess.run(
        [PY, "-m", "prof2027.cli", "loader", "run", "--manifest", str(MANIFEST), "--db", str(DB)],
        cwd=ROOT, check=True)),
    ("dataset build", lambda: subprocess.run(
        [PY, "-m", "prof2027.cli", "dataset", "build", "--db", str(DB)], cwd=ROOT, check=True)),
    ("cluster run", lambda: subprocess.run(
        [PY, "-m", "prof2027.cli", "cluster", "run", "--strategy", "all", "--db", str(DB)],
        cwd=ROOT, check=True)),
    ("train run", lambda: subprocess.run(
        [PY, "-m", "prof2027.cli", "train", "run", "--task", "all", "--db", str(DB)],
        cwd=ROOT, check=True)),
    ("continuous check", lambda: subprocess.run(
        [PY, "-m", "prof2027.cli", "continuous", "check", "--db", str(DB)], cwd=ROOT, check=True)),
    ("predict time", lambda: subprocess.run(
        [PY, "-m", "prof2027.cli", "predict", "time", "--db", str(DB)], cwd=ROOT, check=True)),
    ("predict trajectory", lambda: subprocess.run(
        [PY, "-m", "prof2027.cli", "predict", "trajectory", "--input", str(USER_YAML),
         "--db", str(DB)], cwd=ROOT, check=True)),
    ("report all", lambda: subprocess.run(
        [PY, "-m", "prof2027.cli", "report", "all", "--db", str(DB)], cwd=ROOT, check=True)),
]


def main() -> int:
    start = time.perf_counter()
    print("PROF_2027 — демо-прогон (офлайн, замер времени)")
    print("=" * 60)
    if DB.exists():
        print("demo.db найден — пересоздаю для чистого замера.")
        DB.unlink()
    timings = []
    for name, fn in STAGES:
        t0 = time.perf_counter()
        print(f"\n--- {name} ---")
        fn()
        dt = time.perf_counter() - t0
        timings.append((name, dt))
        print(f"--- {name}: {dt:.1f} с")
    total = time.perf_counter() - start
    print("\n" + "=" * 60)
    print(f"ИТОГО: {total:.1f} с = {total / 60:.2f} мин (лимит {LIMIT_MINUTES} мин)")
    ok = total <= LIMIT_MINUTES * 60
    print("Проверка:", "В ЛИМИТЕ" if ok else "СВЕРХ ЛИМИТА")
    print("\nСтадии:")
    for name, dt in timings:
        print(f"  {name:22} {dt:7.1f} с")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
