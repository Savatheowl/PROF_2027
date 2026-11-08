"""Сборка офлайн-пакета PROF_2027 (PyInstaller --onedir).

Результат: packaging/runtime/dist/prof2027/
  ├─ prof2027.exe  (+ библиотеки в _internal)
  ├─ models/       (копия ~4,6 ГБ — офлайн-модели sava)
  ├─ config/       (YAML-каталоги и настройки)
  ├─ data/         (входящие файлы, БД создаётся при первом запуске)
  ├─ reports/      (отчёты и графики)
  └─ .env.example  (переименуйте в .env при необходимости)

Запуск: packaging\runtime\dist\prof2027\prof2027.exe  (меню лаунчера)
Проверка: exe работает без интернета (HF_HUB_OFFLINE=1, модели локальные).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
RUNTIME = ROOT / "packaging" / "runtime"
ENTRY = ROOT / "packaging" / "entry_prof2027.py"

# Директории, которые копируются рядом с exe (вне бандла _internal).
DATA_DIRS = ("models", "config", "data", "reports")


def _excludes() -> list[str]:
    # Тяжёлые/неиспользуемые пакеты, чтобы не раздувать бандл.
    return ["pytest", "matplotlib.tests", "sklearn.tests", "scipy", "IPython"]


def build(no_clean: bool = False) -> Path:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    dist = RUNTIME / "dist" / "prof2027"
    if dist.exists() and not no_clean:
        shutil.rmtree(RUNTIME / "dist", ignore_errors=True)

    cmd = [
        PY, "-m", "PyInstaller",
        "--noconfirm", "--clean",
        "--onedir",
        "--name", "prof2027",
        "--paths", str(ROOT / "src"),
        "--distpath", str(RUNTIME / "dist"),
        "--workpath", str(RUNTIME / "build"),
        "--specpath", str(RUNTIME),
        # офлайн-окружение демо
        "--collect-all", "dash",
        "--collect-all", "plotly",
        "--collect-all", "aiogram",
    ]
    for mod in _excludes():
        cmd += ["--exclude-module", mod]
    cmd.append(str(ENTRY))

    print("PyInstaller: собираю CLI-бандл (может занять несколько минут)...")
    subprocess.check_call(cmd)

    # Копируем офлайн-данные рядом с exe.
    for name in DATA_DIRS:
        src = ROOT / name
        if not src.exists():
            continue
        dst = dist / name
        print(f"копирую {name} -> {dst}")
        shutil.copytree(src, dst, dirs_exist_ok=True)

    env_example = ROOT / ".env.example"
    if env_example.exists():
        shutil.copy2(env_example, dist / ".env.example")

    # Готовый отчёт о сборке
    dist / "BUILD.txt"
    (dist / "BUILD.txt").write_text(
        f"PROF_2027 офлайн-пакет\nсобрано: {_now()}\n"
        f"exe: prof2027.exe\nмодели: models/ ({_gb(dist / 'models')} ГБ)\n"
        "запуск: prof2027.exe (меню) или prof2027.exe <команда>", encoding="utf-8")

    print("Готово:", dist)
    return dist


def _now() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return round(total / (1024 ** 3), 2)


if __name__ == "__main__":
    build(no_clean="--no-clean" in sys.argv)
