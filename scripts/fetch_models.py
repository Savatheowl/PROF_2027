"""Скачивание офлайн-моделей PROF_2027 с Hugging Face.

Модели большие (~4,5 ГБ) и не хранятся в git — их нужно скачать
один раз перед офлайн-демо. Скрипт раскладывает их по папкам models/,
откуда их читает код (AutoModel.from_pretrained, indra_ocr.load).

Запуск:
  python scripts/fetch_models.py            # скачать все модели
  python scripts/fetch_models.py --only d1-2B indra-ocr   # выборочно
  python scripts/fetch_models.py --check    # только проверить, чего не хватает
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# (ключ, HF-репозиторий, локальная папка, маркер готовности)
MODELS = [
    ("d1-2B", "LiquidAI/d1-2B", "model_classifier", "model.safetensors"),
    ("LaciaSUM", "LaciaStudio/Lacia_sum_small_v1", "anotation-model", "model.safetensors"),
    ("indra-ocr", "datalab-to/indra-ocr", "ocr_model", "det.onnx"),
]

NAMES = {name: (repo, folder, marker) for name, repo, folder, marker in MODELS}


def _installed(key: str) -> bool:
    _repo, folder, marker = NAMES[key]
    return (MODELS_DIR / folder / marker).exists()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Скачивание офлайн-моделей с Hugging Face")
    ap.add_argument("--only", nargs="*", default=None, choices=list(NAMES),
                    help="скачать только указанные модели (по умолчанию — все)")
    ap.add_argument("--check", action="store_true",
                    help="не скачивать, только вывести, чего не хватает")
    args = ap.parse_args(argv)

    keys = args.only or list(NAMES)
    missing, have = [], []
    for key in keys:
        (have if _installed(key) else missing).append(key)

    print(f"models/: {len(have)} готовы, {len(missing)} отсутствуют")
    for key in keys:
        _repo, folder, marker = NAMES[key]
        status = "OK" if key in have else "нет"
        print(f"  {key:10} [{status}] models/{folder}/  (маркер: {marker})")

    if args.check or not missing:
        return 0

    print("\nСкачивание с Hugging Face...")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        print(f"Ошибка: нужен huggingface_hub ({e}). Установите: pip install huggingface_hub")
        return 1

    for key in missing:
        repo, folder, _marker = NAMES[key]
        dst = MODELS_DIR / folder
        print(f"\n[{key}] {repo} -> {dst}")
        dst.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(repo_id=repo, local_dir=dst)
        except Exception as e:
            print(f"  ошибка: {type(e).__name__}: {e}")
            return 1
        if not _installed(key):
            print(f"  предупреждение: маркер {_marker} не найден после загрузки")
            return 1

    print("\nГотово. Все модели на месте.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
