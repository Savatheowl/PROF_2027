# PROF_2027

Универсальная мультиагентная система для учебных материалов (школа / СПО / вуз): загрузка → датасет → кластеризация → обучение → прогноз → сервисы и отчёты.

## Стек

Python · SQLite · Dash · FastAPI · aiogram · scikit-learn · (OCR, d1-2B, LaciaSUM) · TF-IDF · PyInstaller `--onedir`.

## Установка

<<<<<<< HEAD
Требуется **Python 3.11+**.

```bash
# 1. виртуальное окружение
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 2. зависимости
pip install -e .
pip install -r requirements.txt
# CPU-версия torch (без CUDA):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. конфигурация
copy .env.example .env          # Windows
# cp .env.example .env          # Linux / macOS

# 4. офлайн-модели (~4.5 ГБ, один раз)
python scripts/fetch_models.py

# 5. инициализация папок, БД, каталогов и пользователей
prof2027 init
```

Проверка: `prof2027 --help`, `prof2027 config show`.

## Архитектура

- **Одна универсальная система** для любых уровней образования: уровень (`school` / `spo` / `university` / `unknown`) — атрибут материала, а не режим работы.
- **Настройка данными:** предметы, 24 методических требования и справочники — редактируемые YAML (`config/`), классификатор строит вопросы динамически.
- **Сквозной pipeline:** `loader` (фильтр → OCR → классификация d1-2B → аннотация → модерация) → `dataset` (признаки, prev/next, важность) → `cluster` (параллельно / последовательно / по сложности) → `train` (3 задачи × ≥3 алгоритма, GroupKFold, дрейф) → `predict` (время, траектории) → `dashboard` / `api` / `bot` / `report`.
- **AI-слой:** `indra-ocr` (OCR), `d1-2B` (классификация), `LaciaSUM` (аннотация), TF-IDF + SVD (семантика); модели локальные, работают офлайн.
- **БД — 8 таблиц:** `subjects`, `materials`, `methodological_requirements`, `material_compliance`, `cluster_runs`, `model_versions`, `users`, `trajectories`.
- **Без дублей:** SHA-256 + URL; повторный запуск обновляет записи, а не копирует.
=======
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — архитектура


## Установка

```bash
python -m venv .venv
pip install -r requirements.txt
pip install -e .
python ./scripts/fetch_models.py
```

>>>>>>> 0118ef8f6c0c7e5339cd09aaa2a23179572d431a

## Pipeline

```bash
prof2027 init
<<<<<<< HEAD
python scripts/fetch_models.py        # офлайн-модели, ~4.5 ГБ, один раз
prof2027 loader run --manifest data/incoming/manifest.yaml  # по списку источников
prof2027 loader run --folder data/incoming/learning         # рекурсивно сканировать папку
=======
prof2027 loader run --manifest data/incoming/ (или через manifest.yaml)
>>>>>>> 0118ef8f6c0c7e5339cd09aaa2a23179572d431a
prof2027 dataset build
prof2027 cluster run --strategy all
prof2027 train run --task all
prof2027 continuous check
prof2027 predict time
prof2027 predict trajectory --input data/incoming/user_demo.yaml
prof2027 dashboard   # http://127.0.0.1:8050
prof2027 api         # http://127.0.0.1:8080/docs
prof2027 bot         # требует токен в .env (пустой токен не роняет)
prof2027 report all  # отчёты по модулям А–Д в reports/report/
```

## Доступы к дашборду

```
admin   / admin
analyst / analyst
viewer  / viewer
```

## Сервисы

| Сервис | Что даёт |
|---|---|
| Дашборд (Dash) | роли admin/analyst/viewer, 10 вкладок, графики и отчёты |
| REST API (FastAPI) | 24 эндпоинта, Swagger `/docs`, заголовок `X-API-Key` (`PROF2027_API_KEY`; демо-ключ печатается при запуске) |
| Telegram-бот (aiogram) | команды, приём файлов, траектория с Gantt; токен `PROF2027_TELEGRAM_TOKEN` в `.env` |
| Отчёты | `prof2027 report all` — модули А–Д + комплексная документация |
