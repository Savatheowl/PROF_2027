# PROF_2027 — офлайн-пакет (PyInstaller)

Сборка выполняется скриптом:

```bash
python scripts/build_package.py
```

## Результат

```
packaging/runtime/dist/prof2027/
├─ prof2027.exe          # CLI + лаунчер (меню при запуске без аргументов)
├─ _internal/            # библиотеки (torch, transformers, dash, fastapi, ...)
├─ models/               # модели sava (~4,6 ГБ): model_classifier, anotation-model, ocr_model
├─ config/               # config.yaml + YAML-каталоги (предметы, требования, ...)
├─ data/                 # входящие файлы, demo.db, артефакты
├─ reports/              # отчёты и графики
└─ .env.example          # переименуйте в .env при необходимости
```

## Запуск

```bash
prof2027.exe                      # меню лаунчера
prof2027.exe report all           # отчёты А–Д
prof2027.exe dashboard            # http://127.0.0.1:8050
prof2027.exe api                  # http://127.0.0.1:8080/docs
prof2027.exe bot                  # требует токен PROF2027_TELEGRAM_TOKEN
```

Работает без интернета: модели локальные, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`.

## Ограничения демо-набора

В `data/` приходит демо-БД из 3 материалов (дроби, кинематика, химическая связь).
Для обучения моделей с метриками нужно ≥4 материалов минимум в 2 предметах —
рекомендуется 30+ размеченных материалов (см. docs/DATA_REQUESTS.md).
