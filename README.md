# PROF_2027


## Стек

Python · SQLite · Dash · FastAPI · aiogram · scikit-learn · (OCR, d1-2B, LaciaSUM) · TF-IDF · PyInstaller `--onedir`.

## Документация

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — архитектура
- [docs/ROADMAP.md](docs/ROADMAP.md) — этапы разработки (статус)

## Pipeline

```bash
prof2027 init
prof2027 loader run --manifest data/incoming/manifest.yaml
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



