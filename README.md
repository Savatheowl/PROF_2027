# PROF_2027

Единая мультиагентная система для конкурсного задания «Машинное обучение и большие данные».

## Идея

Одна универсальная система для любых учебных материалов (школа, СПО, вуз, курсы). Уровень образования — атрибут материала. Всё настраивается YAML, классификатор подхватывает сам.

## Стек

Python · SQLite (8 таблиц) · Dash · FastAPI · aiogram · scikit-learn · модели из `sava` (OCR, d1-2B, LaciaSUM) · TF-IDF · PyInstaller `--onedir`.

## Документация

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — архитектура v1.0 (короткая)
- [docs/DATA_REQUESTS.md](docs/DATA_REQUESTS.md) — данные, которые нужны
- [docs/ROADMAP.md](docs/ROADMAP.md) — этапы разработки

## Pipeline

```bash
prof2027 init
prof2027 loader run --manifest data/incoming/manifest.yaml
prof2027 dataset build
prof2027 cluster run --strategy all
prof2027 train run --task all
prof2027 continuous check
prof2027 predict time --scope material
prof2027 predict trajectory --input user.yaml
prof2027 dashboard
prof2027 api
prof2027 bot
prof2027 report all
```

## Доступы к дашборду

```
admin   / admin
analyst / analyst
viewer  / viewer
```

Печатаются в консоли при запуске. Меняются через `.env`.

## Модели

Скопированы из `sava` (не симлинки): `models/model_classifier` (d1-2B), `models/anotation-model` (LaciaSUM), `models/ocr_model` (indra-ocr). Итого ~4,6 ГБ.

## Офлайн

На демонстрации интернет не нужен: модели локальные, `HF_HUB_OFFLINE=1`, загрузка из `data/incoming/`.
