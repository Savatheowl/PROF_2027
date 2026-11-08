# Модели PROF_2027

Реальные копии моделей из Hugging Face (не симлинки). Папка исключена из git —
модели большие (~4,5 ГБ) и доставляются скриптом `scripts/fetch_models.py`.

| Папка | HF-репозиторий | Модель | Назначение | Размер |
|---|---|---|---|---|
| `model_classifier/` | `LiquidAI/d1-2B` | d1-2B | классификация: категория, дисциплина, тема, уровень, язык, тип, сложность, соответствие требованиям | ~3,6 ГБ |
| `anotation-model/` | `LaciaStudio/Lacia_sum_small_v1` | LaciaSUM v1 | аннотации текста | ~0,85 ГБ |
| `ocr_model/` | `datalab-to/indra-ocr` | indra-ocr (ONNX) | OCR PDF/сканов/изображений | ~0,1 ГБ |

## Скачивание

```bash
python scripts/fetch_models.py            # все модели
python scripts/fetch_models.py --only d1-2B indra-ocr   # выборочно
python scripts/fetch_models.py --check    # проверить, чего не хватает
```

Загрузка из кода — по относительному пути `models/<папка>`:

- `AutoModel.from_pretrained("models/model_classifier", trust_remote_code=True)`;
- `AutoModelForSeq2SeqLM.from_pretrained("models/anotation-model")`;
- `indra_ocr.load(path="models/ocr_model")`.

Папка исключена из git (большие бинарные файлы). Код и конфиги коммитятся.
