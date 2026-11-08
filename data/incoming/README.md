# Входящие материалы

Папка для входных файлов и manifest'ов.

Рабочий набор данных находится в `C:\Users\sava\Desktop\ms\dop_test_1`
(276 файлов, ~5,4 ГБ: PDF, DOCX, PPTX, JPEG, MP4, AVI, MP3, EPUB, FB2).

## Manifest-файлы (этап 1)

- `demo_manifest.txt` — 8 файлов разных форматов (4 PDF, EPUB, PPTX, DOCX, JPG);
  быстрый прогон: `prof2027 loader run --source data/incoming/demo_manifest.txt --skip-ai --no-ocr`.
- `ai_demo_manifest.txt` — 5 файлов с цифровым текстом (математика, геометрия, физика,
  русский язык, презентация); полный цикл с d1-2B/LaciaSUM:
  `prof2027 loader run --source data/incoming/ai_demo_manifest.txt --no-ocr`.

Формат manifest: текстовый (путь на строку, `#` — комментарий) или JSON
(список путей или список `{"path": ...}`).

## Весь корпус

`prof2027 loader run --source C:\Users\sava\Desktop\ms\dop_test_1`

Для демонстрации рекомендуется manifest на 10–20 файлов — 14 видео по ~400 МБ
обрабатывать долго (метаданные берутся из заголовков, без транскодирования).
