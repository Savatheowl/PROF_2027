"""Интеграционный тест pipeline загрузчика (AI-адаптеры заменены фейками)."""

from pathlib import Path

from prof2027.agents.loader.pipeline import run_loader
from prof2027.db import Material, MaterialCompliance, make_session


class FakeOcr:
    def is_available(self):
        return False

    def convert(self, fp, max_pages=20):
        return ""


class FakeSummarizer:
    def summarize(self, text, max_length=150):
        return "Аннотация тестового материала."


class FakeClassifier:
    def __init__(self, catalogs):
        self.catalogs = catalogs

    def classify(self, text, title):
        return {
            "subject": "Алгебра",
            "subject_confidence": 0.9,
            "subject_is_auto": False,
            "category": "Математика и статистика",
            "education_level": "school",
            "language": "ru",
            "material_type": "student_material",
            "class_type": "lesson",
            "topic": "Дроби",
            "complexity_score": 2.0,
            "complexity_level": "низкая",
            "compliance": [{"requirement": r, "score": 0.8, "is_compliant": True} for r in self.catalogs.requirements],
        }


def _setup(tmp_path, cfg, cat):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    (incoming / "m1.txt").write_text(
        "Обыкновенные дроби. Числитель и знаменатель. Примеры задач по дробям.", encoding="utf-8")
    # m2.txt — идентичен m1.txt → должен уйти как дубль
    (incoming / "m2.txt").write_text(
        "Обыкновенные дроби. Числитель и знаменатель. Примеры задач по дробям.", encoding="utf-8")
    # небезопасный формат
    (incoming / "tool.exe").write_bytes(b"MZ\x90\x00\x03\x00\x00\x00\x00\x00\x00\x00")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "sources:\n"
        f"  - path: \"{(incoming / 'm1.txt').as_posix()}\"\n"
        f"  - path: \"{(incoming / 'm2.txt').as_posix()}\"\n"
        f"  - path: \"{(incoming / 'tool.exe').as_posix()}\"\n",
        encoding="utf-8",
    )
    return manifest


def test_loader_pipeline(tmp_path, cfg, cat):
    manifest = _setup(tmp_path, cfg, cat)
    db = tmp_path / "loader.db"
    res = run_loader(
        cfg, cat, manifest, db=db,
        media_root=tmp_path / "media", report_path=tmp_path / "report.md",
        ocr=FakeOcr(), summarizer=FakeSummarizer(), classifier=FakeClassifier(cat),
    )

    assert res["stats"]["ok"] == 1
    assert res["stats"]["skipped_duplicate"] == 1
    assert res["stats"]["skipped_unsupported"] == 1

    from prof2027.db import get_engine

    engine = get_engine(db)
    with make_session(engine)() as s:
        mats = s.query(Material).all()
        assert len(mats) == 1
        m = mats[0]
        assert m.subject == "Алгебра"
        assert m.category == "Математика и статистика"
        assert m.education_level == "school"
        assert m.topic == "Дроби"
        assert m.moderation_verdict == "approved"  # mean 0.8 >= 0.7
        assert len(m.text_content) > 0
        compliance = s.query(MaterialCompliance).filter_by(material_id=m.id).all()
        assert len(compliance) == 24

    report = Path(res["report"])
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "принят" in text and "дубль" in text


def test_loader_idempotent_second_run(tmp_path, cfg, cat):
    manifest = _setup(tmp_path, cfg, cat)
    db = tmp_path / "loader2.db"
    fake = FakeClassifier(cat)
    kwargs = dict(db=db, media_root=tmp_path / "media2",
                  report_path=tmp_path / "r2.md", ocr=FakeOcr(),
                  summarizer=FakeSummarizer(), classifier=fake)
    first = run_loader(cfg, cat, manifest, **kwargs)
    second = run_loader(cfg, cat, manifest, **kwargs)
    assert first["stats"]["ok"] == 1
    assert second["stats"]["ok"] == 0
    assert second["stats"]["skipped_duplicate"] == 2


def test_loader_folder_scan(tmp_path, cfg, cat):
    """Папка вместо манифеста: сканируется по разрешённым расширениям."""
    from prof2027.agents.loader.manifest import sources_from_dir

    folder = tmp_path / "learning"
    folder.mkdir()
    (folder / "math.txt").write_text(
        "Обыкновенные дроби. Числитель и знаменатель. Примеры задач по дробям.", encoding="utf-8")
    (folder / "ignore.exe").write_bytes(b"MZ\x90\x00\x03\x00\x00\x00\x00\x00\x00\x00")
    sub = folder / "sub"
    sub.mkdir()
    (sub / "chem.md").write_text("Химические связи. Ковалентная связь. Ионная связь.", encoding="utf-8")

    srcs = sources_from_dir(folder, cfg.allowed_extensions)
    names = sorted(Path(s["path"]).name for s in srcs)
    assert names == ["chem.md", "math.txt"]  # exe отфильтрован, подпапки пройдены

    db = tmp_path / "folder.db"
    res = run_loader(cfg, cat, folder, db=db, media_root=tmp_path / "media",
                     report_path=tmp_path / "report_folder.md",
                     ocr=FakeOcr(), summarizer=FakeSummarizer(),
                     classifier=FakeClassifier(cat))
    assert res["stats"]["ok"] == 2
    assert res["stats"]["skipped_unsupported"] == 0
