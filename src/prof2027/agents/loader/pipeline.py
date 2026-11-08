"""Pipeline загрузчика: источник → фильтр → парсинг → аннотация → классификация → модерация → БД.

Все AI-модели загружаются лениво и только при первом материале.
Для тестов можно передать фейковые адаптеры (ocr, summarizer, classifier).
"""

from __future__ import annotations

import hashlib
import urllib.request
from datetime import datetime
from pathlib import Path

from sqlalchemy import or_

from prof2027.ai.classifier import Classifier
from prof2027.ai.ocr import Ocr
from prof2027.ai.summarizer import Summarizer
from prof2027.agents.loader.manifest import load_manifest
from prof2027.agents.loader.report import write_report
from prof2027.catalogs import Catalogs
from prof2027.config import Config
from prof2027.db import (
    Material,
    MaterialCompliance,
    MethodologicalRequirement,
    get_engine,
    init_db,
    make_session,
    seed_catalogs,
)
from prof2027.parsing import parse_file
from prof2027.validation import quarantine, validate_file

MAX_TEXT = 50000  # символы текста, передаваемые в классификатор и суммаризатор


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest_dir: Path, name: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    fp = dest_dir / name
    urllib.request.urlretrieve(url, fp)
    return fp


def _verdict(score: float, cfg: Config) -> str:
    if score >= cfg.moderation["approved_threshold"]:
        return "approved"
    if score >= cfg.moderation["review_threshold"]:
        return "needs_review"
    return "rejected"


def run_loader(
    cfg: Config,
    catalogs: Catalogs,
    manifest_path: Path | str,
    *,
    models_dir: Path | str | None = None,
    db: Path | str | None = None,
    media_root: Path | str | None = None,
    download_dir: Path | str | None = None,
    report_path: Path | str | None = None,
    ocr=None,
    summarizer=None,
    classifier=None,
) -> dict:
    models_dir = Path(models_dir) if models_dir else cfg.root / "models"
    ocr = ocr or Ocr(models_dir)
    summarizer = summarizer or Summarizer(models_dir)
    classifier = classifier or Classifier(models_dir, catalogs)

    engine = get_engine(db or cfg.db_path)
    init_db(engine)
    seed_catalogs(engine, catalogs)
    Session = make_session(engine)

    with Session() as s:
        req_map = {r.text: r.id for r in s.query(MethodologicalRequirement).all()}

    media_root = Path(media_root) if media_root else cfg.root / "data" / "raw" / "media"
    download_dir = Path(download_dir) if download_dir else cfg.root / "data" / "incoming" / "cached"
    download_dir.mkdir(parents=True, exist_ok=True)

    sources = load_manifest(manifest_path)
    results: list[dict] = []
    stats = {
        "ok": 0, "skipped_duplicate": 0, "skipped_unsupported": 0,
        "skipped_too_large": 0, "rejected_signature": 0,
        "rejected_path": 0, "error": 0,
    }
    total_mb = 0.0

    for idx, src in enumerate(sources, 1):
        entry = {
            "index": idx,
            "source": src.get("url") or src.get("path", ""),
            "status": "", "reason": "", "material_id": None,
        }
        try:
            if src.get("url"):
                name = Path(src["url"].split("?")[0]).name or f"download_{idx}"
                try:
                    path = _download(src["url"], download_dir, name)
                except Exception as e:  # на демо без интернета URL недоступен
                    entry.update(status="error", reason=f"скачивание: {e}")
                    results.append(entry); stats["error"] += 1
                    continue
            elif src.get("path"):
                path = cfg.root / src["path"]
            else:
                entry.update(status="error", reason="источник без url/path")
                results.append(entry); stats["error"] += 1
                continue

            vr = validate_file(path, cfg, total_mb)
            if vr.status != "ok":
                if vr.status == "rejected_signature":
                    quarantine(path, cfg)
                entry.update(status=vr.status, reason=vr.reason)
                stats[vr.status] = stats.get(vr.status, 0) + 1
                results.append(entry)
                continue
            total_mb += path.stat().st_size / (1024 * 1024)

            file_hash = _sha256(path)
            with Session() as s:
                dup = s.query(Material).filter(
                    or_(Material.source_hash == file_hash, Material.source_url == src.get("url", ""))
                ).first()
                if dup is not None:
                    entry.update(status="skipped_duplicate", reason=f"уже в БД: id={dup.id}")
                    results.append(entry); stats["skipped_duplicate"] += 1
                    continue

            parsed = parse_file(path, media_root, ocr)
            annotation = summarizer.summarize(parsed.text[:MAX_TEXT])
            cls = classifier.classify(parsed.text[:MAX_TEXT], parsed.title)

            scores = [c["score"] for c in cls["compliance"]]
            methodology_score = round(sum(scores) / len(scores), 3) if scores else 0.0
            verdict = _verdict(methodology_score, cfg)

            with Session() as s:
                m = Material(
                    source_url=src.get("url") or None,
                    source_hash=file_hash,
                    file_path=str(path),
                    file_type=path.suffix.lower(),
                    title=parsed.title,
                    category=cls["category"],
                    subject=cls["subject"],
                    topic=cls["topic"],
                    education_level=cls["education_level"],
                    language=cls["language"],
                    material_type=cls["material_type"],
                    class_type=cls["class_type"],
                    text_content=parsed.text[:MAX_TEXT],
                    annotation=annotation,
                    media_descriptions=repr(parsed.media),
                    word_count=parsed.word_count,
                    char_count=parsed.char_count,
                    sentence_count=parsed.sentence_count,
                    media_count=len(parsed.media),
                    complexity_score=cls["complexity_score"],
                    complexity_level=cls["complexity_level"],
                    methodology_score=methodology_score,
                    moderation_verdict=verdict,
                )
                s.add(m)
                s.flush()
                for c in cls["compliance"]:
                    req_id = req_map.get(c["requirement"])
                    if req_id is not None:
                        s.add(MaterialCompliance(
                            material_id=m.id,
                            requirement_id=req_id,
                            score=c["score"],
                            is_compliant=c["is_compliant"],
                            details=f"d1-2B, p={c['score']:.2f}",
                        ))
                s.commit()
                mid = m.id

            entry.update(status="ok", reason=f"id={mid}, {cls['subject']}, {verdict}", material_id=mid)
            stats["ok"] += 1
            results.append(entry)
        except Exception as e:
            entry.update(status="error", reason=str(e))
            results.append(entry)
            stats["error"] += 1

    report_file = report_path or cfg.root / "reports" / f"loader_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    write_report(report_file, manifest_path, results, stats, cfg)

    return {"stats": stats, "report": str(report_file), "results": results}
