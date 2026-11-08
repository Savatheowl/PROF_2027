"""Тесты БД: 8 таблиц, наполнение каталогами, пользователи."""

from prof2027.catalogs import Catalogs
from prof2027.config import Config
from prof2027.db import (
    Material,
    MethodologicalRequirement,
    Subject,
    User,
    get_engine,
    init_db,
    make_session,
    seed_catalogs,
)
from prof2027.security import create_default_users, hash_password, verify_password


def test_init_and_seed(cfg: Config, cat: Catalogs):
    engine = get_engine(cfg.db_path)
    init_db(engine)
    seed_catalogs(engine, cat)
    Session = make_session(engine)
    with Session() as s:
        assert s.query(Subject).count() >= 30
        assert s.query(MethodologicalRequirement).count() == 24


def test_seed_is_idempotent(cfg: Config, cat: Catalogs):
    engine = get_engine(cfg.db_path)
    init_db(engine)
    seed_catalogs(engine, cat)
    seed_catalogs(engine, cat)
    Session = make_session(engine)
    with Session() as s:
        assert s.query(Subject).count() >= 30


def test_default_users(cfg: Config):
    engine = get_engine(cfg.db_path)
    init_db(engine)
    Session = make_session(engine)
    with Session() as s:
        created = create_default_users(s, cfg)
        assert len(created) == 3
    with Session() as s:
        assert s.query(User).count() == 3
        admin = s.query(User).filter_by(role="admin").first()
        assert admin.username == "admin"
        assert verify_password("admin", admin.password_hash)
        assert not verify_password("wrong", admin.password_hash)


def test_password_roundtrip():
    digest = hash_password("secret")
    assert verify_password("secret", digest)
    assert not verify_password("Secret", digest)


def test_material_defaults(cfg: Config):
    engine = get_engine(cfg.db_path)
    init_db(engine)
    Session = make_session(engine)
    with Session() as s:
        m = Material(title="Тестовый материал", source_hash="abc123", word_count=10)
        s.add(m)
        s.commit()
        assert m.id is not None
        assert m.moderation_verdict == "pending"
        assert m.education_level == "unknown"
        assert m.complexity_level == "unknown"
