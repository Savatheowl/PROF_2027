"""Сборка бота aiogram 3: команды, приём файлов, траектории.

Если PROF2027_TELEGRAM_TOKEN пуст — make_bot() возвращает None, CLI выводит
подсказку и не падает (демо без интернета).
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import FSInputFile, Message

from prof2027.agents.loader.pipeline import run_loader
from prof2027.bot import handlers
from prof2027.catalogs import Catalogs
from prof2027.config import Config


def build_router(cfg: Config) -> Router:
    router = Router(name="prof2027")

    @router.message(CommandStart())
    async def cmd_start(message: Message):
        await message.answer(handlers.build_start_text(cfg))

    @router.message(Command(commands=["help"]))
    async def cmd_help(message: Message):
        await message.answer(handlers.build_help_text())

    @router.message(Command(commands=["status"]))
    async def cmd_status(message: Message):
        await message.answer(handlers.build_status_text(cfg))

    @router.message(Command(commands=["materials"]))
    async def cmd_materials(message: Message):
        await message.answer(handlers.build_materials_text(cfg))

    @router.message(Command(commands=["subjects"]))
    async def cmd_subjects(message: Message):
        await message.answer(handlers.build_subjects_text(cfg))

    @router.message(Command(commands=["trajectory"]))
    async def cmd_trajectory(message: Message, command: CommandObject):
        args = (command.args or "").split()
        hours_per_day = float(args[0]) if len(args) > 0 and _floatable(args[0]) else 2.0
        deadline_days = int(args[1]) if len(args) > 1 and args[1].isdigit() else 14
        try:
            text, result = handlers.build_trajectory_text(
                cfg, Catalogs(cfg).load(), hours_per_day, deadline_days, with_gantt=True)
        except Exception as e:
            await message.answer(f"Ошибка при построении траектории: {e}")
            return
        await message.answer(text)
        if result.get("gantt"):
            try:
                await message.answer_photo(FSInputFile(result["gantt"]),
                                           caption="Gantt-диаграмма плана")
            except Exception:
                pass

    @router.message(F.document)
    async def on_document(message: Message, bot: Bot):
        doc = message.document
        name = doc.file_name or "upload.bin"
        ext = Path(name).suffix.lower()
        if ext not in cfg.allowed_extensions:
            await message.answer(
                f"Формат «{ext or 'без расширения'}» не поддерживается. "
                f"Разрешены: {', '.join(sorted(cfg.allowed_extensions))}")
            return
        await message.answer(
            f"Загружаю «{name}» ({doc.file_size} байт). Обработка на CPU займёт "
            f"пару минут — модели загружаются при первом материале.")
        incoming = cfg.root / "data" / "incoming"
        incoming.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = incoming / f"bot_{stamp}_{name}"
        try:
            await bot.download(doc, destination=str(dest))
        except Exception as e:
            await message.answer(f"Не удалось скачать файл: {e}")
            return
        rel = dest.relative_to(cfg.root) if dest.is_relative_to(cfg.root) else dest
        manifest = incoming / f"bot_manifest_{stamp}.yaml"
        manifest.write_text(
            f"sources:\n  - path: \"{rel.as_posix()}\"\n", encoding="utf-8")
        try:
            summary = run_loader(cfg, Catalogs(cfg).load(), manifest)
        except Exception as e:
            await message.answer(f"Ошибка загрузчика: {e}")
            return
        await message.answer(handlers.build_loader_result_text(summary))

    return router


def _floatable(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def make_bot(cfg: Config):
    """Возвращает (bot, dispatcher) или None, если токен не задан."""
    token = os.getenv("PROF2027_TELEGRAM_TOKEN", "")
    if not token:
        return None
    bot = Bot(token=token)
    dp = Dispatcher()
    dp.include_router(build_router(cfg))
    return bot, dp
