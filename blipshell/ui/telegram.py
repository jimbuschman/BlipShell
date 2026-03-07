"""Telegram bot bridge — chat with BlipShell from your phone.

Runs as a separate process (`blipshell telegram`), creates its own Agent
instance backed by the same SQLite/ChromaDB. Each Telegram chat gets its
own BlipShell session.

Setup:
    1. Message @BotFather on Telegram, /newbot, get the token
    2. Add to config.yaml:
         telegram:
           bot_token: "123456:ABC-DEF..."
           allowed_user_ids: [YOUR_TELEGRAM_USER_ID]
           enabled: true
    3. Run: blipshell telegram

To find your user ID, temporarily set allowed_user_ids to [] (allow all),
send the bot a message, and check the log for "Unauthorized user ID: XXXXX".
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from blipshell.core.agent import Agent
from blipshell.core.config import ConfigManager
from blipshell.models.config import TelegramConfig, resolve_env_vars

logger = logging.getLogger(__name__)

# Agent and session state
_agent: Optional[Agent] = None
_session_id: Optional[int] = None


async def _ensure_session() -> int:
    """Start a session if we don't have one."""
    global _session_id
    if _session_id is None:
        _session_id = await _agent.start_session()
        logger.info("Telegram session started: %d", _session_id)
    return _session_id


async def _handle_message(update, context) -> None:
    """Handle an incoming Telegram text message."""
    from telegram import Update
    from telegram.constants import ChatAction

    if not isinstance(update, Update) or not update.message or not update.message.text:
        return

    user_id = update.effective_user.id
    user_text = update.message.text.strip()
    if not user_text:
        return

    logger.info("Telegram message from %s: %s", user_id, user_text[:80])

    # Show typing indicator
    await update.message.chat.send_action(ChatAction.TYPING)

    try:
        await _ensure_session()
        response = await _agent.chat(user_text)

        # Telegram has a 4096 char limit per message
        if len(response) <= 4096:
            await update.message.reply_text(response)
        else:
            # Split into chunks at newlines
            chunks = _split_message(response, 4096)
            for chunk in chunks:
                await update.message.reply_text(chunk)

    except Exception as e:
        logger.error("Error handling Telegram message: %s", e)
        await update.message.reply_text(f"Error: {e}")


async def _handle_start(update, context) -> None:
    """Handle /start command."""
    await update.message.reply_text(
        "BlipShell connected. Send me a message and I'll respond "
        "using the same memory and LLM as the desktop app."
    )


async def _handle_newsession(update, context) -> None:
    """Handle /newsession command — start a fresh session."""
    global _session_id
    if _agent and _session_id:
        await _agent.end_session()
    _session_id = None
    await _ensure_session()
    await update.message.reply_text(f"New session started (#{_session_id}).")


async def _handle_status(update, context) -> None:
    """Handle /status command."""
    if _agent:
        status = _agent.get_status()
        lines = [f"Session: {_session_id or 'none'}"]
        for k, v in status.items():
            lines.append(f"{k}: {v}")
        await update.message.reply_text("\n".join(lines))
    else:
        await update.message.reply_text("Agent not initialized.")


def _make_user_filter(config: TelegramConfig):
    """Create a filter that only allows configured user IDs."""
    from telegram.ext import filters

    if not config.allowed_user_ids:
        logger.warning(
            "No allowed_user_ids configured — bot is open to EVERYONE. "
            "Set telegram.allowed_user_ids in config.yaml."
        )
        return filters.ALL

    class AllowedUsers(filters.BaseFilter):
        def filter(self, message) -> bool:
            user_id = message.from_user.id if message.from_user else None
            if user_id in config.allowed_user_ids:
                return True
            logger.warning("Unauthorized Telegram user ID: %s", user_id)
            return False

    return AllowedUsers()


def _split_message(text: str, max_len: int = 4096) -> list[str]:
    """Split a message into chunks, preferring newline boundaries."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        # Find last newline within limit
        split_at = text.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = max_len
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks


async def run_telegram_bot(config_path: str | None = None) -> None:
    """Start the Telegram bot (blocking)."""
    global _agent

    from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

    config_manager = ConfigManager(config_path)
    config = config_manager.load()
    tg_config = config.telegram

    token = resolve_env_vars(tg_config.bot_token)
    if not token:
        raise RuntimeError(
            "No Telegram bot token configured. "
            "Set telegram.bot_token in config.yaml or BLIPSHELL_TELEGRAM_TOKEN env var."
        )

    # Initialize BlipShell agent
    _agent = Agent(config, config_manager)
    await _agent.initialize()
    logger.info("BlipShell agent initialized for Telegram")

    user_filter = _make_user_filter(tg_config)

    # Build Telegram bot
    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", _handle_start, filters=user_filter))
    app.add_handler(CommandHandler("newsession", _handle_newsession, filters=user_filter))
    app.add_handler(CommandHandler("status", _handle_status, filters=user_filter))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & user_filter, _handle_message))

    logger.info("Telegram bot starting (polling)...")

    try:
        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)

        # Run forever until interrupted
        stop_event = asyncio.Event()
        await stop_event.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Telegram bot shutting down...")
    finally:
        if _agent:
            await _agent.end_session()
        await app.updater.stop()
        await app.stop()
        await app.shutdown()
