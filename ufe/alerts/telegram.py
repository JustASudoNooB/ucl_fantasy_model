"""Telegram alerts (free): deadlines in IST, line-up surprises in your squad, live switch prompts.

Setup: message @BotFather -> /newbot -> copy the token; send any message to your bot, then open
https://api.telegram.org/bot<TOKEN>/getUpdates to read your chat id. Export:
    UFE_TELEGRAM_TOKEN=...   UFE_TELEGRAM_CHAT=...
"""
from __future__ import annotations

import os

import httpx

from ..config import settings


def send(text: str) -> bool:
    cfg = settings()["alerts"]
    token, chat = os.environ.get(cfg["telegram_token_env"]), os.environ.get(cfg["telegram_chat_env"])
    if not token or not chat:
        print("[alert]", text)
        return False
    try:
        r = httpx.post(f"https://api.telegram.org/bot{token}/sendMessage",
                       json=dict(chat_id=chat, text=text, parse_mode="HTML", disable_web_page_preview=True), timeout=15)
        return r.status_code == 200
    except httpx.HTTPError:
        return False
