"""Notification channels for the daily signal (stdlib only).

Credentials are read from environment variables, never from the config
file (which is committed to git):

* Telegram: ``TELEGRAM_BOT_TOKEN``, ``TELEGRAM_CHAT_ID``
* Email (SMTP): ``SMTP_HOST``, ``SMTP_PORT`` (default 587),
  ``SMTP_USER``, ``SMTP_PASSWORD``, ``EMAIL_TO``

A failing channel logs a warning and does not block the others.
"""

from __future__ import annotations

import json
import os
import smtplib
import urllib.request
from email.message import EmailMessage


def send_console(message: str) -> None:
    """Print the message to stdout."""
    print(message)


def send_telegram(message: str) -> None:
    """Send the message through a Telegram bot."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in the environment."
        )
    payload = json.dumps({"chat_id": chat_id, "text": message}).encode("utf-8")
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"Telegram API returned HTTP {response.status}")


def send_email(message: str, subject: str = "vol_ml_fund — signal quotidien") -> None:
    """Send the message by email over SMTP with STARTTLS."""
    host = os.environ.get("SMTP_HOST")
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASSWORD")
    recipient = os.environ.get("EMAIL_TO")
    port = int(os.environ.get("SMTP_PORT", "587"))
    if not all([host, user, password, recipient]):
        raise RuntimeError(
            "SMTP_HOST, SMTP_USER, SMTP_PASSWORD and EMAIL_TO must be set "
            "in the environment."
        )
    email = EmailMessage()
    email["From"] = user
    email["To"] = recipient
    email["Subject"] = subject
    email.set_content(message)
    with smtplib.SMTP(host, port, timeout=30) as server:
        server.starttls()
        server.login(user, password)
        server.send_message(email)


_CHANNELS = {
    "console": send_console,
    "telegram": send_telegram,
    "email": send_email,
}


def dispatch(message: str, channels: list[str]) -> dict[str, str]:
    """Send the message on every configured channel.

    Returns a per-channel status ("ok" or the error message); a failure
    on one channel never prevents the others.
    """
    statuses: dict[str, str] = {}
    for channel in channels:
        sender = _CHANNELS.get(channel)
        if sender is None:
            statuses[channel] = f"unknown channel (choose from {list(_CHANNELS)})"
            continue
        try:
            sender(message)
            statuses[channel] = "ok"
        except Exception as error:  # noqa: BLE001 — report, don't block others
            statuses[channel] = str(error)
    return statuses
