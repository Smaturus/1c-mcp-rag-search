"""Owner conversation for index progress ticks (shared by ctl + stop-hook)."""

from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNER = ROOT / "data" / "index_chat_owner.json"
LAST_CONV = ROOT / "data" / "index_chat_last_conversation.json"

# Placeholder from early setup. Stop-hook must not treat it as a real chat.
STUB_CONVERSATION_ID = "reindex-chat-uuid"


def is_real_conversation_id(cid: str | None) -> bool:
    """True when ``cid`` can bind followups to one chat."""
    text = (cid or "").strip()
    return bool(text) and text != STUB_CONVERSATION_ID


def extract_conversation_id(payload: dict | None) -> str | None:
    """Read conversation id from a Cursor hook payload (several key shapes)."""
    if not isinstance(payload, dict):
        return None
    for key in (
        "conversation_id",
        "conversationId",
        "composerId",
        "composer_id",
        "chatId",
        "chat_id",
    ):
        raw = payload.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    nested = payload.get("conversation")
    if isinstance(nested, dict):
        for key in ("id", "conversation_id", "conversationId"):
            raw = nested.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
    return None


def read_owner() -> dict | None:
    if not OWNER.exists():
        return None
    try:
        data = json.loads(OWNER.read_text(encoding="utf-8"))
    except Exception:
        return None
    cid = (data.get("conversation_id") or "").strip()
    if not is_real_conversation_id(cid):
        return None
    return data


def owner_conversation_id() -> str | None:
    data = read_owner()
    if not data:
        return None
    return (data.get("conversation_id") or "").strip() or None


def read_last_conversation_id() -> str | None:
    if not LAST_CONV.exists():
        return None
    try:
        data = json.loads(LAST_CONV.read_text(encoding="utf-8"))
    except Exception:
        return None
    cid = (data.get("conversation_id") or "").strip()
    return cid if is_real_conversation_id(cid) else None


def write_owner(cid: str, note: str = "") -> dict:
    """Bind ticks to ``cid``. Refuses the stub id."""
    if not is_real_conversation_id(cid):
        raise RuntimeError(
            "нужен настоящий идентификатор чата, не заглушка; "
            "отправьте сообщение в чате переиндексации и повторите claim"
        )
    payload = {
        "conversation_id": cid.strip(),
        "claimed_unix": time.time(),
        "note": note or "reindex",
    }
    OWNER.parent.mkdir(parents=True, exist_ok=True)
    OWNER.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def claim_from_last(note: str = "") -> dict:
    cid = read_last_conversation_id()
    if not cid:
        raise RuntimeError(
            "нет настоящего идентификатора чата; "
            "отправьте любое сообщение в чате переиндексации, затем claim"
        )
    return write_owner(cid, note=note)


def adopt_if_unclaimed(cid: str | None, note: str = "stop-hook") -> str | None:
    """If there is no real owner yet, claim ``cid``. Returns the owner id."""
    own = owner_conversation_id()
    if own:
        return own
    if not is_real_conversation_id(cid):
        return None
    write_owner(cid or "", note=note)
    return (cid or "").strip()


def clear_owner() -> None:
    OWNER.unlink(missing_ok=True)


def is_owner(conversation_id: str | None) -> bool:
    own = owner_conversation_id()
    if not own or not conversation_id:
        return False
    return own == conversation_id.strip()
