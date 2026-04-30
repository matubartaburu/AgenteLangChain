import logging
from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_SERVICE_KEY

logger = logging.getLogger(__name__)

MAX_MESSAGES = 20

_supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def get_history(phone_number: str) -> list[dict]:
    result = (
        _supabase.table("conversations")
        .select("role, content")
        .eq("phone_number", phone_number)
        .order("created_at", desc=False)
        .limit(MAX_MESSAGES)
        .execute()
    )
    return [{"role": r["role"], "content": r["content"]} for r in result.data]


def add_message(phone_number: str, role: str, content: str) -> None:
    _supabase.table("conversations").insert({
        "phone_number": phone_number,
        "role": role,
        "content": content,
    }).execute()

    # Descarta mensajes viejos que superen el límite (via función en DB)
    _supabase.rpc("trim_conversation", {
        "p_phone_number": phone_number,
        "p_max_messages": MAX_MESSAGES,
    }).execute()


def clear_history(phone_number: str) -> None:
    _supabase.table("conversations").delete().eq("phone_number", phone_number).execute()


def active_conversations() -> int:
    result = _supabase.table("conversations").select("phone_number").execute()
    return len({r["phone_number"] for r in result.data})
