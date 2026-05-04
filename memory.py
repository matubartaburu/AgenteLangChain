"""
memory.py — Historial de conversaciones en Supabase

Tablas requeridas (ver supabase_setup.sql):
  conversations(id, phone_number, role, content, created_at)
  client_notes(id, phone_number, note, created_at)

El agente guarda hasta MAX_MESSAGES mensajes por número.
Mensajes más viejos se eliminan automáticamente via la función trim_conversation.
"""

import logging
from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_SERVICE_KEY

logger = logging.getLogger(__name__)

MAX_MESSAGES = 20

_supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


# ─────────────────────────────────────────────────────────────────────────────
#  HISTORIAL DE CONVERSACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def get_history(phone_number: str) -> list[dict]:
    """Devuelve los últimos MAX_MESSAGES mensajes del número."""
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
    """Guarda un mensaje y descarta los más viejos si supera el límite."""
    _supabase.table("conversations").insert({
        "phone_number": phone_number,
        "role":         role,
        "content":      content,
    }).execute()

    # Llamada RPC que elimina mensajes viejos directamente en la DB
    # (ver supabase_setup.sql para la definición de esta función)
    _supabase.rpc("trim_conversation", {
        "p_phone_number": phone_number,
        "p_max_messages": MAX_MESSAGES,
    }).execute()


def clear_history(phone_number: str) -> None:
    """Elimina todo el historial de un número (útil para tests o reset manual)."""
    _supabase.table("conversations").delete().eq("phone_number", phone_number).execute()


# ─────────────────────────────────────────────────────────────────────────────
#  NOTAS DE CLIENTE (memoria persistente entre conversaciones)
# ─────────────────────────────────────────────────────────────────────────────

def save_client_note(phone_number: str, note: str) -> None:
    """
    Guarda información relevante del cliente para futuras conversaciones.
    El agente detecta estas notas automáticamente durante la conversación
    (ej: 'prefiere masajes con Silvana', 'tiene lesión en rodilla derecha').
    """
    if not note or not note.strip():
        return
    try:
        _supabase.table("client_notes").insert({
            "phone_number": phone_number,
            "note":         note.strip(),
        }).execute()
        logger.info("Nota de cliente guardada | phone=%s | note=%r", phone_number, note)
    except Exception as exc:
        logger.warning("No se pudo guardar nota de cliente: %s", exc)


def get_client_notes(phone_number: str) -> str:
    """
    Recupera las notas guardadas del cliente como texto plano.
    Retorna cadena vacía si no hay notas o si la tabla no existe todavía.
    """
    try:
        result = (
            _supabase.table("client_notes")
            .select("note")
            .eq("phone_number", phone_number)
            .order("created_at", desc=False)
            .limit(10)
            .execute()
        )
        if not result.data:
            return ""
        return "- " + "\n- ".join(r["note"] for r in result.data)
    except Exception as exc:
        logger.warning("No se pudieron obtener notas del cliente: %s", exc)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
#  UTILIDADES
# ─────────────────────────────────────────────────────────────────────────────

def active_conversations() -> int:
    """Cuenta números únicos con conversaciones activas."""
    try:
        # Intenta con función RPC eficiente (COUNT DISTINCT en DB)
        result = _supabase.rpc("count_active_conversations").execute()
        return result.data or 0
    except Exception:
        # Fallback si la función RPC no existe todavía
        result = _supabase.table("conversations").select("phone_number").execute()
        return len({r["phone_number"] for r in result.data})
