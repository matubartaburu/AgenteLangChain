"""
Servidor FastAPI — Agente WhatsApp Valentina · Spa Ángela De María

Stack 100% Kapso:
  · Webhook entrante: POST /kapso  (JSON, evento whatsapp.message.received)
  · Envío de respuestas: POST https://api.kapso.ai/meta/whatsapp/v24.0/{phone_number_id}/messages
  · Auth: X-API-Key

DESARROLLO LOCAL CON NGROK:
  1. uvicorn main:app --port 8000
  2. ngrok http 8000
  3. Pegá la URL pública en Kapso → Webhooks → "Add Webhook"
     Endpoint: https://<id>.ngrok-free.dev/kapso
     Eventos: whatsapp.message.received
"""

import time
import base64
import asyncio
import random
import logging
import httpx
from fastapi import FastAPI, Request
from anthropic import Anthropic

from config import KAPSO_API_KEY, ANTHROPIC_API_KEY
from memory import get_history, add_message, active_conversations, save_client_note, get_client_notes
from agent import run_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

KAPSO_API_BASE = "https://api.kapso.ai"

FALLBACK_MESSAGE = (
    "Lo siento, en este momento tengo un inconveniente técnico. "
    "Por favor, intentá de nuevo en unos minutos o llamanos al 2600 5557."
)

# ── Tiempos (segundos) ───────────────────────────────────────────────────────
AGGREGATION_WINDOW = 8       # silencio antes de procesar mensajes acumulados
INITIAL_PAUSE = (5, 9)       # "leer + pensar" antes del primer bloque
BETWEEN_PAUSE = (1, 3)       # pausa breve antes de tipear el siguiente bloque
TYPING_SPEED  = 14           # chars/segundo (tipeo móvil casual)

# ── Estado en memoria por número ─────────────────────────────────────────────
_buffer:    dict[str, list[str]]    = {}   # mensajes pendientes
_last_seen: dict[str, float]        = {}   # timestamp último mensaje
_workers:   dict[str, asyncio.Task] = {}   # task que procesa ese número
_meta:      dict[str, dict]         = {}   # phone_number_id por número
_seen_keys: set[str]                = set()  # idempotencia (X-Idempotency-Key)


# ── Sender via Kapso API ─────────────────────────────────────────────────────

async def send_whatsapp_text(phone_number_id: str, to: str, body: str) -> None:
    """Envía un mensaje de texto vía Kapso (Meta Cloud API proxy)."""
    url = f"{KAPSO_API_BASE}/meta/whatsapp/v24.0/{phone_number_id}/messages"
    headers = {"X-API-Key": KAPSO_API_KEY, "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "text",
        "text": {"body": body},
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code >= 400:
            logger.error("Kapso API error %d: %s", resp.status_code, resp.text)
        resp.raise_for_status()


# ── Manejo de media (audio + imagen) ─────────────────────────────────────────

_anthropic_vision = Anthropic(api_key=ANTHROPIC_API_KEY)


async def _download_kapso_media(url: str) -> bytes:
    """Descarga bytes de un media URL de Kapso (requiere auth)."""
    headers = {"X-API-Key": KAPSO_API_KEY}
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.content


async def _describe_image(image_bytes: bytes, mime_type: str, caption: str) -> str:
    """
    Usa Claude Sonnet 4.6 con visión para describir lo que muestra la foto
    en contexto de un spa de estética. Devuelve 1-2 frases en rioplatense.
    """
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    prompt = (
        "Esta es una foto que mandó un cliente o clienta del spa Ángela De María "
        "(belleza, estética, masajes) por WhatsApp.\n"
        "En 1-2 frases NEUTRAS y descriptivas, decí qué se ve en la foto y, si aplica, "
        "qué tipo de tratamiento estético podría estar consultando.\n"
        "No saludes, no respondas al cliente, no des opiniones sobre la persona. "
        "Solo describí objetivamente lo que muestra la foto."
    )
    if caption:
        prompt += f'\nJunto a la foto la persona escribió: "{caption}"'

    response = await asyncio.to_thread(
        _anthropic_vision.messages.create,
        model="claude-sonnet-4-6",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return response.content[0].text.strip()


async def send_typing_indicator(phone_number_id: str, message_id: str) -> None:
    """
    Marca el mensaje como leído y muestra "escribiendo..." en el chat.
    El indicador dura ~25 segundos o hasta que se envía un mensaje (lo que ocurra primero).
    """
    url = f"{KAPSO_API_BASE}/meta/whatsapp/v24.0/{phone_number_id}/messages"
    headers = {"X-API-Key": KAPSO_API_KEY, "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": message_id,
        "typing_indicator": {"type": "text"},
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code >= 400:
                logger.warning("Typing indicator falló %d: %s", resp.status_code, resp.text)
    except Exception as exc:
        # No crítico — si falla seguimos con el flujo normal
        logger.warning("Error mandando typing indicator: %s", exc)


# ── Lógica de respuesta ──────────────────────────────────────────────────────

import re

def _split_blocks(text: str) -> list[str]:
    """
    Divide la respuesta en bloques para enviar como mensajes WhatsApp separados.
    Prioridad:
      1. Si el modelo puso ||| → usar esos cortes
      2. Si hay pregunta final con ¿ → separarla del cuerpo
      3. Si el texto es largo y sin corte → buscar corte natural en oración
      4. Caso base → un solo bloque
    """
    # 1. Cortes explícitos del modelo
    if "|||" in text:
        raw = [b.strip() for b in text.split("|||") if b.strip()]
        # Cada bloque puede seguir siendo largo; intentamos resplit en ese caso
        result = []
        for b in raw:
            result.extend(_maybe_resplit_long(b))
        # Si excedemos el max, preservar el último (suele ser la pregunta)
        if len(result) > _MAX_BLOCKS:
            last = result[-1]
            result = result[:_MAX_BLOCKS - 1] + [last]
        return result

    text = text.strip()

    # 2. Pregunta final aislada
    last_q = text.rfind("¿")
    if last_q > 10:
        statement = text[:last_q].strip()
        question = text[last_q:].strip()
        if statement and question:
            return _maybe_resplit_long(statement) + [question]

    # 3. Sin pregunta pero texto largo → buscar partido natural
    if len(text) > 150:
        return _maybe_resplit_long(text)

    return [text]


_TARGET_BLOCK_LEN = 180  # chars deseados por bloque cuando hay que partir
_MAX_BLOCKS = 4          # límite duro de mensajes WhatsApp por respuesta


def _maybe_resplit_long(text: str) -> list[str]:
    """
    Si el texto pasa el target, lo parte respetando oraciones (puntos, ?, !).
    Si no hay puntos pero es muy largo, fallback a partir por comas.
    Devuelve hasta _MAX_BLOCKS bloques.
    """
    text = text.strip()
    if len(text) <= _TARGET_BLOCK_LEN:
        return [text]

    # 1. Intentar partir en oraciones (punto/!/? seguido de espacio + mayúscula o ¿)
    units = re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ¿])", text)

    # 2. Si quedó como una sola unidad y es muy largo, partir por comas como último recurso
    if len(units) == 1 and len(text) > 240:
        units = [u.strip() for u in re.split(r",\s+", text) if u.strip()]
        joiner = ", "
    else:
        joiner = " "

    if len(units) <= 1:
        return [text]  # no se puede partir natural

    # 3. Agrupar unidades consecutivas en bloques de hasta TARGET chars
    blocks: list[str] = []
    current = ""
    for u in units:
        if not current:
            current = u
        elif len(current) + len(u) + len(joiner) <= _TARGET_BLOCK_LEN:
            current = current + joiner + u
        else:
            blocks.append(current)
            current = u
    if current:
        blocks.append(current)

    # 4. Si quedó solo uno (todas las unidades juntas no superaban el target), devolver original
    if len(blocks) <= 1:
        return [text]

    return blocks[:_MAX_BLOCKS]


def _typing_delay(text: str, pause_range: tuple[float, float]) -> float:
    return random.uniform(*pause_range) + len(text) / TYPING_SPEED


async def _wait_for_silence(phone: str) -> None:
    while True:
        elapsed = time.time() - _last_seen.get(phone, 0)
        if elapsed >= AGGREGATION_WINDOW:
            return
        await asyncio.sleep(AGGREGATION_WINDOW - elapsed + 0.1)


async def _worker(phone: str) -> None:
    """Loop por número que procesa los turnos en orden (sin solapamiento)."""
    try:
        while True:
            await _wait_for_silence(phone)
            messages = _buffer.pop(phone, [])
            if not messages:
                return

            combined = " ".join(messages)
            logger.info("Procesando | from=%s | mensajes=%d | texto=%r",
                        phone, len(messages), combined)

            add_message(phone, "user", combined)
            history = get_history(phone)
            # Notas persistidas del cliente (de conversaciones anteriores)
            previous_notes = get_client_notes(phone)
            if previous_notes:
                logger.info("Cliente recurrente con %d nota(s)", previous_notes.count("\n- ") + 1)

            try:
                reply, handoff_needed, client_note = await asyncio.to_thread(run_agent, history, previous_notes)
                if client_note:
                    save_client_note(phone, client_note)
                if handoff_needed:
                    logger.warning("⚠️  HANDOFF | phone=%s | revisar conversación", phone)
            except Exception as exc:
                logger.exception("Error en agente | from=%s | %s", phone, exc)
                reply = FALLBACK_MESSAGE

            add_message(phone, "assistant", reply)

            meta = _meta.get(phone, {})
            phone_number_id = meta.get("phone_number_id")
            last_wamid = meta.get("last_wamid")
            if not phone_number_id:
                logger.error("Sin phone_number_id para %s — no se puede responder", phone)
                continue

            blocks = _split_blocks(reply)
            for i, block in enumerate(blocks):
                # Mostrar "escribiendo..." antes de cada bloque (dura ~25s, se descarta al enviar)
                if last_wamid:
                    await send_typing_indicator(phone_number_id, last_wamid)

                pause_range = INITIAL_PAUSE if i == 0 else BETWEEN_PAUSE
                delay = _typing_delay(block, pause_range)
                logger.info("Esperando %.1fs antes del bloque [%d/%d] (%d chars)",
                            delay, i + 1, len(blocks), len(block))
                await asyncio.sleep(delay)

                try:
                    await send_whatsapp_text(phone_number_id, phone, block)
                    logger.info("Bloque enviado [%d/%d] | to=%s | %r",
                                i + 1, len(blocks), phone, block)
                except Exception as exc:
                    logger.error("Error enviando bloque a %s: %s", phone, exc)
    finally:
        _workers.pop(phone, None)


async def _enqueue(phone: str, text: str) -> None:
    if phone not in _buffer:
        _buffer[phone] = []
    _buffer[phone].append(text)
    _last_seen[phone] = time.time()
    worker = _workers.get(phone)
    if worker is None or worker.done():
        _workers[phone] = asyncio.create_task(_worker(phone))


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Valentina — Spa Ángela De María",
    description="Agente de WhatsApp para el Spa Ángela De María",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "agent": "Valentina",
        "spa": "Ángela De María",
        "active_conversations": active_conversations(),
        "pending_buffers": len(_buffer),
        "active_workers": sum(1 for w in _workers.values() if not w.done()),
    }


def _extract_events(body) -> list[dict]:
    """Acepta cualquier forma razonable que mande Kapso y devuelve lista de eventos."""
    # Caso 1: body es directamente una lista de eventos
    if isinstance(body, list):
        return body
    # Caso 2: body es un dict con clave "data"
    if isinstance(body, dict):
        data = body.get("data")
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        # Caso 3: body es directamente un evento (sin wrapper)
        return [body]
    return []


@app.post("/kapso")
async def kapso_webhook(request: Request):
    """
    Webhook de Kapso. Recibe eventos en formato JSON.
    El nombre del evento viene en el header X-Webhook-Event (también puede venir en body.event).
    """
    body = await request.json()
    headers = request.headers

    idempotency_key = headers.get("x-idempotency-key", "")
    if idempotency_key and idempotency_key in _seen_keys:
        logger.info("Webhook duplicado ignorado | key=%s", idempotency_key)
        return {"status": "duplicate"}

    event = headers.get("x-webhook-event") or (body.get("event", "") if isinstance(body, dict) else "")
    events_list = _extract_events(body)

    logger.info("Webhook Kapso | event=%s | items=%d | batch=%s",
                event, len(events_list), headers.get("x-webhook-batch", "false"))

    for data in events_list:
        try:
            await _process_kapso_event(event, data)
        except Exception as exc:
            logger.exception("Error procesando evento Kapso: %s", exc)

    # Marcar idempotencia DESPUÉS de procesar correctamente,
    # así si crasheamos los retries no quedan bloqueados
    if idempotency_key:
        _seen_keys.add(idempotency_key)
        if len(_seen_keys) > 5000:
            _seen_keys.clear()

    return {"status": "ok"}


async def _process_kapso_event(event: str, data: dict) -> None:
    """Procesa un evento individual de Kapso (texto, audio o imagen)."""
    if event and event != "whatsapp.message.received":
        logger.info("Evento ignorado: %s", event)
        return

    message = data.get("message") or {}
    msg_type = message.get("type")
    phone = message.get("from")
    wamid = message.get("id")
    phone_number_id = (
        data.get("phone_number_id")
        or (data.get("conversation") or {}).get("phone_number_id")
    )
    if not phone or not phone_number_id:
        logger.warning("Payload sin phone/pnid: %r %r", phone, phone_number_id)
        return

    kapso_meta = message.get("kapso") or {}

    # ── TEXTO ───────────────────────────────────────────────────────────────
    if msg_type == "text":
        text = (message.get("text") or {}).get("body", "").strip()
        if not text:
            return
        logger.info("Texto entrante | from=%s | %r", phone, text)

    # ── AUDIO / NOTA DE VOZ (Kapso ya lo transcribió) ───────────────────────
    elif msg_type in ("audio", "voice"):
        transcript = (kapso_meta.get("transcript") or {}).get("text", "").strip()
        if not transcript:
            logger.warning("Audio sin transcripción de Kapso, descartando")
            return
        text = transcript
        logger.info("Audio transcripto | from=%s | %r", phone, text)

    # ── IMAGEN (descargamos y la analizamos con Claude vision) ──────────────
    elif msg_type == "image":
        media_url = kapso_meta.get("media_url") or (kapso_meta.get("media_data") or {}).get("url")
        mime_type = (kapso_meta.get("media_data") or {}).get("content_type", "image/jpeg")
        caption = (message.get("image") or {}).get("caption", "").strip()

        if not media_url:
            logger.warning("Imagen sin media_url, descartando")
            return

        try:
            img_bytes = await _download_kapso_media(media_url)
            description = await _describe_image(img_bytes, mime_type, caption)
        except Exception as exc:
            logger.exception("Error procesando imagen: %s", exc)
            text = "[el cliente mandó una foto, no pude analizarla]"
            if caption:
                text = f"{caption} {text}"
        else:
            if caption:
                text = f"{caption}\n\n[foto adjunta: {description}]"
            else:
                text = f"[foto adjunta: {description}]"
            logger.info("Imagen analizada | from=%s | %r", phone, text[:200])

    # ── OTROS TIPOS NO SOPORTADOS ───────────────────────────────────────────
    else:
        logger.info("Tipo de mensaje no soportado: %s", msg_type)
        return

    _meta[phone] = {"phone_number_id": phone_number_id, "last_wamid": wamid}
    await _enqueue(phone, text)
