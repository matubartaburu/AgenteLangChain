"""
Servidor FastAPI — Agente WhatsApp Valentina · Spa Ángela De María

DESARROLLO LOCAL CON NGROK:
  1. Instalá ngrok: https://ngrok.com/download
  2. Corré el servidor:
       uvicorn main:app --reload --port 8000
  3. En otra terminal, exponé el puerto:
       ngrok http 8000
  4. Copiá la URL pública de ngrok (ej: https://abc123.ngrok.io)
  5. En la consola de Twilio (Messaging > WhatsApp Sandbox o tu número):
       Webhook URL: https://abc123.ngrok.io/webhook
       Método: HTTP POST
  6. Enviá un mensaje de WhatsApp al número de Twilio para probar.
"""

import logging
from fastapi import FastAPI, Form, Response, Request
from fastapi.responses import PlainTextResponse
from twilio.twiml.messaging_response import MessagingResponse

from memory import get_history, add_message, active_conversations
from agent import run_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

FALLBACK_MESSAGE = (
    "Lo siento, en este momento tengo un inconveniente técnico. 🌸 "
    "Por favor, intentá de nuevo en unos minutos o llamanos directamente al spa."
)

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
    }


@app.post("/webhook")
async def webhook(
    request: Request,
    From: str = Form(...),
    Body: str = Form(...),
):
    """
    Twilio envía un POST a este endpoint cuando llega un mensaje de WhatsApp.
    Devuelve TwiML válido con la respuesta de Valentina.

    NOTA DE SEGURIDAD: en producción validá la firma de Twilio para evitar
    requests maliciosos. Ver: https://www.twilio.com/docs/usage/webhooks/webhooks-security
    """
    phone_number = From.strip()
    user_text = Body.strip()

    logger.info("Mensaje entrante | from=%s | text=%r", phone_number, user_text)

    # 1. Agregá el mensaje del usuario al historial
    add_message(phone_number, "user", user_text)

    # 2. Obtené el historial actualizado
    history = get_history(phone_number)

    # 3. Llamá al agente
    try:
        reply = run_agent(history)
    except Exception as exc:
        logger.exception("Error al llamar al agente | from=%s | error=%s", phone_number, exc)
        reply = FALLBACK_MESSAGE

    # 4. Guardá la respuesta en el historial
    add_message(phone_number, "assistant", reply)

    logger.info("Respuesta enviada | to=%s | text=%r", phone_number, reply)

    # 5. Devolvé TwiML válido para Twilio
    twiml = MessagingResponse()
    twiml.message(reply)
    return Response(content=str(twiml), media_type="application/xml")
