"""
seed_conversations.py — Genera conversaciones sintéticas y las inserta en Supabase

Uso:
    python seed_conversations.py

Genera 50 conversaciones entre clientes y Valentina (recepcionista de Ángela De María),
con variedad de escenarios: consultas de precios, reservas, reclamos, despedidas,
mensajes mal escritos, audios transcriptos, etc. Tono rioplatense uruguayo.

Las inserta en la tabla `conversations` de Supabase usando números de teléfono
ficticios (+59899000001 a +59899000050).
"""

import json
import time
import logging
from anthropic import Anthropic
from supabase import create_client

from config import ANTHROPIC_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

client   = Anthropic(api_key=ANTHROPIC_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ─────────────────────────────────────────────────────────────────────────────

GENERATION_PROMPT = """
Generá {count} conversaciones de WhatsApp entre un cliente y Valentina, recepcionista virtual de Ángela De María — una clínica de estética y bienestar en Carrasco, Montevideo, Uruguay.

CONTEXTO DEL SPA:
- Servicios: masajes (maderoterapia $1.890, magnetoterapia $2.200, bioenergético $2.800/$3.300, reflexología $1.890), planes corporales, aparatología, faciales, peluquería
- Cuponera 10 masajes: $12.500 con peluquería incluida
- Equipo: Sofía, Carmen, Silvana, Rosalvis, Analía, Daniela, Gastón, Vicky
- Horario: lunes a sábado 8:00 a 20:00
- Valentina habla en rioplatense uruguayo: vos, podés, dale, tranqui, bárbaro, ta

TIPOS DE CONVERSACIONES A VARIAR (distribuir entre las {count}):
- Consulta de precio de un servicio específico
- Pregunta sobre qué masaje es mejor para un problema (espalda, celulitis, estrés)
- Cliente que quiere reservar turno (Valentina deriva a recepcionista humana)
- Consulta de horarios o ubicación
- Reclamo o queja (turno cancelado, demora)
- Despedida rápida o mensaje de agradecimiento
- Mensaje mal escrito o con faltas de ortografía (realista)
- Audio transcripto (el cliente mandó un audio, llegó como texto con muletillas: "ehh", "o sea", "tipo")
- Consulta sobre la cuponera
- Cliente indeciso que pregunta varias cosas
- Pregunta sobre un profesional específico
- Consulta de peluquería

REGLAS PARA VALENTINA:
- Responde en rioplatense uruguayo, texto plano, sin emojis ni markdown
- Usa ||| para separar bloques (máximo 3 por respuesta)
- Si el cliente quiere reservar: "Perfecto, en un momento te contacta una de nuestras recepcionistas para coordinar." y para de responder
- Nunca inventa precios que no sabe

Respondé ÚNICAMENTE con un JSON válido con esta estructura exacta, sin markdown ni explicaciones:
{{
  "conversations": [
    {{
      "scenario": "descripción corta del escenario",
      "messages": [
        {{"role": "user", "content": "mensaje del cliente"}},
        {{"role": "assistant", "content": "respuesta de Valentina"}},
        {{"role": "user", "content": "..."}},
        {{"role": "assistant", "content": "..."}}
      ]
    }}
  ]
}}

Cada conversación debe tener entre 2 y 8 mensajes. Generá exactamente {count} conversaciones.
"""

# ─────────────────────────────────────────────────────────────────────────────

def generate_batch(count: int, batch_num: int) -> list[dict]:
    """Genera un lote de conversaciones con Claude."""
    logger.info("Generando lote %d (%d conversaciones)...", batch_num, count)

    result = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8000,
        messages=[{
            "role": "user",
            "content": GENERATION_PROMPT.format(count=count)
        }]
    )

    text = result.content[0].text.strip()

    # Extraer JSON (a veces Claude agrega texto antes o después)
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No se encontró JSON en la respuesta")

    data = json.loads(text[start:end])
    return data.get("conversations", [])


def insert_conversation(phone: str, messages: list[dict]) -> int:
    """Inserta una conversación completa en Supabase. Retorna cantidad de mensajes insertados."""
    rows = [
        {"phone_number": phone, "role": msg["role"], "content": msg["content"]}
        for msg in messages
    ]
    supabase.table("conversations").insert(rows).execute()
    return len(rows)


def clear_seed_data():
    """Elimina las conversaciones de prueba (números +59899000001 a +59899000050)."""
    for i in range(1, 51):
        phone = f"+59899{i:06d}"
        supabase.table("conversations").delete().eq("phone_number", phone).execute()
    logger.info("Datos de prueba eliminados.")


def main():
    logger.info("=== Iniciando seed de conversaciones ===")

    all_conversations = []

    # Generamos en 2 lotes de 25 para no exceder tokens
    for i, count in enumerate([25, 25], start=1):
        try:
            batch = generate_batch(count, i)
            all_conversations.extend(batch)
            logger.info("Lote %d generado: %d conversaciones", i, len(batch))
        except Exception as e:
            logger.error("Error en lote %d: %s", i, e)

        if i < 2:
            time.sleep(2)  # pausa entre lotes

    logger.info("Total generado: %d conversaciones", len(all_conversations))

    # Insertar en Supabase
    total_messages = 0
    for idx, conv in enumerate(all_conversations, start=1):
        phone    = f"+59899{idx:06d}"
        messages = conv.get("messages", [])
        scenario = conv.get("scenario", "sin descripción")

        if not messages:
            continue

        try:
            count = insert_conversation(phone, messages)
            total_messages += count
            logger.info(
                "[%02d/%02d] %s | phone=%s | %d mensajes",
                idx, len(all_conversations), scenario, phone, count
            )
        except Exception as e:
            logger.error("Error insertando conversación %d: %s", idx, e)

    logger.info("=== Seed completado ===")
    logger.info("Conversaciones: %d | Mensajes totales: %d", len(all_conversations), total_messages)
    logger.info("Números: +59899000001 a +59899%06d", len(all_conversations))
    logger.info("Para limpiar los datos de prueba: llamá a clear_seed_data()")


if __name__ == "__main__":
    main()
