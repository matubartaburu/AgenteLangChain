"""
generate_examples.py — Genera ejemplos sintéticos de conversaciones cliente-Valentina
usando Claude Sonnet 4.6, alimentado por el catálogo real del spa y los ejemplos seed.

Estrategia:
  - 15 temas × 5 perfiles de cliente = 75 batches
  - Cada batch genera 20 ejemplos
  - Total objetivo: ~1.500 conversaciones diversas
  - Prompt caching activado para abaratar costo (~50% menos)
  - Guardado incremental por si se corta a mitad

Uso:
    python generate_examples.py
    python generate_examples.py --resume   # retoma si ya hay parciales

Output:
    examples_generated.jsonl  (uno por línea, append-only)
    examples_generated.json   (consolidado y deduplicado al terminar)
"""

import os
import sys
import json
import time
import logging
from anthropic import Anthropic

from config import ANTHROPIC_API_KEY
from agent import SYSTEM_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

client = Anthropic(api_key=ANTHROPIC_API_KEY)
MODEL = "claude-sonnet-4-6"
EXAMPLES_PER_BATCH = 20

JSONL_FILE = "examples_generated.jsonl"
FINAL_FILE = "examples_generated.json"


# ── Tool para forzar JSON estructurado ──────────────────────────────────────
EXAMPLES_TOOL = {
    "name": "submit_examples",
    "description": "Submit a list of client-Valentina conversation examples",
    "input_schema": {
        "type": "object",
        "properties": {
            "examples": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "client": {
                            "type": "string",
                            "description": "Mensaje exacto que el cliente escribiría en WhatsApp",
                        },
                        "valentina": {
                            "type": "string",
                            "description": "Respuesta de Valentina con bloques separados por ||| (max 3 bloques)",
                        },
                    },
                    "required": ["client", "valentina"],
                },
            }
        },
        "required": ["examples"],
    },
}


# ── Temas (15) ──────────────────────────────────────────────────────────────
THEMES = [
    "saludos iniciales: cliente arranca la conversación (formal, informal, dudoso, audio transcripto, hola hola)",
    "precios de masajes: cliente pregunta precio de algún masaje específico o pide ver opciones generales",
    "recomendaciones para dolor o contracturas musculares (espalda, cuello, lumbar, lesiones)",
    "recomendaciones para celulitis, grasa localizada, retención de líquidos, piernas pesadas",
    "recomendaciones para piel: manchas, arrugas, acné, fotoenvejecimiento, flacidez",
    "consulta sobre tratamientos faciales: qué hace cada uno, precios, duración",
    "depilación definitiva: zonas, precios, promo 50% off, cuponera x6",
    "peluquería: cortes, alisados, balayage, babylights, color, Olaplex",
    "cuponeras y planes: cuáles hay, cuánto salen, qué incluyen, si son compartibles",
    "reservas: cliente quiere agendar turno (con o sin profesional preferido)",
    "cancelaciones, reagendamientos o cambios de turno existente",
    "consultas logísticas: ubicación, horarios, métodos de pago, estacionamiento, wifi",
    "despedidas y cierres: gracias, lo pienso, te aviso, chau, lo charlo con mi marido",
    "edge cases: cliente confundido, mal escrito, urgencia, fuera de horario, errores",
    "preguntas sobre el equipo: con quién hago tal cosa, especialidad de cada profesional",
]

# ── Perfiles de cliente (5) ─────────────────────────────────────────────────
PERSONAS = [
    "mujer joven (20-35 años), tono informal, usa 'dale', 'ta', 'jajaja', a veces sin signos de puntuación, mensajes cortos",
    "mujer madura (50+ años), tono más formal y cálido, oraciones correctas gramaticalmente, mensajes algo más largos",
    "mamá apurada con poco tiempo, mensajes cortos y directos, va al grano, a veces escribe rápido con typos",
    "hombre que busca información concreta, mensajes cortos sin adornos, puede ser corte o consulta puntual",
    "cliente nueva, hace muchas preguntas, dudosa, tono curioso, quiere que le expliquen bien antes de decidir",
]


def build_prompt(theme: str, persona: str, n: int) -> str:
    return f"""Estás generando datos de entrenamiento para Valentina, recepcionista virtual del spa Ángela De María en Carrasco, Montevideo.

PERFIL DEL CLIENTE en estos ejemplos:
{persona}

TEMA de los ejemplos:
{theme}

INSTRUCCIONES CRÍTICAS:
- Generá EXACTAMENTE {n} pares cliente/Valentina, todos DIFERENTES entre sí
- Cliente: escribe como en WhatsApp real (puede tener typos leves, abreviaciones, audio transcripto con errores menores)
- Valentina: respeta ESTRICTAMENTE las reglas del system prompt (rioplatense, sin emojis, sin punto final en bloques, ||| para separar bloques, max 3 bloques)
- Variá MUCHO las frases — NO repitas frases de Valentina ni del cliente entre los {n} ejemplos
- Para reservas, Valentina dice EXACTAMENTE "Perfecto, en un momento te contactamos para coordinar" y nada más (sin |||)
- Si Valentina no sabe un precio, dice que se confirma o que pueden llamar al 2600 5557
- Usá SOLO precios y servicios reales del catálogo del system prompt
- Las despedidas cortas no llevan |||
- Los signos ¿ y ! sí se usan, pero los puntos al final de un bloque NO

Devolvé la lista de {n} ejemplos vía la tool submit_examples."""


def generate_batch(theme: str, persona: str, n: int = EXAMPLES_PER_BATCH) -> list[dict]:
    """Hace un call a Sonnet para generar n ejemplos de (theme, persona)."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        # System prompt con cache_control para abaratar calls subsiguientes
        system=[
            {
                "type": "text",
                "text": (
                    "Sos un generador experto de datos sintéticos de conversaciones de WhatsApp.\n"
                    "Tu salida debe respetar al pie de la letra las reglas de Valentina, definidas a continuación:\n\n"
                    + SYSTEM_PROMPT
                ),
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": build_prompt(theme, persona, n)}],
        tools=[EXAMPLES_TOOL],
        tool_choice={"type": "tool", "name": "submit_examples"},
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_examples":
            return block.input.get("examples", [])
    return []


def already_done_combos() -> set[tuple[str, str]]:
    """Si existe el .jsonl, lee qué combinaciones ya están listas para retomar."""
    if not os.path.exists(JSONL_FILE):
        return set()
    done = set()
    with open(JSONL_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                done.add((rec.get("theme", ""), rec.get("persona", "")))
            except Exception:
                continue
    return done


def main(resume: bool = False):
    done = already_done_combos() if resume else set()
    if done:
        logger.info("Modo resume: %d combinaciones ya hechas, las salteo", len(done))

    total_combos = len(THEMES) * len(PERSONAS)
    counter = 0
    total_examples = 0
    cost_estimate = 0.0
    t0 = time.time()

    # Si no es resume, limpiamos archivos previos
    if not resume:
        for fp in (JSONL_FILE, FINAL_FILE):
            if os.path.exists(fp):
                os.remove(fp)

    for theme_idx, theme in enumerate(THEMES):
        for persona_idx, persona in enumerate(PERSONAS):
            counter += 1
            if (theme, persona) in done:
                logger.info("[%d/%d] (skipped, ya hecho)", counter, total_combos)
                continue

            short_theme = theme.split(":")[0]
            short_persona = persona.split(",")[0]
            logger.info("[%d/%d] Tema=%r | Perfil=%r", counter, total_combos, short_theme, short_persona)

            try:
                batch = generate_batch(theme, persona, EXAMPLES_PER_BATCH)
                # Append al jsonl con metadata
                with open(JSONL_FILE, "a", encoding="utf-8") as f:
                    for ex in batch:
                        f.write(json.dumps({"theme": theme, "persona": persona, **ex}, ensure_ascii=False) + "\n")
                total_examples += len(batch)
                logger.info("  → %d ejemplos generados (acumulado: %d)", len(batch), total_examples)
                # Pequeña pausa para no abusar del rate limit
                time.sleep(0.5)
            except Exception as exc:
                logger.error("  ⚠️  Error en este batch: %s", exc)
                # Espera más larga si hay problema, después seguimos
                time.sleep(5)

    # Consolidado: leemos el jsonl, deduplicamos, escribimos JSON final
    logger.info("Consolidando y deduplicando...")
    seen_clients = set()
    unique_examples = []
    with open(JSONL_FILE, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            key = rec["client"].lower().strip()
            if key in seen_clients:
                continue
            seen_clients.add(key)
            unique_examples.append({"client": rec["client"], "valentina": rec["valentina"]})

    with open(FINAL_FILE, "w", encoding="utf-8") as f:
        json.dump(unique_examples, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("Total generados: %d | Únicos tras dedup: %d", total_examples, len(unique_examples))
    logger.info("Tiempo total: %.1f minutos", elapsed / 60)
    logger.info("Output guardado en %s", FINAL_FILE)


if __name__ == "__main__":
    resume = "--resume" in sys.argv
    main(resume=resume)
