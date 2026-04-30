# Proyecto: Agente WhatsApp — Spa Ángela De María

## Qué es esto
Agente de WhatsApp llamado **Valentina**, recepcionista virtual del Spa Ángela De María.
Recibe mensajes vía Twilio, los procesa con un agente LangGraph + Claude (claude-sonnet-4-6),
y responde automáticamente por WhatsApp.

## Stack
| Capa | Tecnología |
|---|---|
| Web server | FastAPI + uvicorn |
| Agente / grafo | LangGraph |
| LLM | Claude claude-sonnet-4-6 via Anthropic SDK |
| WhatsApp | Twilio (webhook POST /webhook) |
| Historial | En memoria (dict Python) — máx 20 mensajes por número |
| Config | python-dotenv (.env) |

## Archivos clave
```
main.py      — FastAPI: POST /webhook (TwiML) + GET /health
agent.py     — Grafo LangGraph + SYSTEM_PROMPT de Valentina
memory.py    — Historial por número de teléfono (en memoria)
config.py    — Carga variables de entorno y avisa si faltan
.env.example — Plantilla de variables necesarias
requirements.txt
```

## Variables de entorno requeridas
```
ANTHROPIC_API_KEY
TWILIO_ACCOUNT_SID
TWILIO_AUTH_TOKEN
TWILIO_WHATSAPP_NUMBER   # formato: whatsapp:+XXXXXXXXXXX
```

## Cómo correr localmente
```bash
pip install -r requirements.txt
cp .env.example .env      # completar con claves reales
uvicorn main:app --reload --port 8000

# En otra terminal:
ngrok http 8000
# URL del webhook: https://<id>.ngrok.io/webhook  →  pegala en Twilio
```

## Flujo de un mensaje
1. WhatsApp usuario → Twilio → POST /webhook
2. `main.py` extrae `From` y `Body` del form data
3. `memory.py` busca/crea historial del número
4. `agent.py` invoca el grafo LangGraph con el historial completo
5. LangGraph llama a Claude con SYSTEM_PROMPT + historial
6. Respuesta se guarda en memoria y se devuelve como TwiML a Twilio
7. Twilio envía la respuesta al usuario por WhatsApp

## Pendientes / TODOs
- [ ] Completar SYSTEM_PROMPT en `agent.py` con servicios, precios y horarios reales
- [ ] Completar datos de contacto y ubicación en el system prompt
- [ ] Configurar número de WhatsApp propio en Twilio (salir del sandbox)
- [ ] (Opcional) Reemplazar historial en memoria por Redis/DB para persistencia
- [ ] (Opcional) Agregar validación de firma Twilio para producción
- [ ] (Opcional) Agregar endpoint para limpiar historial de un número

## Decisiones de diseño
- **Historial externo al grafo**: `memory.py` maneja el historial; el grafo LangGraph es
  stateless por diseño para simplificar. Cada llamada recibe el historial completo.
- **Máximo 20 mensajes**: para no exceder la ventana de contexto; se descartan los más viejos.
- **TwiML directo**: el webhook devuelve TwiML (no usa Twilio client para enviar),
  lo que simplifica el flujo y evita una llamada extra a la API de Twilio.
- **Fallback amigable**: si Claude falla, el usuario recibe un mensaje genérico en lugar de un error HTTP.
