# Proyecto: Agente WhatsApp — Spa Ángela De María

> **Nota sobre este archivo**
> Este documento es público (vive en el repo). Algunos valores fueron redactados a propósito para no exponer datos sensibles:
> - URLs reales de Supabase y ngrok → reemplazadas por placeholders (`<your-project-ref>`, `<your-ngrok-subdomain>`)
> - Números de teléfono personales y de sandbox → reemplazados por placeholders
> - IDs internos del provider de WhatsApp → reemplazados por placeholders
>
> Los valores reales viven en `.env` (gitignored) y no se filtran al repo. Si querés correr el proyecto, copiá `.env.example` a `.env` y completá con tus propias credenciales.

## Qué es esto
Agente de WhatsApp llamado **Valentina**, recepcionista virtual del Spa **Ángela De María — Belleza y Relax** (Divina Comedia 1586, Carrasco, Montevideo). Recibe mensajes vía Kapso (proxy de WhatsApp Cloud API), los procesa con un agente LangGraph + Claude, y responde con tono rioplatense informal, sonando lo más humano posible.

---

## Stack completo

| Capa | Tecnología | Notas |
|---|---|---|
| **Web server** | FastAPI + uvicorn | Puerto 8000 |
| **Agente** | LangGraph (3 nodos: classify → generate → validate) | |
| **LLM principal** | Claude Sonnet 4.6 | `claude-sonnet-4-6` — para responder + visión de imágenes |
| **LLM auxiliar** | Claude Haiku 4.5 | `claude-haiku-4-5-20251001` — clasificación + corrección de formato |
| **WhatsApp** | **Kapso** (Sandbox WhatsApp) | NO Twilio. Ver "Provider" abajo |
| **Storage** | Supabase (Postgres) | Tablas: `conversations`, `client_notes`, `examples` |
| **RAG** | pgvector + fastembed local | Modelo: `paraphrase-multilingual-mpnet-base-v2` (768 dims) |
| **Túnel local** | ngrok | URL: `https://<your-ngrok-subdomain>.ngrok-free.dev` |
| **Config** | python-dotenv | `.env` con credenciales (gitignored) |

---

## Provider de WhatsApp: Kapso (no Twilio)

- Twilio fue deprecado el 30/4. **Solo se usa Kapso ahora**.
- Sandbox WhatsApp number: `<kapso-sandbox-number>`
- Webhook URL configurado en Kapso: `https://<your-ngrok-subdomain>.ngrok-free.dev/kapso`
- Eventos suscritos: **solo `whatsapp.message.received`** (los demás generan ruido)
- Debouncing de Kapso: **DESACTIVADO** (usamos el nuestro de 8 segundos)
- Phone activado en sandbox: `<your-test-number>`
- Phone number ID Kapso: `<phone-number-id>`

API endpoints usados:
- Recibir: `POST /kapso` (webhook) — ver `main.py`
- Enviar: `POST https://api.kapso.ai/meta/whatsapp/v24.0/{phone_number_id}/messages`
- Typing: mismo endpoint con `status: read` + `typing_indicator`

---

## Archivos del proyecto

| Archivo | Función |
|---|---|
| `main.py` | FastAPI server, webhook Kapso, debouncing, splitting, delays, typing |
| `agent.py` | LangGraph (classify→generate→validate), system prompt completo, RAG injection |
| `memory.py` | Supabase queries: historial + notas de cliente |
| `rag.py` | Embeddings locales (fastembed) + búsqueda vectorial pgvector |
| `config.py` | Carga `.env`, valida vars críticas |
| `examples.json` | 64 ejemplos curados (cliente → respuesta de Valentina) |
| `examples_generated.json` | Salida del generador sintético (~1500 cuando termina) |
| `examples_generated.jsonl` | Append-only del generador (para resumir en caso de crash) |
| `load_examples.py` | Embeddea + carga ejemplos a Supabase |
| `generate_examples.py` | Genera 1500 sintéticos con Claude Sonnet (75 batches × 20) |
| `parse_whatsapp_chat.py` | Parsea export `.txt` de WhatsApp → JSON de pares cliente/respuesta |
| `explore_hf_datasets.py` | Explorador de datasets de Hugging Face (script de prueba) |
| `.env` | Credenciales (gitignored) |
| `.env.example` | Plantilla |
| `requirements.txt` | Dependencias |
| `CLAUDE.md` | Este archivo |

---

## Variables de entorno (`.env`)

```
ANTHROPIC_API_KEY=sk-ant-api03-...
KAPSO_API_KEY=...
KAPSO_WEBHOOK_SECRET=...   # para validar firmas (no implementado aún)
SUPABASE_URL=https://<your-project-ref>.supabase.co
SUPABASE_SERVICE_KEY=eyJhbGc...
OPENAI_API_KEY=sk-proj-...   # backup, hoy no se usa (embeddings son locales)
TWILIO_*=...                 # DEPRECADAS, ignorar
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886  # sandbox Twilio (no se usa)
```

---

## Supabase (proyecto: `<your-project-ref>`)

### Tablas
- `conversations(id, phone_number, role, content, created_at)` — historial de chats, máx 20 mensajes por número
- `client_notes(id, phone_number, note, created_at)` — info persistente del cliente (nombre, intereses, lesiones)
- `examples(id, client_message, valentina_response, embedding vector(768), created_at)` — RAG con few-shot examples

### Funciones SQL
- `trim_conversation(p_phone_number, p_max_messages)` — descarta mensajes viejos
- `count_active_conversations()` — cuenta números únicos
- `match_examples(query_embedding, match_count)` — top-K por cosine distance

### Extensiones
- `pgvector` (habilitada)

---

## Features implementadas (lo que YA anda)

### Mensajería
- ✅ Recibir texto, audio (transcripto por Kapso) y imágenes (descripción por Claude vision)
- ✅ Enviar mensajes vía Kapso REST API
- ✅ Typing indicator antes de cada bloque de respuesta
- ✅ Mark-as-read automático

### Procesamiento de mensajes
- ✅ **Debouncing** de 8 segundos (mensajes en ráfaga del mismo número se procesan juntos)
- ✅ **Idempotencia** con `X-Idempotency-Key` (no procesamos webhooks duplicados)
- ✅ **Soporte de batches** de Kapso

### Generación de respuestas
- ✅ LangGraph con 3 nodos: clasificación (Haiku) → generación (Sonnet con tool_use) → validación
- ✅ **RAG** con 64 ejemplos curados + búsqueda vectorial (top-3, umbral 0.55)
- ✅ **Memoria de cliente recurrente** — notas previas se inyectan al system prompt
- ✅ **Persistencia automática de notas** — Sonnet detecta info importante y la guarda en `client_notes`
- ✅ **Catálogo real** del Excel del spa (precios, servicios, equipo)

### Salida
- ✅ **Splitting inteligente** — `|||` del modelo + fallback por oraciones (~150 chars/bloque)
- ✅ **Re-split de bloques largos** — preserva la pregunta final si la hay
- ✅ **Delays naturales** — 5-9s antes del primer bloque + tiempo de "tipeo" proporcional al largo
- ✅ **Pausas entre bloques** — 1-3s + tipo
- ✅ **Serialización por número** (worker loop) — no hay respuestas paralelas que se mezclen

### Estilo
- ✅ Rioplatense informal: "vos", "podés", "dale", "ta", "buenísimo"
- ✅ Sin emojis, sin markdown, sin asteriscos
- ✅ Sin punto final en bloques (estilo WhatsApp casual)
- ✅ "Te contactamos" en lugar de "una recepcionista te contacta" (Valentina ES la recepcionista)
- ✅ Saludos canónicos para cliente nuevo, personalizados para recurrente
- ✅ **No pregunta por costumbre** — solo cuando aporta (~50/50)

---

## En proceso (corriendo cuando guardamos esto)

🟡 **Generador sintético**: corriendo en background. Última lectura: **480/1500 ejemplos** (32%). Cuando termine genera `examples_generated.json` con ~1500 conversaciones diversas (15 temas × 5 perfiles × 20 ejemplos).

Cuando termine, hay que:
1. Mergear con `examples.json` (concatenar listas, deduplicar)
2. Re-correr `python load_examples.py`
3. Reiniciar server (para que recargue embeddings)

---

## Pendiente — Próximos pasos prioritarios

### 🚨 Crítico (próxima sesión)

1. **Notificación al equipo cuando handoff** (1 h) — Cuando `handoff_needed=true`, mandar mensaje a un grupo de WhatsApp / Telegram / Slack / email del staff con número del cliente y resumen. Hoy el handoff queda solo en logs y nadie del spa se entera.

2. **Time-awareness** (30 min) — Inyectar `[Hoy es jueves 1 de mayo, 23:30, spa cerrado hasta mañana 8:00]` al system prompt. Saludos según hora del día. Manejo de mensajes fuera de horario.

3. **Fix del clasificador Haiku** (30 min) — Hoy falla siempre con "Expecting value: line 1 column 1" porque Haiku devuelve JSON con texto extra. Pasar a `tool_use` para forzar JSON estructurado.

### ⭐ Calidad

4. **Variación en frases canónicas** (30 min) — Pool de 4-5 variantes de "te contactamos para coordinar" + rotación aleatoria. Hoy se repite igual siempre.

5. **Largo de respuesta adaptativo** (30 min) — "¿están abiertos?" no debería tener 3 bloques. Regla: largo ∝ complejidad de la consulta.

6. **Reactions en vez de texto corto** (45 min) — "ok gracias" → reaccionar con 👍 (Kapso lo soporta) en lugar de mandar texto.

7. **Self-critique antes de enviar** (2 h) — Pre-validación con Haiku que evalúa si la respuesta matchea energía del cliente antes de enviar.

### 🚀 Producción

8. **Deploy a Railway/Render/Fly** (2 h) — URL estable, sin necesidad de Mac prendida + ngrok.

9. **Validación de firma del webhook Kapso** (30 min) — Verificar `X-Webhook-Signature` con HMAC-SHA256. Tenemos el secret en `.env`.

10. **Manejo de errores Kapso** (30 min) — Reintentos con backoff exponencial si la API responde 5xx.

11. **Métricas básicas** (1 h) — Endpoint `/stats` con: mensajes/día, % handoff, costo Anthropic, latencia P95.

12. **Limpieza de buffers viejos** (15 min) — Task que limpia `_buffer`, `_meta`, `_seen_keys` cada hora.

### 📚 Largo plazo

- **Multi-tenant** — Cuando consigamos cliente #2, refactor para que un solo servidor sirva múltiples negocios. Ver discusión en sesión del 1/5.
- **Chats reales del spa** — Pedirle a la dueña que exporte chats. Parser ya armado (`parse_whatsapp_chat.py`).
- **A/B testing de prompts**
- **Dashboard admin**
- **Test suite / regression tests**

---

## Configuración local para correr

```bash
# Una vez:
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Cada vez:
.venv/bin/uvicorn main:app --port 8000     # server
ngrok http 8000                             # túnel (en otra terminal)

# Verificar:
curl http://localhost:8000/health

# Cargar ejemplos al RAG (después de editar examples.json):
.venv/bin/python load_examples.py

# Generar sintéticos (1500, ~$5, ~15 min):
.venv/bin/python generate_examples.py
```

---

## Decisiones de diseño relevantes

- **Sin Twilio**: el usuario migró a Kapso. Twilio no está activo. El código de Twilio se sacó de `main.py`.
- **Embeddings locales** (no API) — ahorra costos y latencia, y evita depender de OpenAI quota.
- **Anthropic Sonnet 4.6 en vez de gpt-4o** — usuario quiere stack 100% Anthropic.
- **Splitting agresivo (max 4 bloques, ~180 chars cada uno)** — para que se sienta natural en WhatsApp, no como pared de texto.
- **Sin mencionar "una recepcionista"** — Valentina ES la recepcionista. Plural genérico ("te contactamos") implica al equipo.
- **Notas de cliente persistentes** — Sonnet puede agregar notas vía tool_use; estas se inyectan en futuras conversaciones del mismo número.
- **Memoria del agente externalizada** — el grafo es stateless; toda la persistencia vive en Supabase.
