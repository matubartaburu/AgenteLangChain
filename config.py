import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Anthropic — modelo principal (Sonnet) y auxiliar (Haiku)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Supabase — historial de conversaciones
SUPABASE_URL         = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

# Kapso — proveedor alternativo de WhatsApp (Meta Cloud API)
KAPSO_API_KEY = os.getenv("KAPSO_API_KEY", "")

# OpenAI — solo para embeddings (text-embedding-3-small) usados por el RAG
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

_required = {
    "ANTHROPIC_API_KEY":     ANTHROPIC_API_KEY,
    "SUPABASE_URL":          SUPABASE_URL,
    "SUPABASE_SERVICE_KEY":  SUPABASE_SERVICE_KEY,
}
_missing = [k for k, v in _required.items() if not v]
if _missing:
    logger.warning("Variables de entorno faltantes: %s", ", ".join(_missing))
