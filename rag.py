"""
rag.py — Retrieval Augmented Generation con pgvector + embeddings locales.

Embeddings: fastembed (ONNX runtime) con paraphrase-multilingual-MiniLM-L12-v2
(384 dims, multilingual). Corre localmente sin API key, ~120MB de pesos.

Flujo:
  1. Embeddeamos el último mensaje del cliente.
  2. Buscamos los K ejemplos más similares en `examples` de Supabase
     (cosine distance vía función SQL match_examples).
  3. Devolvemos pares (cliente, valentina) listos para inyectar como few-shot.

Tabla `examples` y función `match_examples` se crean en la migración resize_examples_384dims.
Los ejemplos se cargan corriendo `python load_examples.py`.
"""

import logging
from fastembed import TextEmbedding
from supabase import create_client, Client

from config import SUPABASE_URL, SUPABASE_SERVICE_KEY

logger = logging.getLogger(__name__)

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
DEFAULT_K = 3
SIMILARITY_THRESHOLD = 0.55  # ejemplos por debajo de esto no aportan, los descartamos

_sb: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# El modelo se descarga la primera vez (~120MB) y queda cacheado en ~/.cache/fastembed
_embedder: TextEmbedding | None = None


def _get_embedder() -> TextEmbedding:
    """Lazy-load del modelo para no bloquear el startup del servidor."""
    global _embedder
    if _embedder is None:
        logger.info("Cargando modelo de embeddings (primera vez puede tardar) ...")
        _embedder = TextEmbedding(model_name=EMBED_MODEL)
        logger.info("Modelo de embeddings listo: %s", EMBED_MODEL)
    return _embedder


def embed(text: str) -> list[float]:
    """Genera un vector de 384 dimensiones para el texto."""
    emb = _get_embedder()
    vectors = list(emb.embed([text]))
    return vectors[0].tolist()


def find_similar_examples(text: str, k: int = DEFAULT_K) -> list[dict]:
    """
    Devuelve hasta k ejemplos similares: [{client_message, valentina_response, similarity}, ...]
    Si falla cualquier paso devuelve [] para no bloquear el flujo del agente.
    """
    if not text or not text.strip():
        return []
    try:
        emb = embed(text)
        result = _sb.rpc("match_examples", {
            "query_embedding": emb,
            "match_count": k,
        }).execute()
        rows = result.data or []
        # Filtramos los que están por debajo del umbral — no aportan estilo útil
        return [r for r in rows if r.get("similarity", 0) >= SIMILARITY_THRESHOLD]
    except Exception as exc:
        logger.warning("RAG falló (sin ejemplos en este turno): %s", exc)
        return []


def format_examples_for_prompt(examples: list[dict]) -> str:
    """Convierte los ejemplos en una sección de texto pegable al system prompt."""
    if not examples:
        return ""
    lines = [
        "",
        "[EJEMPLOS DE TONO Y ESTILO — imitá exactamente este registro y formato]",
    ]
    for ex in examples:
        lines.append(f"\nCliente: {ex['client_message']}")
        lines.append(f"Valentina: {ex['valentina_response']}")
    return "\n".join(lines) + "\n"
