"""
load_examples.py — Carga los ejemplos de examples.json a la tabla `examples`
de Supabase, embeddeando cada mensaje del cliente con fastembed (modelo local).

Uso:
    python load_examples.py

Borra los ejemplos viejos antes de insertar (idempotente).
"""

import json
import logging
from fastembed import TextEmbedding
from supabase import create_client

from config import SUPABASE_URL, SUPABASE_SERVICE_KEY

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def main():
    sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    with open("examples.json", encoding="utf-8") as f:
        examples = json.load(f)
    logger.info("Cargados %d ejemplos del JSON.", len(examples))

    logger.info("Inicializando modelo %s (descarga inicial ~120MB) ...", EMBED_MODEL)
    embedder = TextEmbedding(model_name=EMBED_MODEL)

    inputs = [e["client"] for e in examples]
    logger.info("Embeddeando %d mensajes ...", len(inputs))
    embeddings = [v.tolist() for v in embedder.embed(inputs)]

    logger.info("Borrando ejemplos previos ...")
    sb.table("examples").delete().neq("id", 0).execute()

    rows = [
        {
            "client_message": ex["client"],
            "valentina_response": ex["valentina"],
            "embedding": emb,
        }
        for ex, emb in zip(examples, embeddings)
    ]
    logger.info("Insertando %d filas en Supabase ...", len(rows))
    CHUNK = 50
    for i in range(0, len(rows), CHUNK):
        sb.table("examples").insert(rows[i:i + CHUNK]).execute()
    logger.info("Listo. %d ejemplos cargados con embeddings.", len(rows))


if __name__ == "__main__":
    main()
