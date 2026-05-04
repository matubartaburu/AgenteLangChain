"""
parse_whatsapp_chat.py — Convierte un .txt exportado de WhatsApp en pares
{"client": ..., "valentina": ...} listos para sumar al RAG.

Formato esperado del .txt (cualquiera de estos funciona):
    [14/3/26, 14:30:00] Mateo: hola, queria saber precios
    14/3/26, 14:30 - Mateo: hola, queria saber precios

Lógica:
  1. Parsea cada línea del .txt
  2. Filtra mensajes de sistema (cifrado, multimedia omitido, etc.)
  3. Anonimiza nombres, teléfonos y emails
  4. Agrupa mensajes consecutivos del mismo remitente
  5. Arma pares (cliente → spa) cuando el spa responde después del cliente
  6. Filtra pares de baja calidad (vacíos, muy cortos, muy largos)

Uso:
    python parse_whatsapp_chat.py chat.txt --spa-name "Angela De Maria"
    python parse_whatsapp_chat.py carpeta/  --spa-name "Angela De Maria"  # batch

Output:
    chat_extracted.json (o el nombre que le pases con --output)
"""

import re
import json
import argparse
from pathlib import Path

# ── Regex para parsear líneas de WhatsApp (iOS y Android, español/inglés) ───
LINE_PATTERNS = [
    # iOS:    [14/3/26, 14:30:00] Sender: message
    re.compile(
        r"^\[(\d+/\d+/\d+),\s+(\d+:\d+(?::\d+)?(?:\s*[ap]\.?\s*m\.?)?)\]\s+([^:]+?):\s*(.*)$",
        re.IGNORECASE,
    ),
    # Android: 14/3/26, 14:30 - Sender: message
    re.compile(
        r"^(\d+/\d+/\d+),\s+(\d+:\d+(?::\d+)?(?:\s*[ap]\.?\s*m\.?)?)\s+-\s+([^:]+?):\s*(.*)$",
        re.IGNORECASE,
    ),
]

# ── Patrones para anonimizar ────────────────────────────────────────────────
PHONE_PATTERN = re.compile(r"(?:\+?\d{2,4}[\s\-]?)?\d{3,4}[\s\-]?\d{3,4}")
EMAIL_PATTERN = re.compile(r"\b[\w.\-]+@[\w.\-]+\.\w+\b")

# ── Marcadores de sistema/media a ignorar (multi-idioma) ────────────────────
SYSTEM_MARKERS = [
    "<multimedia omitido>",
    "<archivo omitido>",
    "<media omitted>",
    "<archivo adjuntado>",
    "imagen omitida",
    "audio omitido",
    "video omitido",
    "gif omitido",
    "sticker omitido",
    "sticker omitted",
    "image omitted",
    "video omitted",
    "audio omitted",
    "<this message was edited>",
    "este mensaje fue editado",
    "this message was deleted",
    "se eliminó este mensaje",
    "mensajes y llamadas están cifrados",
    "messages and calls are end-to-end encrypted",
]


def parse_line(line: str) -> dict | None:
    for pattern in LINE_PATTERNS:
        match = pattern.match(line)
        if match:
            return {
                "date":    match.group(1),
                "time":    match.group(2),
                "sender":  match.group(3).strip(),
                "message": match.group(4).strip(),
            }
    return None


def parse_chat(text: str) -> list[dict]:
    """Parsea el .txt completo a lista de mensajes (soporta multilínea)."""
    messages = []
    current = None
    for line in text.split("\n"):
        parsed = parse_line(line)
        if parsed:
            if current:
                messages.append(current)
            current = parsed
        elif current and line.strip():
            # Continuación de mensaje multilinea
            current["message"] += "\n" + line.strip()
    if current:
        messages.append(current)
    return messages


def is_system_message(msg: dict) -> bool:
    content = msg["message"].lower().strip()
    if not content:
        return True
    return any(marker in content for marker in SYSTEM_MARKERS)


def anonymize(text: str, name_replacements: dict[str, str] | None = None) -> str:
    text = PHONE_PATTERN.sub("[NRO]", text)
    text = EMAIL_PATTERN.sub("[EMAIL]", text)
    if name_replacements:
        for original, replacement in name_replacements.items():
            text = re.sub(rf"\b{re.escape(original)}\b", replacement, text, flags=re.IGNORECASE)
    return text


def group_consecutive(messages: list[dict]) -> list[dict]:
    """Une mensajes consecutivos del mismo sender en uno solo."""
    grouped = []
    for msg in messages:
        if grouped and grouped[-1]["sender"] == msg["sender"]:
            grouped[-1]["message"] += " " + msg["message"]
        else:
            grouped.append(msg.copy())
    return grouped


def extract_pairs(messages: list[dict], spa_sender: str) -> list[dict]:
    """
    Recorre los mensajes y crea pares (cliente → spa) donde el spa responde
    después del cliente. Permite múltiples turnos por chat.
    """
    pairs = []
    spa_lower = spa_sender.lower()
    for i in range(len(messages) - 1):
        a, b = messages[i], messages[i + 1]
        a_is_spa = spa_lower in a["sender"].lower()
        b_is_spa = spa_lower in b["sender"].lower()
        if not a_is_spa and b_is_spa:
            pairs.append({
                "client":    a["message"].strip(),
                "valentina": b["message"].strip(),
            })
    return pairs


def filter_quality(pairs: list[dict]) -> list[dict]:
    """Quita pares basura: muy cortos, muy largos, vacíos, repetidos."""
    seen_clients = set()
    filtered = []
    for p in pairs:
        cli = p["client"].strip()
        val = p["valentina"].strip()
        if len(cli) < 2 or len(val) < 5:
            continue
        if len(cli) > 600 or len(val) > 1200:
            continue
        key = cli.lower()
        if key in seen_clients:
            continue
        seen_clients.add(key)
        filtered.append({"client": cli, "valentina": val})
    return filtered


def process_file(path: Path, spa_name: str, name_replacements: dict) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    messages = parse_chat(text)

    raw_count = len(messages)
    messages = [m for m in messages if not is_system_message(m)]

    for m in messages:
        m["message"] = anonymize(m["message"], name_replacements)

    messages = group_consecutive(messages)
    pairs = extract_pairs(messages, spa_name)
    pairs = filter_quality(pairs)

    print(f"  📄 {path.name}: {raw_count} líneas → {len(pairs)} pares útiles")
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Parser de chats de WhatsApp para RAG")
    parser.add_argument("input", help="Archivo .txt o carpeta con varios .txt")
    parser.add_argument("--spa-name", required=True,
                        help='Nombre del spa en WhatsApp (ej: "Angela De Maria")')
    parser.add_argument("--output", default="chat_extracted.json",
                        help="Archivo JSON de salida")
    parser.add_argument("--anonymize",
                        help='Pares "Nombre=Reemplazo,Otro=Reemplazo" para borrar nombres reales')
    args = parser.parse_args()

    name_replacements = {}
    if args.anonymize:
        for pair in args.anonymize.split(","):
            k, _, v = pair.partition("=")
            name_replacements[k.strip()] = v.strip()

    p = Path(args.input)
    all_pairs = []

    if p.is_file():
        all_pairs = process_file(p, args.spa_name, name_replacements)
    elif p.is_dir():
        files = sorted(p.glob("*.txt"))
        print(f"Procesando {len(files)} archivos...")
        for f in files:
            all_pairs.extend(process_file(f, args.spa_name, name_replacements))
    else:
        print(f"⚠️ No existe: {p}")
        return

    # Dedup global
    seen = set()
    unique = []
    for pair in all_pairs:
        key = pair["client"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(pair)

    Path(args.output).write_text(
        json.dumps(unique, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n✅ {len(unique)} pares únicos guardados en {args.output}")
    print(f"   Listos para sumar a examples.json y correr load_examples.py")


if __name__ == "__main__":
    main()
