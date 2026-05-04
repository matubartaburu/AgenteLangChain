-- ============================================================
-- supabase_setup.sql — Ejecutá esto en Supabase > SQL Editor
-- ============================================================

-- 1. Tabla de conversaciones (historial por número)
CREATE TABLE IF NOT EXISTS conversations (
    id           BIGSERIAL PRIMARY KEY,
    phone_number TEXT      NOT NULL,
    role         TEXT      NOT NULL CHECK (role IN ('user', 'assistant')),
    content      TEXT      NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_conversations_phone_created
    ON conversations (phone_number, created_at);

-- 2. Tabla de notas de cliente (memoria persistente entre conversaciones)
CREATE TABLE IF NOT EXISTS client_notes (
    id           BIGSERIAL PRIMARY KEY,
    phone_number TEXT      NOT NULL,
    note         TEXT      NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_client_notes_phone
    ON client_notes (phone_number);

-- 3. Función que elimina mensajes viejos cuando se supera el límite
CREATE OR REPLACE FUNCTION trim_conversation(
    p_phone_number TEXT,
    p_max_messages INT
)
RETURNS VOID
LANGUAGE plpgsql
AS $$
DECLARE
    v_count INT;
    v_to_delete INT;
BEGIN
    SELECT COUNT(*) INTO v_count
    FROM conversations
    WHERE phone_number = p_phone_number;

    v_to_delete := v_count - p_max_messages;

    IF v_to_delete > 0 THEN
        DELETE FROM conversations
        WHERE id IN (
            SELECT id FROM conversations
            WHERE phone_number = p_phone_number
            ORDER BY created_at ASC
            LIMIT v_to_delete
        );
    END IF;
END;
$$;

-- 4. Función para contar conversaciones activas de forma eficiente
CREATE OR REPLACE FUNCTION count_active_conversations()
RETURNS INT
LANGUAGE sql
AS $$
    SELECT COUNT(DISTINCT phone_number)::INT FROM conversations;
$$;
