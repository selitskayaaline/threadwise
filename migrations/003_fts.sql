ALTER TABLE messages
    ADD COLUMN IF NOT EXISTS tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('russian', text)) STORED;

CREATE INDEX IF NOT EXISTS messages_tsv_idx ON messages USING GIN (tsv);
