-- Ground-truth ledger schema (spec 0.4).
-- SQLite. This is what makes the audit verifiable rather than anecdotal:
-- every mutation applied to every engine is logged here as the EXPECTED
-- state, so Phase 3 can diff actual engine state against it.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    source_path     TEXT NOT NULL,
    first_seen_commit TEXT NOT NULL,
    last_seen_commit  TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('live', 'deleted', 'renamed')),
    renamed_to      TEXT,
    is_synthetic_pii INTEGER NOT NULL DEFAULT 0 CHECK (is_synthetic_pii IN (0, 1))
);

CREATE TABLE IF NOT EXISTS doc_versions (
    version_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id       TEXT NOT NULL REFERENCES documents (doc_id),
    commit_sha   TEXT NOT NULL,
    commit_date  TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    action       TEXT NOT NULL CHECK (action IN ('add', 'update', 'delete', 'rename'))
);
CREATE INDEX IF NOT EXISTS idx_doc_versions_doc_id ON doc_versions (doc_id);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL REFERENCES documents (doc_id),
    version_id  INTEGER NOT NULL REFERENCES doc_versions (version_id),
    ordinal     INTEGER NOT NULL,
    text_sha256 TEXT NOT NULL,
    token_count INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_version_id ON chunks (version_id);

CREATE TABLE IF NOT EXISTS index_events (
    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    engine      TEXT NOT NULL CHECK (engine IN ('pgvector', 'qdrant', 'chroma')),
    chunk_id    TEXT NOT NULL REFERENCES chunks (chunk_id),
    op          TEXT NOT NULL CHECK (op IN ('upsert', 'delete', 'skip')),
    applied_at  TEXT NOT NULL,
    commit_sha  TEXT NOT NULL,
    expected_present INTEGER NOT NULL CHECK (expected_present IN (0, 1))
);
CREATE INDEX IF NOT EXISTS idx_index_events_chunk_id ON index_events (chunk_id);
CREATE INDEX IF NOT EXISTS idx_index_events_engine ON index_events (engine);

CREATE TABLE IF NOT EXISTS erasure_requests (
    req_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id       TEXT NOT NULL REFERENCES documents (doc_id),
    requested_at_commit TEXT NOT NULL,
    engine       TEXT NOT NULL CHECK (engine IN ('pgvector', 'qdrant', 'chroma')),
    expected_absent INTEGER NOT NULL CHECK (expected_absent IN (0, 1))
);
CREATE INDEX IF NOT EXISTS idx_erasure_requests_doc_id ON erasure_requests (doc_id);
