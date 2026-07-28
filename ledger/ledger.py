"""Ground-truth ledger: init happens during setup; CRUD/query helpers used across ingestion, churn replay, and audit land here."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def init_db(path: str | Path) -> sqlite3.Connection:
    """Create (or reuse) the ledger SQLite database from schema.sql. Idempotent."""
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA_PATH.read_text())
    conn.commit()
    return conn


def get_conn(path: str | Path) -> sqlite3.Connection:
    """Open a connection to an already-initialized ledger DB, with FKs enforced."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def insert_document(
    conn: sqlite3.Connection,
    doc_id: str,
    source_path: str,
    first_seen_commit: str,
    last_seen_commit: str,
    status: str = "live",
    renamed_to: str | None = None,
    is_synthetic_pii: int = 0,
) -> None:
    conn.execute(
        """
        INSERT INTO documents
            (doc_id, source_path, first_seen_commit, last_seen_commit, status, renamed_to, is_synthetic_pii)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(doc_id) DO UPDATE SET
            source_path=excluded.source_path,
            last_seen_commit=excluded.last_seen_commit,
            status=excluded.status,
            renamed_to=excluded.renamed_to,
            is_synthetic_pii=excluded.is_synthetic_pii
        """,
        (doc_id, source_path, first_seen_commit, last_seen_commit, status, renamed_to, is_synthetic_pii),
    )


def insert_doc_version(
    conn: sqlite3.Connection,
    doc_id: str,
    commit_sha: str,
    commit_date: str,
    content_sha256: str,
    action: str,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO doc_versions (doc_id, commit_sha, commit_date, content_sha256, action)
        VALUES (?, ?, ?, ?, ?)
        """,
        (doc_id, commit_sha, commit_date, content_sha256, action),
    )
    return cur.lastrowid


def get_or_create_doc_version(
    conn: sqlite3.Connection,
    doc_id: str,
    commit_sha: str,
    commit_date: str,
    content_sha256: str,
    action: str,
) -> int:
    """Idempotent version of insert_doc_version: reuses the existing version_id
    for identical (doc_id, commit_sha, content_sha256, action) rather than
    inserting a duplicate. This keeps chunk_id (which embeds version_id)
    stable across re-runs of the same pipeline invocation, so an interrupted
    embedding pass can resume against a still-valid checkpoint cache."""
    cur = conn.execute(
        """
        SELECT version_id FROM doc_versions
        WHERE doc_id = ? AND commit_sha = ? AND content_sha256 = ? AND action = ?
        """,
        (doc_id, commit_sha, content_sha256, action),
    )
    row = cur.fetchone()
    if row:
        return row[0]
    return insert_doc_version(conn, doc_id, commit_sha, commit_date, content_sha256, action)


def insert_chunks(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    """rows: list of (chunk_id, doc_id, version_id, ordinal, text_sha256, token_count).
    Idempotent: re-inserting an already-present chunk_id (same PK) is a no-op."""
    conn.executemany(
        """
        INSERT OR IGNORE INTO chunks (chunk_id, doc_id, version_id, ordinal, text_sha256, token_count)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def insert_index_events(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    """rows: list of (engine, chunk_id, op, applied_at, commit_sha, expected_present)."""
    conn.executemany(
        """
        INSERT INTO index_events (engine, chunk_id, op, applied_at, commit_sha, expected_present)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def existing_index_event_chunk_ids(conn: sqlite3.Connection, engine: str, op: str = "upsert") -> set[str]:
    """Chunk ids that already have an (engine, op) event logged - used to avoid
    duplicate index_events rows when a run is resumed after an interruption."""
    cur = conn.execute(
        "SELECT chunk_id FROM index_events WHERE engine = ? AND op = ?",
        (engine, op),
    )
    return {row[0] for row in cur.fetchall()}


def latest_version_id(conn: sqlite3.Connection, doc_id: str) -> int | None:
    """The most recently minted version_id for doc_id, or None if it has never
    been versioned. version_id is AUTOINCREMENT, so MAX() is "current"."""
    cur = conn.execute("SELECT MAX(version_id) FROM doc_versions WHERE doc_id = ?", (doc_id,))
    row = cur.fetchone()
    return row[0] if row and row[0] is not None else None


def chunk_ids_for_version(conn: sqlite3.Connection, version_id: int) -> list[str]:
    cur = conn.execute(
        "SELECT chunk_id FROM chunks WHERE version_id = ? ORDER BY ordinal", (version_id,)
    )
    return [row[0] for row in cur.fetchall()]


def chunk_ids_for_doc(conn: sqlite3.Connection, doc_id: str) -> list[str]:
    """All chunks ever created for doc_id, across every version - used only by
    the PII-overlay retrofit to find every remnant of the legacy
    single-doc Team Directory before purging it. Not for general use during
    churn replay, which should scope to a specific version via
    chunk_ids_for_version instead."""
    cur = conn.execute("SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,))
    return [row[0] for row in cur.fetchall()]


def get_document_status(conn: sqlite3.Connection, doc_id: str) -> str | None:
    cur = conn.execute("SELECT status FROM documents WHERE doc_id = ?", (doc_id,))
    row = cur.fetchone()
    return row[0] if row else None


def purge_document(conn: sqlite3.Connection, doc_id: str) -> None:
    """FK-safe full delete of a doc_id's entire history (index_events -> chunks
    -> doc_versions -> documents). Only for the PII-overlay retrofit
    - never use this to model a real git delete or a PII erasure request,
    both of which must go through insert_document's status transition instead
    so the ledger keeps a truthful history of what happened and when."""
    conn.execute(
        "DELETE FROM index_events WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE doc_id = ?)",
        (doc_id,),
    )
    conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
    conn.execute("DELETE FROM doc_versions WHERE doc_id = ?", (doc_id,))
    conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))


def insert_erasure_request(
    conn: sqlite3.Connection, doc_id: str, requested_at_commit: str, engine: str, expected_absent: int
) -> None:
    conn.execute(
        """
        INSERT INTO erasure_requests (doc_id, requested_at_commit, engine, expected_absent)
        VALUES (?, ?, ?, ?)
        """,
        (doc_id, requested_at_commit, engine, expected_absent),
    )


def existing_erasure_request_doc_ids(conn: sqlite3.Connection, engine: str) -> set[str]:
    """Doc ids that already have an erasure_requests row logged for engine -
    mirrors existing_index_event_chunk_ids's resumability pattern."""
    cur = conn.execute("SELECT DISTINCT doc_id FROM erasure_requests WHERE engine = ?", (engine,))
    return {row[0] for row in cur.fetchall()}


# --- Bulk query helpers for downstream check tooling -----------------------
#
# index_events is append-only: a chunk_id can accumulate multiple rows for
# one engine over the T0->HEAD walk (e.g. upsert, then later a delete/skip).
# expected_present on any individual row reflects ledger-intent *at that
# write time*, not the current truth -- so any consumer that needs "what
# does the ledger currently expect for this chunk" must resolve to the
# max-event_id row per (engine, chunk_id), never trust an arbitrary row.
#
# These are deliberately bulk/set-based (not per-chunk_id loops) -- the
# ledger has 287,346 chunk rows and 505,506 index_events rows per engine;
# N+1 queries at that scale would be the actual bottleneck.


def latest_index_events(conn: sqlite3.Connection, engine: str) -> dict[str, dict]:
    """Final (op, expected_present) per chunk_id for one engine, resolved via
    max(event_id) per chunk_id. Ground truth for "what should this chunk's
    presence be in this engine right now, per the ledger.\""""
    cur = conn.execute(
        """
        SELECT ie.chunk_id, ie.op, ie.expected_present, ie.commit_sha, ie.applied_at
        FROM index_events ie
        JOIN (
            SELECT chunk_id, MAX(event_id) AS max_event_id
            FROM index_events
            WHERE engine = ?
            GROUP BY chunk_id
        ) latest ON ie.chunk_id = latest.chunk_id AND ie.event_id = latest.max_event_id
        WHERE ie.engine = ?
        """,
        (engine, engine),
    )
    return {
        chunk_id: {"op": op, "expected_present": expected_present, "commit_sha": commit_sha, "applied_at": applied_at}
        for chunk_id, op, expected_present, commit_sha, applied_at in cur.fetchall()
    }


def chunk_lineage(conn: sqlite3.Connection) -> dict[str, dict]:
    """Per-chunk join of doc/version context, used by downstream staleness/
    orphan-style checks. last_chunked_version_id is MAX(version_id) grouped
    from `chunks` (not `doc_versions`) deliberately: a deleted/renamed doc's
    true latest doc_versions row (action='delete'/'rename') has zero chunks,
    so grouping by doc_versions would find a version with no content.
    Grouping by chunks.doc_id instead finds "the last version that actually
    had chunks" - the correct reference point for target-set construction
    on deleted/renamed docs."""
    cur = conn.execute(
        """
        SELECT
          c.chunk_id, c.doc_id, c.version_id AS chunk_version_id,
          dv.commit_date AS chunk_version_commit_date,
          d.status AS doc_status, d.renamed_to, d.last_seen_commit,
          lc.last_chunked_version_id, lc_dv.commit_date AS last_chunked_version_commit_date
        FROM chunks c
        JOIN doc_versions dv ON dv.version_id = c.version_id
        JOIN documents d ON d.doc_id = c.doc_id
        JOIN (
            SELECT doc_id, MAX(version_id) AS last_chunked_version_id
            FROM chunks GROUP BY doc_id
        ) lc ON lc.doc_id = c.doc_id
        JOIN doc_versions lc_dv ON lc_dv.version_id = lc.last_chunked_version_id
        """
    )
    return {
        chunk_id: {
            "doc_id": doc_id,
            "chunk_version_id": chunk_version_id,
            "chunk_version_commit_date": chunk_version_commit_date,
            "doc_status": doc_status,
            "renamed_to": renamed_to,
            "last_seen_commit": last_seen_commit,
            "last_chunked_version_id": last_chunked_version_id,
            "last_chunked_version_commit_date": last_chunked_version_commit_date,
        }
        for (
            chunk_id, doc_id, chunk_version_id, chunk_version_commit_date,
            doc_status, renamed_to, last_seen_commit,
            last_chunked_version_id, last_chunked_version_commit_date,
        ) in cur.fetchall()
    }


def all_chunk_text_sha256(conn: sqlite3.Connection) -> dict[str, str]:
    """Full scan chunk_id -> text_sha256 for every chunk ever logged (287,346
    rows). Deliberately not a per-engine `WHERE chunk_id IN (...)` query -
    SQLite's ~999-parameter limit rules that out at a 166,947-chunk-per-engine
    scale anyway; load once and filter by Python set-membership instead."""
    cur = conn.execute("SELECT chunk_id, text_sha256 FROM chunks")
    return dict(cur.fetchall())


def last_chunked_version_id_for_doc(conn: sqlite3.Connection, doc_id: str) -> int | None:
    """Single-doc counterpart of chunk_lineage's last_chunked_version_id --
    for small (~40-doc) target-set builds where a per-doc lookup is fine at
    that scale."""
    cur = conn.execute("SELECT MAX(version_id) FROM chunks WHERE doc_id = ?", (doc_id,))
    row = cur.fetchone()
    return row[0] if row and row[0] is not None else None


def mode_b_deleted_doc_targets(conn: sqlite3.Connection) -> list[dict]:
    """Real git-deleted, non-synthetic-PII docs and their last-chunked
    version - an "async-deleted docs" target population for downstream
    deletion/retrievability checks. is_synthetic_pii=0 is deliberate: PII
    docs are also status='deleted' once erased, but they're the
    erasure_requests-driven target set instead (see
    erasure_requests_for_engine) - excluding them here keeps the two
    target populations non-overlapping so a consumer never double-counts a
    doc under both categories."""
    cur = conn.execute(
        """
        SELECT d.doc_id, d.last_seen_commit, lc.last_chunked_version_id
        FROM documents d
        JOIN (
            SELECT doc_id, MAX(version_id) AS last_chunked_version_id
            FROM chunks GROUP BY doc_id
        ) lc ON lc.doc_id = d.doc_id
        WHERE d.status = 'deleted' AND d.is_synthetic_pii = 0
        """
    )
    return [
        {"doc_id": doc_id, "last_seen_commit": last_seen_commit, "last_chunked_version_id": last_chunked_version_id}
        for doc_id, last_seen_commit, last_chunked_version_id in cur.fetchall()
    ]


def erasure_requests_for_engine(conn: sqlite3.Connection, engine: str) -> list[tuple[str, str]]:
    """(doc_id, requested_at_commit) for every expected_absent=1 erasure
    request logged for engine - row-returning sibling of
    existing_erasure_request_doc_ids, which only returns a bare doc_id set."""
    cur = conn.execute(
        "SELECT doc_id, requested_at_commit FROM erasure_requests WHERE engine = ? AND expected_absent = 1",
        (engine,),
    )
    return cur.fetchall()


if __name__ == "__main__":
    init_db(Path(__file__).parent / "ledger.db")
