"""pgvector adapter: HNSW upsert/delete, plus pg_stat_all_tables dead/live tuple snapshots."""

from __future__ import annotations

import psycopg

DEFAULT_DSN = "postgresql://decaydb:decaydb@localhost:5432/decaydb"
EMBEDDING_DIM = 1024
TABLE_NAME = "chunk_vectors"


def get_conn(dsn: str = DEFAULT_DSN) -> psycopg.Connection:
    return psycopg.connect(dsn, autocommit=True)


def ensure_schema(conn: psycopg.Connection) -> None:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            chunk_id TEXT PRIMARY KEY,
            embedding vector({EMBEDDING_DIM})
        );
        """
    )
    # HNSW with pgvector defaults (m=16, ef_construction=64) - defaults, not
    # tuned, per the fairness protocol (label defaults-vs-tuned).
    conn.execute(
        f"""
        CREATE INDEX IF NOT EXISTS {TABLE_NAME}_embedding_hnsw_idx
        ON {TABLE_NAME} USING hnsw (embedding vector_cosine_ops);
        """
    )


def _to_vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


def upsert(conn: psycopg.Connection, chunk_id_vector_pairs: list[tuple[str, list[float]]]) -> None:
    """Sorted by chunk_id, single connection, batched - deterministic HNSW build order
    per seeds.py's documented mitigation for HNSW build-order nondeterminism."""
    pairs = sorted(chunk_id_vector_pairs, key=lambda p: p[0])
    with conn.cursor() as cur:
        cur.executemany(
            f"""
            INSERT INTO {TABLE_NAME} (chunk_id, embedding) VALUES (%s, %s::vector)
            ON CONFLICT (chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding
            """,
            [(chunk_id, _to_vector_literal(vec)) for chunk_id, vec in pairs],
        )


def delete_by_chunk_id(conn: psycopg.Connection, chunk_id: str) -> None:
    conn.execute(f"DELETE FROM {TABLE_NAME} WHERE chunk_id = %s", (chunk_id,))


def get_by_chunk_id(conn: psycopg.Connection, chunk_id: str):
    with conn.cursor() as cur:
        cur.execute(f"SELECT chunk_id, embedding FROM {TABLE_NAME} WHERE chunk_id = %s", (chunk_id,))
        return cur.fetchone()


def list_chunk_ids(conn: psycopg.Connection) -> list[str]:
    """Bulk enumeration of every chunk_id currently live in the engine (the
    audit checks need this as the "actually indexed" population, distinct
    from what the ledger expects)."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT chunk_id FROM {TABLE_NAME} ORDER BY chunk_id")
        return [row[0] for row in cur.fetchall()]


def search(
    conn: psycopg.Connection,
    chunk_id_vector_pairs: list[tuple[str, list[float]]],
    k: int = 10,
    batch_size: int = 200,
    exclude_self: bool = True,
) -> dict[str, list[tuple[str, float]]]:
    """Batched ANN search: one SQL statement per batch_size
    queries via a VALUES(...) CROSS JOIN LATERAL, so pgvector finds each
    query's top-k neighbors server-side in a single round trip rather than
    one HTTP/SQL call per vector (the actual bottleneck at ~166,947 queries).
    Similarity = 1 - cosine_distance (`<=>` is cosine distance under
    vector_cosine_ops, matching ensure_schema's HNSW index).

    exclude_self=True (default, duplicates.py's use case) drops each query
    chunk's guaranteed self-match - a chunk's own vector is always present
    in its own engine while live, so without this every result's top hit
    would trivially be itself. exclude_self=False (retrievable.py's GDPR
    check) keeps it: querying with an erased chunk's own cached vector and
    finding that exact chunk_id still in the results IS the leakage signal
    - dropping it there would make leakage undetectable by construction."""
    limit = k + 1 if exclude_self else k
    results: dict[str, list[tuple[str, float]]] = {}
    for i in range(0, len(chunk_id_vector_pairs), batch_size):
        batch = chunk_id_vector_pairs[i : i + batch_size]
        values_sql = ", ".join(["(%s, %s::vector)"] * len(batch))
        params: list = []
        for cid, vec in batch:
            params.extend([cid, _to_vector_literal(vec)])
        params.append(limit)
        query = f"""
            WITH q(query_chunk_id, qvec) AS (VALUES {values_sql})
            SELECT q.query_chunk_id, m.chunk_id, 1 - (m.embedding <=> q.qvec) AS similarity
            FROM q
            CROSS JOIN LATERAL (
                SELECT chunk_id, embedding
                FROM {TABLE_NAME}
                ORDER BY embedding <=> q.qvec
                LIMIT %s
            ) m
        """
        with conn.cursor() as cur:
            cur.execute(query, params)
            for query_chunk_id, match_chunk_id, similarity in cur.fetchall():
                if exclude_self and match_chunk_id == query_chunk_id:
                    continue
                results.setdefault(query_chunk_id, []).append((match_chunk_id, float(similarity)))
    for cid in results:
        results[cid] = results[cid][:k]
    return results


def snapshot_stats(conn: psycopg.Connection) -> dict:
    # pg_stat_all_tables is fed by the async stats collector, which can lag
    # a beat behind a just-completed insert; ANALYZE forces n_live_tup to be
    # current so this snapshot isn't a stale 0/0 read.
    conn.execute(f"ANALYZE {TABLE_NAME};")
    with conn.cursor() as cur:
        cur.execute(
            "SELECT n_live_tup, n_dead_tup FROM pg_stat_all_tables WHERE relname = %s",
            (TABLE_NAME,),
        )
        row = cur.fetchone()
        n_live_tup, n_dead_tup = row if row else (None, None)
        cur.execute(f"SELECT count(*) FROM {TABLE_NAME}")
        count = cur.fetchone()[0]
    return {
        "engine": "pgvector",
        "count": count,
        "n_live_tup": n_live_tup,
        "n_dead_tup": n_dead_tup,
    }
