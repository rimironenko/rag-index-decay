"""Chroma adapter: upsert/delete, plus count + HNSW index directory size snapshots."""

from __future__ import annotations

import subprocess
from pathlib import Path

import chromadb

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8000
COLLECTION_NAME = "chunk_vectors"
UPSERT_BATCH_SIZE = 256
REPO_ROOT = Path(__file__).parent.parent
# Chroma 0.6.3 actually persists here, not at the docker-compose.yml `/data`
# mount point (which is an unused, empty directory - confirmed live: /data
# is 4096 bytes, /chroma/chroma holds the real HNSW segments + chroma.sqlite3,
# ~2.1GB post-churn). Fixed 2026-07-16; results/t0_baseline.json and
# results/t1_post_churn.json's hnsw_dir_bytes=4096 predate this fix and are
# known-invalid, not real "unbounded HNSW growth" measurements.
CHROMA_PERSIST_DIR = "/chroma/chroma"


def get_client(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> chromadb.HttpClient:
    return chromadb.HttpClient(host=host, port=port)


def ensure_collection(client: chromadb.HttpClient, name: str = COLLECTION_NAME):
    # Only hnsw:space is set - HNSW with Chroma server defaults (M=16,
    # construction_ef=100, search_ef=10), not tuned, per the fairness
    # protocol (label defaults-vs-tuned; see engines/pgvector_adapter.py's
    # equivalent note).
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def upsert(collection, chunk_id_vector_pairs: list[tuple[str, list[float]]]) -> None:
    """Sorted by chunk_id - Chroma accepts arbitrary string ids natively, no
    mapping needed. Batched into UPSERT_BATCH_SIZE-record requests, matching
    the qdrant adapter's batching rationale (avoid one huge HTTP request)."""
    pairs = sorted(chunk_id_vector_pairs, key=lambda p: p[0])
    for i in range(0, len(pairs), UPSERT_BATCH_SIZE):
        batch = pairs[i : i + UPSERT_BATCH_SIZE]
        collection.upsert(
            ids=[chunk_id for chunk_id, _ in batch],
            embeddings=[vec for _, vec in batch],
        )


def delete_by_chunk_id(collection, chunk_id: str) -> None:
    collection.delete(ids=[chunk_id])


def get_by_chunk_id(collection, chunk_id: str):
    """Returns the GetResult dict if chunk_id exists, else None - matching
    pgvector/qdrant's get_by_chunk_id contract (Collection.get() always
    returns a truthy dict-like object even on zero matches, so callers must
    check the actual `ids` list, not the return value's truthiness)."""
    result = collection.get(ids=[chunk_id], include=["embeddings"])
    return result if result["ids"] else None


def list_chunk_ids(collection, page_size: int = 1000) -> list[str]:
    """Bulk enumeration of every chunk_id currently live in the engine, via
    paginated get(). Chroma ids ARE the raw chunk_id strings (no UUID
    mapping needed, unlike Qdrant)."""
    chunk_ids: list[str] = []
    offset = 0
    while True:
        page = collection.get(limit=page_size, offset=offset, include=[])
        ids = page["ids"]
        chunk_ids.extend(ids)
        if len(ids) < page_size:
            break
        offset += page_size
    return chunk_ids


def search(
    collection,
    chunk_id_vector_pairs: list[tuple[str, list[float]]],
    k: int = 10,
    batch_size: int = 200,
    exclude_self: bool = True,
) -> dict[str, list[tuple[str, float]]]:
    """Batched ANN search via collection.query's multi-query
    support - one round trip performs batch_size independent top-k lookups
    server-side, instead of one HTTP call per vector. Similarity = 1 -
    distance (hnsw:space="cosine" - see ensure_collection - reports
    cosine distance).

    exclude_self=True (default, duplicates.py's use case) drops each query
    chunk's guaranteed self-match. exclude_self=False (retrievable.py's GDPR
    check) keeps it: querying with an erased chunk's own cached vector and
    finding that exact chunk_id still in the results IS the leakage signal
    - dropping it there would make leakage undetectable by construction."""
    n_results = k + 1 if exclude_self else k
    results: dict[str, list[tuple[str, float]]] = {}
    for i in range(0, len(chunk_id_vector_pairs), batch_size):
        batch = chunk_id_vector_pairs[i : i + batch_size]
        query_ids = [cid for cid, _ in batch]
        query_vecs = [vec for _, vec in batch]
        response = collection.query(
            query_embeddings=query_vecs, n_results=n_results, include=["distances"]
        )
        for query_chunk_id, match_ids, distances in zip(
            query_ids, response["ids"], response["distances"]
        ):
            hits = [
                (match_id, 1.0 - distance)
                for match_id, distance in zip(match_ids, distances)
                if not (exclude_self and match_id == query_chunk_id)
            ]
            results[query_chunk_id] = hits[:k]
    return results


def _chroma_container_id() -> str | None:
    try:
        result = subprocess.run(
            ["docker", "compose", "ps", "-q", "chroma"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        cid = result.stdout.strip()
        return cid or None
    except Exception:
        return None


def _hnsw_dir_bytes() -> int | None:
    """Best-effort: `du` the container's real persistence dir via `docker
    exec`. Returns None (rather than raising) if Docker isn't reachable,
    since this stat is informational, not load-bearing."""
    cid = _chroma_container_id()
    if not cid:
        return None
    try:
        result = subprocess.run(
            ["docker", "exec", cid, "du", "-sb", CHROMA_PERSIST_DIR],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(result.stdout.split()[0])
    except Exception:
        return None


def snapshot_stats(collection) -> dict:
    return {
        "engine": "chroma",
        "count": collection.count(),
        "hnsw_dir_bytes": _hnsw_dir_bytes(),
    }
