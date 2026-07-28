"""Qdrant adapter: upsert/delete, plus collection info (points_count, optimizer_status, segments_count) snapshots.

Qdrant point IDs must be an unsigned integer or a UUID - not an arbitrary
string - so chunk_id (a TEXT primary key everywhere else) is mapped to a
deterministic UUID5 derived from a fixed namespace. chunk_id is also stored
in the point payload for reverse lookup, and every function here takes/
returns raw chunk_id strings so callers never see the UUID mapping.
"""

from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

DEFAULT_URL = "http://localhost:6333"
COLLECTION_NAME = "chunk_vectors"
EMBEDDING_DIM = 1024
UPSERT_BATCH_SIZE = 256

# Fixed, arbitrary namespace for chunk_id -> Qdrant point id UUID5 derivation.
# Must never change once data has been upserted, or ids stop round-tripping.
NAMESPACE = uuid.UUID("d3f1b1a0-6b1a-4b8e-9c0a-2f9f9b1a0000")


def get_client(url: str = DEFAULT_URL, timeout: int = 120) -> QdrantClient:
    # Default client timeout is too short for a corpus-sized batch upsert
    # (a single multi-thousand-point request can legitimately take minutes).
    return QdrantClient(url=url, timeout=timeout)


def to_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE, chunk_id))


def ensure_collection(client: QdrantClient, name: str = COLLECTION_NAME) -> None:
    # No hnsw_config override - HNSW with Qdrant server defaults (m=16,
    # ef_construct=100), not tuned, per the fairness protocol (label
    # defaults-vs-tuned; see engines/pgvector_adapter.py's equivalent note).
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(
                size=EMBEDDING_DIM, distance=qmodels.Distance.COSINE
            ),
        )


def upsert(
    client: QdrantClient,
    chunk_id_vector_pairs: list[tuple[str, list[float]]],
    name: str = COLLECTION_NAME,
) -> None:
    """Sorted by chunk_id; wait=True since the initial baseline must be fully
    consistent - the "don't wait/flush" failure mode is introduced later,
    during churn replay, not here. Batched into UPSERT_BATCH_SIZE-point requests -
    a single corpus-sized request risks an HTTP timeout even with a
    generous client timeout configured."""
    pairs = sorted(chunk_id_vector_pairs, key=lambda p: p[0])
    for i in range(0, len(pairs), UPSERT_BATCH_SIZE):
        batch = pairs[i : i + UPSERT_BATCH_SIZE]
        points = [
            qmodels.PointStruct(id=to_point_id(chunk_id), vector=vec, payload={"chunk_id": chunk_id})
            for chunk_id, vec in batch
        ]
        client.upsert(collection_name=name, points=points, wait=True)


def delete_by_chunk_id(
    client: QdrantClient, chunk_id: str, name: str = COLLECTION_NAME, wait: bool = True
) -> None:
    # wait=False models churn replay's "async/eventually-consistent delete" failure
    # mode - the caller issues the delete without waiting for it to be fully
    # applied/flushed, same as a real team that fires-and-forgets the API call.
    client.delete(
        collection_name=name,
        points_selector=qmodels.PointIdsList(points=[to_point_id(chunk_id)]),
        wait=wait,
    )


def get_by_chunk_id(client: QdrantClient, chunk_id: str, name: str = COLLECTION_NAME):
    result = client.retrieve(
        collection_name=name, ids=[to_point_id(chunk_id)], with_payload=True, with_vectors=True
    )
    return result[0] if result else None


def list_chunk_ids(client: QdrantClient, name: str = COLLECTION_NAME, page_size: int = 1000) -> list[str]:
    """Bulk enumeration of every chunk_id currently live in the engine, via
    paginated scroll (points_count can be in the hundreds of thousands -
    a single unbounded scroll would be as risky as an unbatched upsert)."""
    chunk_ids: list[str] = []
    offset = None
    while True:
        records, next_offset = client.scroll(
            collection_name=name,
            limit=page_size,
            offset=offset,
            with_payload=["chunk_id"],
            with_vectors=False,
        )
        chunk_ids.extend(record.payload["chunk_id"] for record in records)
        if next_offset is None:
            break
        offset = next_offset
    return chunk_ids


def search(
    client: QdrantClient,
    chunk_id_vector_pairs: list[tuple[str, list[float]]],
    k: int = 10,
    name: str = COLLECTION_NAME,
    batch_size: int = 100,
    exclude_self: bool = True,
) -> dict[str, list[tuple[str, float]]]:
    """Batched ANN search via query_batch_points - one round
    trip performs batch_size independent top-k lookups server-side, instead
    of one HTTP call per vector. .score is already cosine similarity (the
    collection uses Distance.COSINE, see ensure_collection).

    exclude_self=True (default, duplicates.py's use case) drops each query
    chunk's guaranteed self-match. exclude_self=False (retrievable.py's GDPR
    check) keeps it: querying with an erased chunk's own cached vector and
    finding that exact chunk_id still in the results IS the leakage signal
    - dropping it there would make leakage undetectable by construction."""
    limit = k + 1 if exclude_self else k
    results: dict[str, list[tuple[str, float]]] = {}
    for i in range(0, len(chunk_id_vector_pairs), batch_size):
        batch = chunk_id_vector_pairs[i : i + batch_size]
        requests = [
            qmodels.QueryRequest(query=vec, limit=limit, with_payload=True)
            for _cid, vec in batch
        ]
        responses = client.query_batch_points(collection_name=name, requests=requests)
        for (query_chunk_id, _vec), response in zip(batch, responses):
            hits = []
            for point in response.points:
                match_chunk_id = point.payload["chunk_id"]
                if exclude_self and match_chunk_id == query_chunk_id:
                    continue
                hits.append((match_chunk_id, float(point.score)))
            results[query_chunk_id] = hits[:k]
    return results


def snapshot_stats(client: QdrantClient, name: str = COLLECTION_NAME) -> dict:
    # Server 1.15.5's CollectionInfo has no top-level `vectors_count` field
    # (that was removed from the API in favor of `indexed_vectors_count`) -
    # verified directly against the running container rather than assumed.
    info = client.get_collection(name)
    return {
        "engine": "qdrant",
        "points_count": info.points_count,
        "indexed_vectors_count": info.indexed_vectors_count,
        "segments_count": info.segments_count,
        "status": str(info.status),
        "optimizer_status": str(info.optimizer_status),
    }
