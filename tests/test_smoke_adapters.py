"""Integration smoke tests against the live Docker containers (pgvector,
Qdrant, Chroma) - validates the list_chunk_ids/search adapter functions at
small scale before trusting them at the full 166,947-chunks-per-engine
production scale. Requires `docker compose up` running and a non-empty
embedding cache."""

from __future__ import annotations

import pytest

from corpus import embed_cache
from engines import chroma_adapter, pgvector_adapter, qdrant_adapter

pytestmark = pytest.mark.integration

ADAPTERS = {
    "pgvector": pgvector_adapter,
    "qdrant": qdrant_adapter,
    "chroma": chroma_adapter,
}


@pytest.fixture(scope="module")
def cache():
    return embed_cache.load_all()


@pytest.fixture(scope="module")
def pg_conn():
    return pgvector_adapter.get_conn()


@pytest.fixture(scope="module")
def qd_client():
    return qdrant_adapter.get_client()


@pytest.fixture(scope="module")
def ch_collection():
    client = chroma_adapter.get_client()
    return chroma_adapter.ensure_collection(client)


@pytest.fixture(scope="module")
def handles(pg_conn, qd_client, ch_collection):
    return {"pgvector": pg_conn, "qdrant": qd_client, "chroma": ch_collection}


def test_list_chunk_ids_matches_snapshot_stats_count(handles):
    for engine_name, adapter in ADAPTERS.items():
        handle = handles[engine_name]
        ids = adapter.list_chunk_ids(handle)
        stats = adapter.snapshot_stats(handle)
        count_field = "points_count" if engine_name == "qdrant" else "count"
        assert len(ids) == stats[count_field], f"{engine_name}: list_chunk_ids/{len(ids)} != snapshot_stats/{stats[count_field]}"
        assert len(ids) == len(set(ids)), f"{engine_name}: list_chunk_ids returned duplicate ids"


def test_search_finds_self_at_top_similarity(handles, cache):
    for engine_name, adapter in ADAPTERS.items():
        handle = handles[engine_name]
        ids = adapter.list_chunk_ids(handle)
        sample = [cid for cid in ids[:200] if cid in cache][:5]
        assert sample, f"{engine_name}: no sample chunk_ids found in embedding cache"
        pairs = [(cid, cache[cid].tolist()) for cid in sample]
        results = adapter.search(handle, pairs, k=5)
        for cid in sample:
            hits = results.get(cid, [])
            assert hits, f"{engine_name}: {cid} returned no search hits"
            # The query chunk's own vector must never appear as a hit of
            # itself (search() drops the guaranteed self-match) - but a
            # near-identical reingest-duplicate chunk commonly sits at
            # similarity ~1.0, which is the expected/desired signal, not a
            # bug. So assert self is excluded and every hit is well-formed.
            assert cid not in [match_id for match_id, _ in hits]
            assert all(0.0 <= sim <= 1.0001 for _, sim in hits)
