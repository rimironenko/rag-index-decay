"""Read-only loader for the embedding shard cache written by
corpus/ingest_baseline.py's embed_all(). Lives in corpus/ because corpus/
already owns the cache-writing code (EMBED_CACHE_DIR/EMBED_MANIFEST_PATH are
defined in corpus/ingest_baseline.py) - keeping the reader next to the
writer, and reusable by any downstream consumer of the cache."""

from __future__ import annotations

import json

import numpy as np

from corpus.ingest_baseline import EMBED_CACHE_DIR, EMBED_MANIFEST_PATH


def load_all(cache_dir=EMBED_CACHE_DIR, manifest_path=EMBED_MANIFEST_PATH) -> dict[str, np.ndarray]:
    """Loads every completed shard's chunk_id -> float32 vector into one dict
    (~1.1GB for the full 287,346-chunk corpus - fits comfortably in the
    documented 36GB RAM). Deliberately loads everything rather than
    selectively: the cache holds entries for chunk_ids beyond what's
    currently live in any engine (superseded/orphaned chunks are never
    pruned), which is fine - callers look up only the specific keys they
    need. Mirrors ingest_baseline.embed_all()'s shard-reading loop (duplicated
    deliberately rather than refactored out of already-verified baseline/
    churn-replay code this late)."""
    embeddings: dict[str, np.ndarray] = {}
    if not manifest_path.exists():
        return embeddings
    manifest = json.loads(manifest_path.read_text())
    for shard_idx in sorted(manifest.get("completed_shards", [])):
        shard_path = cache_dir / f"shard_{shard_idx:05d}.npz"
        if not shard_path.exists():
            continue
        data = np.load(shard_path, allow_pickle=True)
        for cid, vec in zip(data["chunk_ids"], data["vectors"]):
            embeddings[str(cid)] = vec
    return embeddings
