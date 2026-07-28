"""T0 baseline orchestrator: acquire -> clean -> chunk -> PII overlay -> embed -> upsert -> ledger.

The single driver that sequences corpus/, churn/, ledger/, and engines/ to
build the initial indexed baseline.

Usage:
    python -m corpus.ingest_baseline                 # full corpus (hours on CPU)
    python -m corpus.ingest_baseline --pilot 50       # first 50 kept docs only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import seeds
from corpus import acquire as acquire_mod
from corpus import chunk as chunk_mod
from corpus import clean as clean_mod
from engines import chroma_adapter, pgvector_adapter, qdrant_adapter
from ledger import ledger

REPO_ROOT = Path(__file__).parent.parent
LEDGER_PATH = REPO_ROOT / "ledger" / "ledger.db"
RESULTS_DIR = REPO_ROOT / "results"
EMBED_CACHE_DIR = REPO_ROOT / "corpus" / "_cache" / "embeddings"
EMBED_MANIFEST_PATH = EMBED_CACHE_DIR / "manifest.json"

EMBED_BATCH_SIZE = 32
CHECKPOINT_EVERY = 1000

ENGINES = ("pgvector", "qdrant", "chroma")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def doc_id_for(path: Path, content_dir: Path) -> str:
    """T0-relative path under content/handbook/, .md suffix stripped - stable
    identity for churn-replay rename tracking."""
    return str(path.relative_to(content_dir).with_suffix(""))


def ingest_doc_version(
    conn,
    doc_id: str,
    source_path: str,
    body: str,
    commit_sha: str,
    commit_date: str,
    tokenizer,
    action: str,
    status: str,
    is_synthetic_pii: int = 0,
    renamed_to: str | None = None,
) -> tuple[int, list[tuple[str, str]]]:
    """Cleans/chunks a single doc's body, writes ledger rows, returns
    (version_id, [(chunk_id, text), ...]). Generalizes the baseline add/live
    case (see process_doc below) to any action/status - churn replay
    (update/delete/rename) and the PII-overlay retrofit both need
    this same chunk_id scheme (f"{doc_id}::v{version_id}::{ordinal:04d}")
    with a controllable action/status/commit_sha rather than a hardcoded
    action="add"/status="live"/t0_sha."""
    content_hash = sha256(body)
    ledger.insert_document(
        conn, doc_id, source_path, commit_sha, commit_sha,
        status=status, renamed_to=renamed_to, is_synthetic_pii=is_synthetic_pii,
    )
    version_id = ledger.get_or_create_doc_version(
        conn, doc_id, commit_sha, commit_date, content_hash, action=action
    )

    spans = chunk_mod.chunk_text(body, tokenizer)
    chunk_rows = []
    chunk_records = []
    for span in spans:
        chunk_id = f"{doc_id}::v{version_id}::{span.ordinal:04d}"
        chunk_rows.append(
            (chunk_id, doc_id, version_id, span.ordinal, sha256(span.text), span.token_count)
        )
        chunk_records.append((chunk_id, span.text))
    ledger.insert_chunks(conn, chunk_rows)
    return version_id, chunk_records


def process_doc(conn, doc_id: str, source_path: str, body: str, t0_sha: str, t0_date: str,
                 tokenizer, is_synthetic_pii: int = 0) -> list[tuple[str, str]]:
    """Cleans/chunks a single doc's body, writes ledger rows, returns [(chunk_id, text), ...].
    Thin wrapper over ingest_doc_version fixed to the baseline's only case: action="add", status="live"."""
    _version_id, chunk_records = ingest_doc_version(
        conn, doc_id, source_path, body, t0_sha, t0_date, tokenizer,
        action="add", status="live", is_synthetic_pii=is_synthetic_pii,
    )
    return chunk_records


def embed_all(chunk_records: list[tuple[str, str]]) -> dict[str, list[float]]:
    """Embeds all (chunk_id, text) pairs with BGE-M3, fp32, CPU. Checkpoints
    every CHECKPOINT_EVERY chunks to on-disk shards so an interrupt doesn't
    force a full restart - resume by skipping chunk_ids already embedded."""
    EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    embeddings: dict[str, list[float]] = {}
    completed_shards: set[int] = set()

    if EMBED_MANIFEST_PATH.exists():
        manifest = json.loads(EMBED_MANIFEST_PATH.read_text())
        completed_shards = set(manifest.get("completed_shards", []))
        for shard_idx in completed_shards:
            shard_path = EMBED_CACHE_DIR / f"shard_{shard_idx:05d}.npz"
            if not shard_path.exists():
                continue
            data = np.load(shard_path, allow_pickle=True)
            for cid, vec in zip(data["chunk_ids"], data["vectors"]):
                embeddings[str(cid)] = vec.tolist()

    remaining = [(cid, text) for cid, text in chunk_records if cid not in embeddings]
    if not remaining:
        return embeddings

    # Import here, not at module top: torch/sentence-transformers are heavy
    # and only needed when there's actually embedding work left to do.
    import torch
    from sentence_transformers import SentenceTransformer

    seeds.seed_everything()
    torch.set_num_threads(os.cpu_count() or 4)
    model = SentenceTransformer(
        chunk_mod.MODEL_NAME, revision=chunk_mod.MODEL_REVISION, device="cpu"
    )

    shard_idx = (max(completed_shards) + 1) if completed_shards else 0
    for i in range(0, len(remaining), CHECKPOINT_EVERY):
        batch = remaining[i : i + CHECKPOINT_EVERY]
        ids = [cid for cid, _ in batch]
        texts = [text for _, text in batch]
        vectors = model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        shard_path = EMBED_CACHE_DIR / f"shard_{shard_idx:05d}.npz"
        np.savez(shard_path, chunk_ids=np.array(ids, dtype=object), vectors=vectors)
        completed_shards.add(shard_idx)
        EMBED_MANIFEST_PATH.write_text(
            json.dumps({"completed_shards": sorted(completed_shards)}, indent=2)
        )

        for cid, vec in zip(ids, vectors):
            embeddings[cid] = vec.tolist()
        shard_idx += 1

    return embeddings


def upsert_all_engines(conn, pairs: list[tuple[str, list[float]]], t0_sha: str) -> None:
    """Upserts identical vectors into all three engines, then logs index_events
    per engine - skipping chunk_ids already logged for that engine (resumable)."""
    pg_conn = pgvector_adapter.get_conn()
    pgvector_adapter.ensure_schema(pg_conn)
    pgvector_adapter.upsert(pg_conn, pairs)

    qd_client = qdrant_adapter.get_client()
    qdrant_adapter.ensure_collection(qd_client)
    qdrant_adapter.upsert(qd_client, pairs)

    ch_client = chroma_adapter.get_client()
    ch_coll = chroma_adapter.ensure_collection(ch_client)
    chroma_adapter.upsert(ch_coll, pairs)

    applied_at = datetime.now(timezone.utc).isoformat()
    for engine_name in ENGINES:
        already_logged = ledger.existing_index_event_chunk_ids(conn, engine_name)
        rows = [
            (engine_name, cid, "upsert", applied_at, t0_sha, 1)
            for cid, _ in pairs
            if cid not in already_logged
        ]
        if rows:
            ledger.insert_index_events(conn, rows)
    conn.commit()

    return {
        "pg_conn": pg_conn,
        "qd_client": qd_client,
        "ch_coll": ch_coll,
    }


def run(pilot: int | None = None) -> dict:
    seeds.seed_everything()

    t0 = acquire_mod.acquire()
    content_dir = Path(t0["content_dir"])
    t0_sha = t0["sha"]
    t0_date = t0["commit_date"]

    conn = ledger.init_db(LEDGER_PATH)
    tokenizer = chunk_mod.get_tokenizer()

    md_files = sorted(content_dir.rglob("*.md"))
    if pilot:
        md_files = md_files[:pilot]

    chunk_records: list[tuple[str, str]] = []
    doc_count = 0

    for path in md_files:
        doc = clean_mod.clean_file(path)
        if doc is None:
            continue
        doc_id = doc_id_for(path, content_dir)
        records = process_doc(
            conn, doc_id, doc.source_path, doc.body, t0_sha, t0_date, tokenizer
        )
        chunk_records.extend(records)
        doc_count += 1
    conn.commit()

    # Embed once, checkpointed/resumable.
    embeddings = embed_all(chunk_records)
    pairs = [(cid, embeddings[cid]) for cid, _ in chunk_records]

    clients = upsert_all_engines(conn, pairs, t0_sha)

    # Synthetic PII overlay: 100 self-contained per-employee docs,
    # one chunk each - self-contained (own embed_all/upsert calls), see
    # churn/pii_overlay.py. Deferred import: avoids a module-level circular
    # import (churn.pii_overlay imports corpus.ingest_baseline for the shared
    # ingest_doc_version/embed_all helpers).
    from churn import pii_overlay
    pii_result = pii_overlay.migrate_pii_overlay_to_per_employee(conn, tokenizer, t0_sha, t0_date)
    # Fresh ledger count rather than arithmetic: correct whether this is a
    # from-scratch run (pii_result has real counts) or a rerun where the PII
    # overlay was already migrated in a prior run (pii_result is {"skipped": True}).
    chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    stats = {
        "t0_sha": t0_sha,
        "t0_commit_date": t0_date,
        "doc_count": doc_count,
        "chunk_count": chunk_count,
        "pgvector": pgvector_adapter.snapshot_stats(clients["pg_conn"]),
        "qdrant": qdrant_adapter.snapshot_stats(clients["qd_client"]),
        "chroma": chroma_adapter.snapshot_stats(clients["ch_coll"]),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "t0_baseline.json").write_text(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Acquire, clean, chunk, embed, upsert T0 baseline.")
    parser.add_argument("--pilot", type=int, default=None, help="Limit to the first N markdown files (pilot run).")
    args = parser.parse_args()
    result = run(pilot=args.pilot)
    print(json.dumps(result, indent=2))
