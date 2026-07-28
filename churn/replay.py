"""Walk git history T0 -> HEAD, classify changes and apply the four canonical
ingestion failure modes, the mid-timeline full-reingest
duplicate event and the labeled PII erasure requests, then
snapshot post-churn engine stats.

Usage:
    python -m churn.replay                    # full ~4,742-commit T0->HEAD walk
    python -m churn.replay --max-commits 100  # smoke test
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import git

import seeds
from churn import pii_overlay
from corpus import acquire as acquire_mod
from corpus import chunk as chunk_mod
from corpus import clean as clean_mod
from corpus import ingest_baseline
from engines import chroma_adapter, pgvector_adapter, qdrant_adapter
from ledger import ledger

REPO_ROOT = Path(__file__).parent.parent
LEDGER_PATH = REPO_ROOT / "ledger" / "ledger.db"
RESULTS_DIR = REPO_ROOT / "results"

T0_SHA = "d9dfa4eb7acde0d7ca4497f6014371c0f5ae847d"
T0_DATE = "2025-06-02T02:04:00+00:00"
# Pinned, not dynamically re-resolved from origin/main - a rerun weeks later
# would otherwise walk a different (larger) commit range, silently shifting
# the reingest/erasure commit selection and breaking reproducibility against
# this project's own "pin every version" ethos (see the fairness protocol
# section of README.md).
HEAD_SHA = "6a7e263deb491dffe5c490fb2d26a0cbadc431d5"
HEAD_DATE = "2026-07-10T17:26:48-07:00"

CONTENT_PATHSPEC = "content/handbook"

UPDATE_BUCKET_SALT = "mode-a-update"
RENAME_BUCKET_SALT = "mode-d-rename"
BUGGY_RATIO_PCT = 50  # ~50/50 buggy-vs-correct mix, per doc (hash of doc_id alone,
                      # so a doc is consistently buggy or correct for its lifetime here)

ERASURE_SELECTION_SALT = "pii-erasure-2.3"
ERASURE_COUNT = 40
ERASURE_COMMIT_FRACTION = 0.9  # near the end of the walk, not literally HEAD


@dataclass
class FileChange:
    action: str  # "add" | "update" | "delete" | "rename"
    old_path: str | None
    new_path: str | None


@dataclass
class EngineOp:
    commit_sha: str
    op: str  # "upsert" | "delete" | "skip"
    chunk_id: str
    expected_present: int
    source_chunk_id_for_vector: str | None = None  # set only for mode-C duplicates


@dataclass
class WalkState:
    """Run-scoped, in-memory tracking of "current" per-doc version/status as of
    each point in the chronological walk. Deliberately NOT re-derived from
    ledger.latest_version_id()/get_document_status() during Pass 1: those
    reflect the ledger's fully-accumulated state, which on a rerun already
    contains every future event from the prior run. Using them directly would
    make an "update" event reprocessed on rerun see its doc's FINAL (e.g.
    already-deleted) state rather than its state at that point in a fresh
    walk - concretely, it would compute the wrong "old chunk ids" to
    skip/delete, and could even resurrect content a later event in the same
    walk correctly deleted. Seeding once from each doc's T0 version (identified
    by commit_sha == T0_SHA, which is only ever set by the baseline ingest / the PII
    migration, never by this walk) and then updating purely in-memory as Pass 1
    progresses keeps this correct and rerun-safe."""

    doc_version: dict[str, int]
    doc_status: dict[str, str]
    pending_embed: list[tuple[str, str]]
    engine_ops: list[EngineOp]
    counters: dict[str, int]


def seed_walk_state(conn) -> WalkState:
    cur = conn.execute("SELECT doc_id, version_id FROM doc_versions WHERE commit_sha = ?", (T0_SHA,))
    doc_version = {doc_id: version_id for doc_id, version_id in cur.fetchall()}
    doc_status = {doc_id: "live" for doc_id in doc_version}
    return WalkState(
        doc_version=doc_version, doc_status=doc_status,
        pending_embed=[], engine_ops=[], counters=defaultdict(int),
    )


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def doc_id_for_git_path(git_path: str) -> str:
    return ingest_baseline.doc_id_for(Path(git_path), Path(CONTENT_PATHSPEC))


def read_blob_text(repo: git.Repo, commit_sha: str, path: str) -> str:
    blob = repo.commit(commit_sha).tree / path
    return blob.data_stream.read().decode("utf-8", errors="replace")


def _is_buggy(key: str, salt: str, ratio_pct: int = BUGGY_RATIO_PCT) -> bool:
    digest = hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()
    return (int(digest, 16) % 100) < ratio_pct


def select_erasure_employee_ids(employee_doc_ids: list[str], n: int = ERASURE_COUNT) -> list[str]:
    """Exact-count deterministic selection: sort ALL employee doc_ids by a
    stable hash digest, take the first n."""
    ranked = sorted(
        employee_doc_ids,
        key=lambda d: hashlib.sha256(f"{ERASURE_SELECTION_SALT}:{d}".encode()).hexdigest(),
    )
    return ranked[:n]


def apply_delete(pg_conn, qd_client, ch_coll, chunk_id: str) -> None:
    """The ONE place mode-B (async/eventually-consistent delete) semantics
    live. Used uniformly for every delete in the system - real git deletes,
    the "correct" bucket's counterpart delete on update/rename, and PII
    erasure - so the failure mode is centralized, not scattered."""
    pgvector_adapter.delete_by_chunk_id(pg_conn, chunk_id)  # no VACUUM anywhere -- inherent
    qdrant_adapter.delete_by_chunk_id(qd_client, chunk_id, wait=False)
    chroma_adapter.delete_by_chunk_id(ch_coll, chunk_id)  # no compaction call -- inherent


# --------------------------------------------------------------------------
# Git walk and diff classification
# --------------------------------------------------------------------------

def walk_commits(repo: git.Repo, t0_sha: str, head_sha: str) -> list[git.Commit]:
    # first-parent: collapses GitLab's MR-merge workflow to one changeset per
    # merged MR (4,742 commits vs. 8,381 walking every merge+branch commit).
    return list(
        repo.iter_commits(f"{t0_sha}..{head_sha}", first_parent=True, reverse=True, paths=CONTENT_PATHSPEC)
    )


def classify_diff(repo: git.Repo, prev_sha: str, commit_sha: str) -> list[FileChange]:
    """Diffs between consecutive commits in the filtered first-parent list
    (not each commit's literal git parent), so one changeset = one merged MR."""
    raw = repo.git.diff(prev_sha, commit_sha, "--name-status", "-M", "--", CONTENT_PATHSPEC)
    changes: list[FileChange] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0]
        if status.startswith("R"):
            old_path, new_path = parts[1], parts[2]
            old_md, new_md = old_path.endswith(".md"), new_path.endswith(".md")
            if old_md and new_md:
                changes.append(FileChange("rename", old_path, new_path))
            elif new_md:
                changes.append(FileChange("add", None, new_path))
            elif old_md:
                changes.append(FileChange("delete", old_path, None))
            # neither side .md -> irrelevant, skip
        else:
            path = parts[1]
            if not path.endswith(".md"):
                continue
            if status == "A":
                changes.append(FileChange("add", None, path))
            elif status == "M":
                changes.append(FileChange("update", path, path))
            elif status == "D":
                changes.append(FileChange("delete", path, None))
            # ignore other statuses (e.g. T type-change) - none expected for
            # a markdown-only content dir
    return changes


# --------------------------------------------------------------------------
# Mid-timeline / erasure commit selection
# --------------------------------------------------------------------------

def select_reingest_commit(commits: list[git.Commit], t0_dt, head_dt) -> git.Commit:
    midpoint = t0_dt + (head_dt - t0_dt) / 2
    for c in commits:
        if c.committed_datetime >= midpoint:
            return c
    return commits[-1]


def select_erasure_commit(commits: list[git.Commit], fraction: float = ERASURE_COMMIT_FRACTION) -> git.Commit:
    idx = min(int(len(commits) * fraction), len(commits) - 1)
    return commits[idx]


# --------------------------------------------------------------------------
# Pass 1 - classify + ledger writes + accumulate engine ops
# --------------------------------------------------------------------------

def _apply_effective_delete(conn, doc_id: str, source_path: str, commit: git.Commit, state: WalkState) -> None:
    """Records a real (or effective, e.g. content shrank below the stub
    threshold) delete. Writes documents/doc_versions rows directly rather
    than via ingest_doc_version, since a delete-action version must carry
    zero chunks (there's no body to chunk)."""
    commit_sha, commit_date = commit.hexsha, commit.committed_datetime.isoformat()
    old_version_id = state.doc_version.get(doc_id)
    chunk_ids = ledger.chunk_ids_for_version(conn, old_version_id) if old_version_id else []

    ledger.insert_document(conn, doc_id, source_path, commit_sha, commit_sha, status="deleted")
    version_id = ledger.get_or_create_doc_version(
        conn, doc_id, commit_sha, commit_date, ingest_baseline.sha256(f"DELETED:{doc_id}:{commit_sha}"), action="delete"
    )

    for cid in chunk_ids:
        # Real deletes are ALWAYS mode B - never bucketed buggy/correct (no
        # "correct" counterpart needed. Mode B's soft-delete behavior is the
        # engines' own default, not something to contrast against).
        state.engine_ops.append(EngineOp(commit_sha, "delete", cid, 0))

    state.doc_status[doc_id] = "deleted"
    state.doc_version[doc_id] = version_id
    state.counters["delete"] += 1


def _apply_change_to_ledger(conn, repo: git.Repo, change: FileChange, commit: git.Commit, tokenizer, state: WalkState) -> None:
    commit_sha = commit.hexsha
    commit_date = commit.committed_datetime.isoformat()

    if change.action == "add":
        doc_id = doc_id_for_git_path(change.new_path)
        cleaned = clean_mod.clean_text(read_blob_text(repo, commit_sha, change.new_path), change.new_path)
        if cleaned is None:
            return  # stub - never tracked, matches the baseline ingest's semantics
        version_id, records = ingest_baseline.ingest_doc_version(
            conn, doc_id, change.new_path, cleaned.body, commit_sha, commit_date, tokenizer,
            action="add", status="live",
        )
        for cid, text in records:
            state.pending_embed.append((cid, text))
            state.engine_ops.append(EngineOp(commit_sha, "upsert", cid, 1))
        state.doc_status[doc_id] = "live"
        state.doc_version[doc_id] = version_id
        state.counters["add"] += 1

    elif change.action == "update":
        doc_id = doc_id_for_git_path(change.new_path)
        cleaned = clean_mod.clean_text(read_blob_text(repo, commit_sha, change.new_path), change.new_path)
        prior_status = state.doc_status.get(doc_id)

        if cleaned is None:
            if prior_status == "live":
                _apply_effective_delete(conn, doc_id, change.old_path, commit, state)
            return  # else: was never tracked, still a stub -> no-op

        if prior_status != "live":
            # First real appearance (was a stub at every prior point, or never seen).
            version_id, records = ingest_baseline.ingest_doc_version(
                conn, doc_id, change.new_path, cleaned.body, commit_sha, commit_date, tokenizer,
                action="add", status="live",
            )
            for cid, text in records:
                state.pending_embed.append((cid, text))
                state.engine_ops.append(EngineOp(commit_sha, "upsert", cid, 1))
            state.doc_status[doc_id] = "live"
            state.doc_version[doc_id] = version_id
            state.counters["add_from_stub"] += 1
            return

        old_version_id = state.doc_version.get(doc_id)
        old_chunk_ids = ledger.chunk_ids_for_version(conn, old_version_id) if old_version_id else []

        version_id, records = ingest_baseline.ingest_doc_version(
            conn, doc_id, change.new_path, cleaned.body, commit_sha, commit_date, tokenizer,
            action="update", status="live",
        )
        for cid, text in records:
            state.pending_embed.append((cid, text))
            state.engine_ops.append(EngineOp(commit_sha, "upsert", cid, 1))

        buggy = _is_buggy(doc_id, UPDATE_BUCKET_SALT)
        for old_cid in old_chunk_ids:
            state.engine_ops.append(EngineOp(commit_sha, "skip" if buggy else "delete", old_cid, 0))

        state.doc_status[doc_id] = "live"
        state.doc_version[doc_id] = version_id
        state.counters["update_buggy" if buggy else "update_correct"] += 1

    elif change.action == "delete":
        doc_id = doc_id_for_git_path(change.old_path)
        if state.doc_status.get(doc_id) != "live":
            return  # never tracked (stub) or already gone
        _apply_effective_delete(conn, doc_id, change.old_path, commit, state)

    elif change.action == "rename":
        old_doc_id = doc_id_for_git_path(change.old_path)
        new_doc_id = doc_id_for_git_path(change.new_path)
        prior_status = state.doc_status.get(old_doc_id)
        cleaned = clean_mod.clean_text(read_blob_text(repo, commit_sha, change.new_path), change.new_path)

        if prior_status != "live":
            # Old side was never tracked (stub) - degrade to a plain add of the new path.
            if cleaned is not None:
                version_id, records = ingest_baseline.ingest_doc_version(
                    conn, new_doc_id, change.new_path, cleaned.body, commit_sha, commit_date, tokenizer,
                    action="add", status="live",
                )
                for cid, text in records:
                    state.pending_embed.append((cid, text))
                    state.engine_ops.append(EngineOp(commit_sha, "upsert", cid, 1))
                state.doc_status[new_doc_id] = "live"
                state.doc_version[new_doc_id] = version_id
            state.counters["rename_degraded_to_add"] += 1
            return

        old_version_id = state.doc_version.get(old_doc_id)
        old_chunk_ids = ledger.chunk_ids_for_version(conn, old_version_id) if old_version_id else []

        # Ground truth ALWAYS correctly records the rename, regardless of bucket.
        ledger.insert_document(
            conn, old_doc_id, change.old_path, commit_sha, commit_sha, status="renamed", renamed_to=new_doc_id
        )
        rename_marker_version_id = ledger.get_or_create_doc_version(
            conn, old_doc_id, commit_sha, commit_date,
            ingest_baseline.sha256(f"RENAMED:{old_doc_id}->{new_doc_id}:{commit_sha}"), action="rename",
        )
        state.doc_status[old_doc_id] = "renamed"
        state.doc_version[old_doc_id] = rename_marker_version_id

        if cleaned is not None:
            version_id, records = ingest_baseline.ingest_doc_version(
                conn, new_doc_id, change.new_path, cleaned.body, commit_sha, commit_date, tokenizer,
                action="rename", status="live",
            )
            for cid, text in records:
                state.pending_embed.append((cid, text))
                state.engine_ops.append(EngineOp(commit_sha, "upsert", cid, 1))
            state.doc_status[new_doc_id] = "live"
            state.doc_version[new_doc_id] = version_id

        buggy = _is_buggy(old_doc_id, RENAME_BUCKET_SALT)
        for old_cid in old_chunk_ids:
            state.engine_ops.append(EngineOp(commit_sha, "skip" if buggy else "delete", old_cid, 0))
        state.counters["rename_buggy" if buggy else "rename_correct"] += 1


# --------------------------------------------------------------------------
# Mode C - full re-ingest duplicate explosion
# --------------------------------------------------------------------------

def _apply_full_reingest(conn, commit: git.Commit, state: WalkState) -> None:
    commit_sha = commit.hexsha
    reingest_sha_marker = f"{commit_sha}::reingest"
    commit_date = commit.committed_datetime.isoformat()

    # is_synthetic_pii docs are deliberately included - a real full-reingest
    # job wouldn't special-case them.
    live_doc_ids = [doc_id for doc_id, status in state.doc_status.items() if status == "live"]

    for doc_id in live_doc_ids:
        version_id = state.doc_version.get(doc_id)
        if version_id is None:
            continue
        old_chunk_ids = ledger.chunk_ids_for_version(conn, version_id)
        if not old_chunk_ids:
            continue
        content_sha = conn.execute(
            "SELECT content_sha256 FROM doc_versions WHERE version_id = ?", (version_id,)
        ).fetchone()[0]
        new_version_id = ledger.get_or_create_doc_version(
            conn, doc_id, reingest_sha_marker, commit_date, content_sha, action="add"
        )
        old_rows = conn.execute(
            "SELECT ordinal, text_sha256, token_count FROM chunks WHERE version_id = ? ORDER BY ordinal",
            (version_id,),
        ).fetchall()
        new_chunk_rows = []
        for (ordinal, text_sha, token_count), old_cid in zip(old_rows, old_chunk_ids):
            new_cid = f"{doc_id}::v{new_version_id}::{ordinal:04d}"
            new_chunk_rows.append((new_cid, doc_id, new_version_id, ordinal, text_sha, token_count))
            # No re-embedding needed: embed_all() loads every completed cache
            # shard unconditionally before computing what's left to do, so
            # by the time Pass 3 resolves vectors, embeddings[old_cid] is
            # already guaranteed present - content is byte-identical, so
            # BGE-M3's deterministic fp32 CPU embedding is bit-identical too.
            state.engine_ops.append(
                EngineOp(commit_sha, "upsert", new_cid, 1, source_chunk_id_for_vector=old_cid)
            )
        ledger.insert_chunks(conn, new_chunk_rows)
        state.doc_version[doc_id] = new_version_id
        state.counters["reingest_duplicate_chunks"] += len(new_chunk_rows)


# --------------------------------------------------------------------------
# PII erasure
# --------------------------------------------------------------------------

def _apply_pii_erasure(conn, commit: git.Commit, state: WalkState) -> None:
    commit_sha = commit.hexsha
    all_employee_doc_ids = [
        pii_overlay.employee_doc_id(f"SYN-{i + 1:04d}") for i in range(pii_overlay.N_SYNTHETIC_EMPLOYEES)
    ]
    selected = select_erasure_employee_ids(all_employee_doc_ids, ERASURE_COUNT)

    existing_by_engine = {
        engine_name: ledger.existing_erasure_request_doc_ids(conn, engine_name)
        for engine_name in ingest_baseline.ENGINES
    }

    for doc_id in selected:
        version_id = state.doc_version.get(doc_id)
        chunk_ids = ledger.chunk_ids_for_version(conn, version_id) if version_id else []  # exactly 1, per profile design
        for cid in chunk_ids:
            state.engine_ops.append(EngineOp(commit_sha, "delete", cid, 0))

        for engine_name in ingest_baseline.ENGINES:
            if doc_id not in existing_by_engine[engine_name]:
                ledger.insert_erasure_request(conn, doc_id, commit_sha, engine_name, expected_absent=1)

        ledger.insert_document(
            conn, doc_id, f"synthetic:employee/{doc_id.rsplit('/', 1)[-1]}.md",
            commit_sha, commit_sha, status="deleted", is_synthetic_pii=1,
        )
        state.doc_status[doc_id] = "deleted"

    state.counters["pii_erasure_employee_count"] = len(selected)


# --------------------------------------------------------------------------
# Pass 3 - apply to engines (batched) + write index_events
# --------------------------------------------------------------------------

def _apply_engine_ops(conn, engine_ops: list[EngineOp], embeddings: dict[str, list[float]]) -> dict:
    pg_conn = pgvector_adapter.get_conn()
    pgvector_adapter.ensure_schema(pg_conn)
    qd_client = qdrant_adapter.get_client()
    qdrant_adapter.ensure_collection(qd_client)
    ch_client = chroma_adapter.get_client()
    ch_coll = chroma_adapter.ensure_collection(ch_client)

    upsert_ops = [op for op in engine_ops if op.op == "upsert"]
    delete_ops = [op for op in engine_ops if op.op == "delete"]
    skip_ops = [op for op in engine_ops if op.op == "skip"]

    def vector_for(op: EngineOp) -> list[float]:
        key = op.source_chunk_id_for_vector or op.chunk_id
        return embeddings[key]

    # Batched (per user decision): all ledger bookkeeping above happened
    # per-commit in chronological order (commit_sha/applied_at on every
    # index_events row preserves full historical provenance); only the
    # physical engine writes are batched here for runtime. Safe because a
    # given chunk_id never appears in both an upsert-kind and a delete-kind
    # op within one run - chunk_id embeds version_id, and each version's
    # chunks are upserted-once (on creation) or deleted/skipped-once (on
    # being superseded), never both.
    pairs = [(op.chunk_id, vector_for(op)) for op in upsert_ops]
    if pairs:
        pgvector_adapter.upsert(pg_conn, pairs)
        qdrant_adapter.upsert(qd_client, pairs)
        chroma_adapter.upsert(ch_coll, pairs)

    for op in delete_ops:
        apply_delete(pg_conn, qd_client, ch_coll, op.chunk_id)

    applied_at = datetime.now(timezone.utc).isoformat()
    for engine_name in ingest_baseline.ENGINES:
        for op_kind, ops in (("upsert", upsert_ops), ("delete", delete_ops), ("skip", skip_ops)):
            already = ledger.existing_index_event_chunk_ids(conn, engine_name, op=op_kind)
            rows = [
                (engine_name, op.chunk_id, op_kind, applied_at, op.commit_sha, op.expected_present)
                for op in ops
                if op.chunk_id not in already
            ]
            if rows:
                ledger.insert_index_events(conn, rows)
    conn.commit()
    return {"pg_conn": pg_conn, "qd_client": qd_client, "ch_coll": ch_coll}


# --------------------------------------------------------------------------
# Post-churn snapshot
# --------------------------------------------------------------------------

def _snapshot_and_write_results(conn, clients, commits, reingest_commit, erasure_commit, counters, pii_migration) -> dict:
    doc_count_live = conn.execute("SELECT COUNT(*) FROM documents WHERE status='live'").fetchone()[0]
    doc_count_deleted = conn.execute("SELECT COUNT(*) FROM documents WHERE status='deleted'").fetchone()[0]
    doc_count_renamed = conn.execute("SELECT COUNT(*) FROM documents WHERE status='renamed'").fetchone()[0]
    chunk_count_total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    index_events_counts = {
        e: conn.execute("SELECT COUNT(*) FROM index_events WHERE engine=?", (e,)).fetchone()[0]
        for e in ingest_baseline.ENGINES
    }
    erasure_requests_count = conn.execute("SELECT COUNT(*) FROM erasure_requests").fetchone()[0]

    pii_migration_summary = {k: v for k, v in pii_migration.items() if k not in ("pg_conn", "qd_client", "ch_coll")}

    stats = {
        "t0_sha": T0_SHA,
        "t0_commit_date": T0_DATE,
        "head_sha": HEAD_SHA,
        "head_commit_date": HEAD_DATE,
        "commits_walked": len(commits),
        "reingest_commit_sha": reingest_commit.hexsha,
        "reingest_commit_date": reingest_commit.committed_datetime.isoformat(),
        "erasure_commit_sha": erasure_commit.hexsha,
        "erasure_commit_date": erasure_commit.committed_datetime.isoformat(),
        "doc_count_live": doc_count_live,
        "doc_count_deleted": doc_count_deleted,
        "doc_count_renamed": doc_count_renamed,
        "chunk_count_total_ledger": chunk_count_total,
        "churn_counters": dict(counters),
        "index_events_written": index_events_counts,
        "erasure_requests_written": erasure_requests_count,
        "pii_overlay_restructure": pii_migration_summary,
        "pgvector": pgvector_adapter.snapshot_stats(clients["pg_conn"]),
        "qdrant": qdrant_adapter.snapshot_stats(clients["qd_client"]),
        "chroma": chroma_adapter.snapshot_stats(clients["ch_coll"]),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "t1_post_churn.json").write_text(json.dumps(stats, indent=2))
    return stats


# --------------------------------------------------------------------------
# Top-level orchestrator
# --------------------------------------------------------------------------

def run(max_commits: int | None = None) -> dict:
    seeds.seed_everything()
    conn = ledger.get_conn(LEDGER_PATH)
    tokenizer = chunk_mod.get_tokenizer()

    pii_migration = pii_overlay.migrate_pii_overlay_to_per_employee(conn, tokenizer, T0_SHA, T0_DATE)

    repo = acquire_mod.clone_or_update()  # fetches, but the walk range uses the pinned HEAD_SHA below
    commits = walk_commits(repo, T0_SHA, HEAD_SHA)
    if max_commits:
        commits = commits[:max_commits]

    t0_dt = repo.commit(T0_SHA).committed_datetime
    head_dt = repo.commit(HEAD_SHA).committed_datetime
    reingest_commit = select_reingest_commit(commits, t0_dt, head_dt)
    erasure_commit = select_erasure_commit(commits)

    state = seed_walk_state(conn)

    prev_sha = T0_SHA
    for commit in commits:
        for change in classify_diff(repo, prev_sha, commit.hexsha):
            _apply_change_to_ledger(conn, repo, change, commit, tokenizer, state)
        if commit.hexsha == reingest_commit.hexsha:
            _apply_full_reingest(conn, commit, state)
        if commit.hexsha == erasure_commit.hexsha:
            _apply_pii_erasure(conn, commit, state)
        prev_sha = commit.hexsha
    conn.commit()

    embeddings = ingest_baseline.embed_all(state.pending_embed)
    clients = _apply_engine_ops(conn, state.engine_ops, embeddings)

    return _snapshot_and_write_results(
        conn, clients, commits, reingest_commit, erasure_commit, state.counters, pii_migration
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replay real git churn, apply the four ingestion failure modes, PII erasure."
    )
    parser.add_argument(
        "--max-commits", type=int, default=None, help="Limit the walk to the first N commits (smoke test)."
    )
    args = parser.parse_args()
    result = run(max_commits=args.max_commits)
    print(json.dumps(result, indent=2))
