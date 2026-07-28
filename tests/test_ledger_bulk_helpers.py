"""Offline, in-memory tests for the bulk query helpers in ledger/ledger.py.
No Docker required - covers the highest-risk code (the set-based SQL
joins) against hand-crafted scenarios with known-correct answers, before
trusting them at 287,346-row production scale."""

from __future__ import annotations

import sqlite3

import pytest

from ledger import ledger


@pytest.fixture
def conn() -> sqlite3.Connection:
    conn = ledger.init_db(":memory:")

    # live-doc: two versions, v1 superseded by v2 -> v1's chunk is stale.
    ledger.insert_document(conn, "live-doc", "live-doc.md", "c1", "c2", status="live")
    v1 = ledger.insert_doc_version(conn, "live-doc", "c1", "2025-06-01T00:00:00+00:00", "hashLiveV1", "add")
    v2 = ledger.insert_doc_version(conn, "live-doc", "c2", "2025-12-01T00:00:00+00:00", "hashLiveV2", "update")
    ledger.insert_chunks(
        conn,
        [
            ("live-doc::v%d::0000" % v1, "live-doc", v1, 0, "hashLiveV1", 100),
            ("live-doc::v%d::0000" % v2, "live-doc", v2, 0, "hashLiveV2", 100),
        ],
    )
    # v1's chunk: correctly deleted from pgvector (upsert then delete).
    ledger.insert_index_events(
        conn,
        [
            ("pgvector", "live-doc::v%d::0000" % v1, "upsert", "2025-06-01T00:00:00+00:00", "c1", 1),
            ("pgvector", "live-doc::v%d::0000" % v1, "delete", "2025-12-01T00:00:00+00:00", "c2", 0),
            ("pgvector", "live-doc::v%d::0000" % v2, "upsert", "2025-12-01T00:00:00+00:00", "c2", 1),
        ],
    )

    # deleted-doc: one version with chunks, then a delete doc_version with NO
    # chunks - last_chunked_version_id must resolve to v1, not the (chunkless)
    # delete version.
    ledger.insert_document(conn, "deleted-doc", "deleted-doc.md", "c1", "c3", status="deleted")
    dv1 = ledger.insert_doc_version(conn, "deleted-doc", "c1", "2025-06-01T00:00:00+00:00", "hashDeleted", "add")
    ledger.insert_doc_version(conn, "deleted-doc", "c3", "2026-01-01T00:00:00+00:00", "hashDeleted", "delete")
    ledger.insert_chunks(conn, [("deleted-doc::v%d::0000" % dv1, "deleted-doc", dv1, 0, "hashDeleted", 100)])

    # old-name -> new-name rename: old-name's chunk lineage must show
    # doc_status='renamed', renamed_to='new-name'.
    ledger.insert_document(conn, "old-name", "old-name.md", "c1", "c4", status="renamed", renamed_to="new-name")
    ov1 = ledger.insert_doc_version(conn, "old-name", "c1", "2025-06-01T00:00:00+00:00", "hashOld", "add")
    ledger.insert_doc_version(conn, "old-name", "c4", "2026-02-01T00:00:00+00:00", "hashOld", "rename")
    ledger.insert_chunks(conn, [("old-name::v%d::0000" % ov1, "old-name", ov1, 0, "hashOld", 100)])

    ledger.insert_document(conn, "new-name", "new-name.md", "c4", "c4", status="live")
    nv1 = ledger.insert_doc_version(conn, "new-name", "c4", "2026-02-01T00:00:00+00:00", "hashOld", "add")
    ledger.insert_chunks(conn, [("new-name::v%d::0000" % nv1, "new-name", nv1, 0, "hashOld", 100)])

    # exact-duplicate pair: two unrelated docs sharing one text_sha256.
    ledger.insert_document(conn, "dup-a", "dup-a.md", "c1", "c1", status="live")
    da1 = ledger.insert_doc_version(conn, "dup-a", "c1", "2025-06-01T00:00:00+00:00", "hashDup", "add")
    ledger.insert_chunks(conn, [("dup-a::v%d::0000" % da1, "dup-a", da1, 0, "hashDup", 100)])
    ledger.insert_document(conn, "dup-b", "dup-b.md", "c1", "c1", status="live")
    db1 = ledger.insert_doc_version(conn, "dup-b", "c1", "2025-06-01T00:00:00+00:00", "hashDup", "add")
    ledger.insert_chunks(conn, [("dup-b::v%d::0000" % db1, "dup-b", db1, 0, "hashDup", 100)])

    # a synthetic-PII doc, erased via erasure_requests - must be excluded
    # from mode_b_deleted_doc_targets despite status='deleted'.
    ledger.insert_document(
        conn, "synthetic-pii/employee/SYN-9999", "synthetic-pii/employee/SYN-9999.md",
        "c1", "c5", status="deleted", is_synthetic_pii=1,
    )
    pv1 = ledger.insert_doc_version(conn, "synthetic-pii/employee/SYN-9999", "c1", "2025-06-01T00:00:00+00:00", "hashPII", "add")
    ledger.insert_chunks(
        conn, [("synthetic-pii/employee/SYN-9999::v%d::0000" % pv1, "synthetic-pii/employee/SYN-9999", pv1, 0, "hashPII", 100)]
    )
    ledger.insert_erasure_request(conn, "synthetic-pii/employee/SYN-9999", "c5", "pgvector", expected_absent=1)

    conn.commit()
    return conn


def test_latest_index_events_resolves_to_max_event_id(conn):
    events = ledger.latest_index_events(conn, "pgvector")
    v1_events = ledger.chunk_ids_for_version(conn, 1)
    stale_chunk_id = v1_events[0]
    assert events[stale_chunk_id]["op"] == "delete"
    assert events[stale_chunk_id]["expected_present"] == 0


def test_chunk_lineage_stale_live_doc(conn):
    lineage = ledger.chunk_lineage(conn)
    v1_chunk = ledger.chunk_ids_for_version(conn, 1)[0]
    v2_chunk = ledger.chunk_ids_for_version(conn, 2)[0]
    assert lineage[v1_chunk]["doc_status"] == "live"
    assert lineage[v1_chunk]["chunk_version_id"] != lineage[v1_chunk]["last_chunked_version_id"]
    assert lineage[v2_chunk]["chunk_version_id"] == lineage[v2_chunk]["last_chunked_version_id"]


def test_chunk_lineage_deleted_doc_resolves_to_last_chunked_version(conn):
    lineage = ledger.chunk_lineage(conn)
    deleted_chunk_id = [cid for cid in lineage if lineage[cid]["doc_id"] == "deleted-doc"][0]
    info = lineage[deleted_chunk_id]
    assert info["doc_status"] == "deleted"
    # last_chunked_version_id must be the add-version (which has chunks), not
    # the delete-version (which has none) - the core correctness detail.
    assert info["last_chunked_version_id"] == info["chunk_version_id"]


def test_chunk_lineage_renamed_doc(conn):
    lineage = ledger.chunk_lineage(conn)
    old_chunk_id = [cid for cid in lineage if lineage[cid]["doc_id"] == "old-name"][0]
    info = lineage[old_chunk_id]
    assert info["doc_status"] == "renamed"
    assert info["renamed_to"] == "new-name"


def test_all_chunk_text_sha256_exact_duplicate_pair(conn):
    sha = ledger.all_chunk_text_sha256(conn)
    dup_a = ledger.chunk_ids_for_doc(conn, "dup-a")[0]
    dup_b = ledger.chunk_ids_for_doc(conn, "dup-b")[0]
    assert sha[dup_a] == sha[dup_b] == "hashDup"


def test_last_chunked_version_id_for_doc(conn):
    deleted_v1 = ledger.chunk_ids_for_doc(conn, "deleted-doc")[0]
    version_id = ledger.chunk_lineage(conn)[deleted_v1]["chunk_version_id"]
    assert ledger.last_chunked_version_id_for_doc(conn, "deleted-doc") == version_id


def test_mode_b_deleted_doc_targets_excludes_pii(conn):
    targets = ledger.mode_b_deleted_doc_targets(conn)
    doc_ids = {t["doc_id"] for t in targets}
    assert "deleted-doc" in doc_ids
    assert "synthetic-pii/employee/SYN-9999" not in doc_ids


def test_erasure_requests_for_engine(conn):
    rows = ledger.erasure_requests_for_engine(conn, "pgvector")
    assert rows == [("synthetic-pii/employee/SYN-9999", "c5")]
    assert ledger.erasure_requests_for_engine(conn, "qdrant") == []
