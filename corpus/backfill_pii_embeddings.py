"""One-time data repair: the PII-overlay migration
(churn/pii_overlay.py's migrate_pii_overlay_to_per_employee) called
ingest_baseline.embed_all() for its 100 per-employee chunks, but those
embeddings were absent from the current corpus/_cache/embeddings/ shard
cache - likely lost in an earlier shard-pruning pass whose retention
target predated the +100 employee chunks the PII migration added. This
blocks any downstream check that needs each erased employee's own original
vector as the worst-case query for a deletion/erasure probe.

Re-embedding requires the *exact original* employee text. Faker is
unpinned in pyproject.toml, so regenerating via generate_fake_employees()
now (verified during this repair) reproduces DIFFERENT records than the
ones originally embedded - the installed Faker version has drifted since
the 2026-07-13 migration. The one artifact that still holds the exact
original 100 records is the already-committed
results/synthetic_pii_manifest.json (written by write_manifest() at
migration time) - this script reconstructs each employee's markdown from
that manifest instead of regenerating and verifies every reconstruction's
sha256 against the ledger's doc_versions.content_sha256 before embedding,
so a verification failure aborts loudly rather than silently caching wrong
vectors under real chunk_ids.

Usage:
    python -m corpus.backfill_pii_embeddings
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from churn.pii_overlay import DOC_ID_PREFIX, FakeEmployee, render_employee_profile_md
from corpus import clean as clean_mod
from corpus import ingest_baseline
from ledger import ledger

MANIFEST_PATH = Path(__file__).parent.parent / "results" / "synthetic_pii_manifest.json"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run(ledger_path: Path = ingest_baseline.LEDGER_PATH, manifest_path: Path = MANIFEST_PATH) -> dict:
    conn = ledger.get_conn(ledger_path)
    manifest = json.loads(manifest_path.read_text())

    chunk_records: list[tuple[str, str]] = []
    verified_doc_count = 0
    verified_chunk_count = 0
    mismatches: list[dict] = []

    for record in manifest["employees"]:
        emp = FakeEmployee(**{k: v for k, v in record.items() if k != "chunk_id"})
        raw_md = render_employee_profile_md(emp)
        cleaned = clean_mod.clean_text(raw_md, f"synthetic:employee/{emp.employee_id}.md")
        assert cleaned is not None, f"{emp.employee_id}: reconstructed profile treated as a stub"
        reconstructed_hash = sha256(cleaned.body)

        cur = conn.execute(
            "SELECT version_id, content_sha256 FROM doc_versions WHERE doc_id = ? ORDER BY version_id",
            (f"{DOC_ID_PREFIX}/{emp.employee_id}",),
        )
        version_rows = cur.fetchall()
        if not version_rows:
            mismatches.append({"employee_id": emp.employee_id, "reason": "no doc_versions rows found"})
            continue
        verified_doc_count += 1

        for version_id, content_sha256 in version_rows:
            if content_sha256 != reconstructed_hash:
                mismatches.append(
                    {
                        "employee_id": emp.employee_id,
                        "version_id": version_id,
                        "reason": "content_sha256 mismatch",
                        "ledger_hash": content_sha256,
                        "reconstructed_hash": reconstructed_hash,
                    }
                )
                continue
            chunk_cur = conn.execute(
                "SELECT chunk_id FROM chunks WHERE version_id = ?", (version_id,)
            )
            for (chunk_id,) in chunk_cur.fetchall():
                chunk_records.append((chunk_id, cleaned.body))
                verified_chunk_count += 1

    if mismatches:
        raise RuntimeError(
            f"PII embedding backfill aborted: {len(mismatches)} verification failures "
            f"(manifest reconstruction didn't match ledger content_sha256). "
            f"Refusing to cache embeddings under unverified chunk_ids. Details: {mismatches[:5]}"
        )

    embeddings = ingest_baseline.embed_all(chunk_records)
    missing = [cid for cid, _ in chunk_records if cid not in embeddings]

    return {
        "manifest_employee_count": len(manifest["employees"]),
        "verified_doc_count": verified_doc_count,
        "verified_chunk_count": verified_chunk_count,
        "embedded_or_already_cached": len(chunk_records) - len(missing),
        "still_missing_after_embed": missing,
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
