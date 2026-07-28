# rag-index-decay

A reproducible "decayed RAG index" demo: replay a real 13-month git history
(the GitLab Handbook, CC BY-SA 4.0) through a synthetic-PII overlay, index it
into pgvector, Qdrant and Chroma under an identical, pinned fairness
protocol, and publish the raw ground-truth state that results from letting
real ingestion churn (updates, deletes, renames, re-ingests) run unchecked.

## What this is - and isn't

This repo is the **corpus/churn/ledger/repro harness** behind a public
teardown write-up on decayed RAG indexes. It reproduces the exact
multi-engine index state the write-up's findings were computed against, and
publishes the raw findings (`results/*.json`) for independent inspection.

**It is not the audit tool itself.** The check algorithms that turned this
index state into findings (staleness/orphan/duplicate/retrievability
detection, the GDPR-style deleted-but-retrievable probe, the ledger-based
precision/recall scorecard) are not part of this repo. Two related, separate
projects:

- [`rag-staleness-check`](https://github.com/rimironenko/rag-staleness-check)
  — an open-source (Apache-2.0), read-only CLI for running a subset of these
  checks against your own single-engine index.
- [ragproof.io](https://ragproof.io) - the paid, multi-engine audit (GDPR
  erasure verification, cross-engine orchestration, the full ledger-based
  precision/recall harness) that this exact methodology backs.

**Disclosure:** I work on productizing this, so I have an interest in index decay being a real problem.

## Fairness protocol

- **Identical vectors, computed once.** `corpus/ingest_baseline.py`'s
  `embed_all()` embeds every chunk exactly once; `upsert_all_engines()` then
  upserts the unmodified `(chunk_id, vector)` pairs into all three engines -
  no engine gets a separately-computed embedding.
- **Pinned image digests** (not bare tags):

  | Service  | Image:tag                | Digest                                                                    |
  | -------- | ------------------------ | ------------------------------------------------------------------------- |
  | pgvector | `pgvector/pgvector:pg17` | `sha256:d2ef61f42ef767baa5a1475393303cc235bcd92febd9d7014eddb48b41f3bad0` |
  | qdrant   | `qdrant/qdrant:v1.15.5`  | `sha256:0fb8897412abc81d1c0430a899b9a81eb8328aa634e7242d1bc804c1fe8fe863` |
  | chroma   | `chromadb/chroma:0.6.3`  | `sha256:e0e78dc7609a599b63c99753442c7d01b1d3d369ce0e3bf3e0540536fec4fa7a` |

- **Pinned embedding model.** `BAAI/bge-m3` (MIT) via
  `sentence-transformers==5.6.0`, pinned to HF revision
  `5617a9f61b028005a4858fdac845db406aefb181` (see `corpus/chunk.py`).
- **Defaults, not tuned - and disclosed as such.** All three engines
  (`engines/pgvector_adapter.py`, `qdrant_adapter.py`, `chroma_adapter.py`)
  run bare HNSW server defaults with cosine distance; each adapter's
  `ensure_schema`/`ensure_collection` carries a comment saying so, rather
  than silently tuning one engine and not the others.
- **No cross-engine speed claims.** Nothing in this repo or its published
  results makes a timing/latency/QPS comparison between engines.

## Limitations

- **One corpus.** These are the GitLab Handbook's specific 13 months of edit
  patterns — a different corpus with a different churn shape could look
  meaningfully better or worse. Corpus content is CC BY-SA 4.0 per GitLab's
  own [handbook-license blog post](https://about.gitlab.com/blog/our-handbook-is-open-source-heres-why/)
  (not the handbook repo's own boilerplate `LICENSE` file, which is silent
  on content licensing) - it's cloned fresh at repro time
  (`corpus/acquire.py`), not vendored into this repo.
- **Synthetic PII is clearly labeled.** The 100 fake employees
  (`synthetic-pii/employee/SYN-XXXX`) are generated with Faker (seed=42),
  isolated in their own doc_id namespace, carry an in-document "SYNTHETIC
  DATA NOTICE" and are flagged via a dedicated `is_synthetic_pii` ledger
  column - never mixed with real handbook attribution. See
  `results/synthetic_pii_manifest.json`.
- **HNSW build-order nondeterminism is mitigated, not eliminated.** Every
  adapter's `upsert()` sorts pairs by `chunk_id` before insertion and the
  master seed is fixed (`seeds.py`), but single-digit rerun variance in
  HNSW-dependent counts is still possible.
- **Not a speed benchmark.**

## What's published here vs. what isn't

`results/*.json` contains the raw findings from running check algorithms
against this exact index state - chunk/doc counts, staleness/orphan/
duplicate/retrievability figures, per-engine stats. The check algorithms
themselves aren't in this repo; they're what the paid audit product and the
OSS CLI (linked above) separately implement. This repo's job is to make the
*index state those checks ran against* fully reproducible and independently
inspectable.

You don't have to trust the private code to check the findings. The
populated ground-truth ledger is published as a release asset
([`ledger.db.gz`](https://github.com/rimironenko/rag-index-decay/releases/tag/ledger-db-v1)) - too large for git, but byte-identical to
what `python -m churn.replay` rebuilds. `index_events` records which chunks
should be present in which engine after every operation, `erasure_requests`
records what should be gone after the synthetic-PII erasure step (40
employees x 3 engines = 120 rows, all `expected_absent=1`), and
`chunks`/`doc_versions` carry the content hashes and commit SHAs behind
every predicate. Every headline number in `results/t2_audit.json` is
re-derivable from that file with SQL.

One thing `t2_audit.json` does *not* contain is chunk-level finding IDs -
it publishes counts and percentages, not the 116,798 individual `chunk_id`
values behind the staleness row. The ledger is where the chunk-level
ground truth lives, which is why it's published.

To be clear about what the OSS CLI is *not*: a way to corroborate
`t2_audit.json`. It implements the same four check types, but
single-engine and manifest-based - a narrower slice by design. There's no
packaged way to point it at this repro index, and if you wired that up
yourself it would report smaller numbers. That's the subset semantics,
not a contradiction.

## Reproduce

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
docker compose up -d          # starts pgvector, Qdrant, Chroma
python -m corpus.ingest_baseline   # T0 baseline: clone + chunk + embed + upsert (hours on CPU; --pilot 50 for a quick check)
python -m churn.replay        # T0 -> HEAD churn replay: full ledger + index_events (--max-commits 100 for a smoke test)
```

Both steps are resumable/idempotent - safe to interrupt and rerun. This
reproduces the identical decayed index state (same doc/chunk counts, same
per-engine live counts) reflected in `results/t0_baseline.json` and
`results/t1_post_churn.json`. The numbers in `results/t2_audit.json` come from the private/paid tooling
linked above - the raw output is published here, not the code that
produced it. You can still check those numbers, though: the published ledger carries
the expected state of every chunk, so every finding is re-derivable
against ground truth without the check code (see "What's published here
vs. what isn't" above).

Raw findings: `results/t0_baseline.json`, `results/t1_post_churn.json`,
`results/t2_audit.json`.

## Contributing

Corrections and PRs welcome - especially anything that improves reproducing
this exact index state or clarifies the fairness protocol above.

## Licensing

- **Code in this repo:** Apache-2.0 (see `LICENSE`).
- **GitLab Handbook corpus:** CC BY-SA 4.0, cloned from
  `https://gitlab.com/gitlab-com/content-sites/handbook.git` at repro time
  (not vendored here) - see "Limitations" above for the attribution source.
- **`BAAI/bge-m3`** embedding model: MIT.
- **`rag-staleness-check`** OSS CLI: [separate repo](https://github.com/rimironenko/rag-staleness-check), also Apache-2.0.
