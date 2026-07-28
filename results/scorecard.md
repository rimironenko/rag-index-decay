# Scorecard - decayed RAG index, T0 → 13 months of real git churn

Corpus: GitLab Handbook, T0 commit `d9dfa4eb7a` (2025-06-02) → HEAD `6a7e263deb`
(2026-07-10), 4,742 first-parent commits walked. Identical embeddings
(BGE-M3, fp32, seed 42) upserted into pgvector (HNSW/cosine), Qdrant
(cosine), and Chroma (cosine), all on documented defaults. Full raw data: [`results/t0_baseline.json`](https://github.com/rimironenko/rag-index-decay/blob/main/results/t0_baseline.json), [`results/t1_post_churn.json`](https://github.com/rimironenko/rag-index-decay/blob/main/results/t1_post_churn.json), [`results/t2_audit.json`](https://github.com/rimironenko/rag-index-decay/blob/main/results/t2_audit.json). These files contain counts, percentages and per-engine stats - not chunk-level finding IDs. The chunk-level ground truth lives in the published ledger
([`ledger.db.gz`](https://github.com/rimironenko/rag-index-decay/releases/latest)).

| Check | pgvector | Qdrant | Chroma | Ground-truth (ledger) | Audit precision/recall |
|---|---|---|---|---|---|
| Chunks indexed at T0 | 33,286 | 33,286 | 33,286 | 33,286 | — |
| Chunks after 13mo churn (live) | 166,947 | 166,947 | 166,947 | 287,346 ever logged | — |
| Stale chunks (behind source) | 116,798 (69.96%) | 116,798 (69.96%) | 116,798 (69.96%) | 116,798 (69.96%) | 1.0 / 1.0 |
| Orphaned chunks (dead source) | 131,241 (78.61%) | 131,241 (78.61%) | 131,241 (78.61%) | 131,241 (78.61%) | 1.0 / 1.0 |
| Near-duplicate chunks (≥0.98) | 127,061 (76.11%) | 127,448 (76.34%) | 127,498 (76.37%) | 111,758 exact-hash confirmed | exact-hash 1.0 / 1.0; cosine-ANN qualitative (no ledger ground truth for near-but-not-identical content) |
| **Deleted docs still in top-5 / top-10** | 0/637, 0/637 | 0/637, 0/637 | 0/637, 0/637 | 0 expected | n/a (direct count, cross-validated via control check) |
| Erased synthetic-PII still retrievable (top-5 / top-10) | 0/40, 0/40 | 0/40, 0/40 | 0/40, 0/40 | 0 expected | n/a (direct count, cross-validated via control check) |
| Storage-layer persistence (deleted docs + PII, combined) | 0/677 | 0/677 | 0/677 | 0 expected | — |
| Retrievability-detection control check (self-match pass rate) | 195/200 (97.5%) | 196/200 (98.0%) | 197/200 (98.5%) | — | — |
| Dead/deleted vectors on disk | **254,060 dead tuples at T1** (post-churn, captured during the 2026-07-15 churn replay) → **0 by T2** (this audit): PostgreSQL's autovacuum reclaimed them on its own schedule (`autovacuum_count`=3, last run 2026-07-16T16:37:57Z) - no manual VACUUM was ever run (`manual_vacuum_ever_run`=false) | not directly measurable server-side (`deleted_threshold`=0.2, `vacuum_min_vector_number`=1000 are optimizer settings, not a live count) | HNSW dir: ~2.1GB (2,117,865,666 bytes) for 166,947 live vectors | - | - |

(**Staleness denominator.** 69.96% is against the full 166,947 live chunks,
for consistency with the orphan and near-duplicate rows. Against the narrower
population of 152,504 live chunks whose source document still exists, the same
116,798 chunks are 76.59%. The count is identical; only the denominator
differs. Both figures appear in the write-up.)

**Duplicates methodology note:** the 127k/76% figures are exact-hash
(111,758, identical across engines, 100% precision by construction) plus a
non-overlapping cosine-ANN pass on the remainder. That remainder is
**not** `166,947 − 111,758 = 55,189` - the exact-hash pass also excludes
each cluster's one "kept" canonical copy from the ANN candidate pool (to
avoid re-flagging the same duplicates a second time at cosine=1.0), and
there are 36,761 such clusters. The reproducible arithmetic is 166,947 - 111,758 (excess duplicates) - 36,761 (kept cluster originals) = 18,428-chunk remainder. Three different counts appear against this remainder in the raw results, and they are not interchangeable: `cosine_ann_flagged` (15,303–15,740) is the number of chunks flagged near-duplicate at >=0.98 and is what's added to exact_hash_excess to produce the total_bloat / ~76% figure reported above. `cosine_ann_additional_chunk_ids_count` (18,076–18,496) is a separate, larger count from an earlier candidate-generation pass and is not used in any headline claim in this scorecard or the blog post. Only `cosine_ann_flagged` should be cited publicly as "near-duplicates found by the ANN pass". The other figure is retained in `t2_audit.json` for audit-methodology review only.

**Index growth:** 33,286 chunks at T0 → 166,947 live at HEAD, a a 5.0x increase (a clean pipeline would hold 35,706 - the corpus itself grew
7.3%), out of 287,346 chunk rows ever logged across the replay.

**Growth gap = orphan set.** The 131,241 gap between live (166,947) and
clean-pipeline (35,706) is not just numerically equal to the orphan count -
it is the same `chunk_id` set, cause for cause (4,723 deleted + 9,720 renamed +
116,798 superseded), identical on all three engines. Within it, 97,761
chunks are skip-flagged in the ledger (a delete should have been issued and
never was - failure modes A/D); the other 33,480 carry no later ledger event
at all - upserted once, never touched again, source doc moved on afterwards.

**Deleted-doc count note:** the deletion-retrievability check runs against 637 docs (677 total ledger-deleted minus the 40 synthetic-PII erasures, tracked separately). This is lower than `churn_counters.delete` (649) in `t1_post_churn.json`. The 12-doc gap is docs that were deleted and later re-added under `add_from_stub` (27 such events), which nets out differently depending on whether a doc's final state or its total delete events is being counted. The retrievability check uses final state - a doc counts as 'deleted' only if it is absent at HEAD - which is the correct population for a leak check.

**The real headline: how much of the index fails at least one check.**
Stale, orphaned, and duplicate chunks overlap heavily - `stale_chunk_ids`
is a strict *subset* of `orphan_chunk_ids` by construction (identical predicate - staleness *is* the `"superseded_live_doc"` cause of
orphan status; the check implementations are not in this repo, see the
README's "What's published here vs. what isn't"), so the three rows above are not independent
and summing them is not defensible. The actual union, computed via
`chunk_id` set operations against the ledger (not re-derived heuristics), with
the resulting counts committed in [`results/t2_audit.json`](../results/t2_audit.json)'s
`chunk_set_union_analysis`:

- **Conservative (exact-hash duplicates only): 149,800 of 166,947 live
  chunks fail at least one check - 89.73%.** Identical across all three
  engines (pure ledger joins for stale/orphaned; exact-hash duplicates are
  also engine-independent).
- **Including the cosine-ANN near-duplicate layer** (no ledger ground
  truth for this layer - reported as an upper bound, not a confirmed
  count): 152,418–152,479 chunks, 91.30–91.33% depending on engine.
- **The complement - chunks passing every check:** 17,147 (10.27%)
  conservative, 14,468–14,529 (8.67–8.70%) including cosine-ANN.
- Full 7-region Venn breakdown (stale-only, orphan-only, dup-only, and all
  four overlaps - `stale_only` reads 0 in every engine by construction,
  per the subset relationship above) is in the same JSON block, per engine,
  for both duplicate definitions.

Note this union is *higher* than any individual row's percentage, not
lower - the naive "76–79%" range quoted from the individual rows
undersold the real headline, because the duplicate set contributes real
non-overlapping content beyond what staleness/orphans already cover, even
though staleness itself contributes nothing beyond orphans.

**Dead/deleted-vectors row is intentionally asymmetric across engines** -
each engine exposes a different (or no) way to measure this server-side;
reporting a fabricated number for Qdrant would be worse than reporting
none.
