# nanorag: useful local knowledge-base assistant

Goal: make the existing offline RAG engine useful for searching a changing
Markdown/text support knowledge base, with reproducible correctness checks.
This builds on the account's AI engineering focus and existing retrieval work.

## Phase 1 — Document update correctness (2026-10-06)

Completed source-scoped replacement, collision-resistant storage IDs based on
resolved paths, empty-file clearing, missing-path errors, and keyword-index
refresh when content changes without changing the chunk count. Unchanged files
skip embedding. An embedding failure preserves that file's previous passages.

Validation: 31 tests passed, including six new document-update regressions.
The bundled 20-question synthetic corpus produced recall@5 1.0, MRR 0.975,
and citation rate 1.0. These small fixture results are regression checks,
not evidence of production accuracy or real-world user impact.

## Next phases

2. Add explicit deletion/synchronization scoped to one corpus, with a preview
   of removals and tests protecting other indexed directories. Preserve manual
   ingestion semantics; do not silently delete unrelated sources.
3. Add an original practical support-document sample and labelled questions,
   including unanswerable questions, to evaluate usefulness beyond the current
   fictional demo corpus. Add reliable evaluation thresholds and CI.
4. Add a local browser interface for search, cited answers, and source inspection.
5. Document deployment, performance measurements, limits, and a reproducible demo.

## Engineering tradeoffs to explain in interviews

- Content-based chunk deduplication does not remove superseded content; document
  identity and document replacement are separate responsibilities.
- Keyword indexes must refresh on content changes, not just size changes.
- Preparing embeddings before replacement avoids destructive per-file failure.
- Absolute source paths are simple for local use but require explicit handling
  of moved corpora and do not provide portable document identity.

Potential CV wording after reviewing and understanding the implementation:
"Extended an offline Python RAG engine with source-aware document replacement
and regression coverage for stale retrieval, identity collisions, and failed
embedding updates; validated with 31 tests and a 20-question retrieval suite."
