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

## Phase 2 — Scoped deletion reconciliation (2026-10-09)

Completed `prune <directory>` with preview by default and explicit `--apply`.
The operation removes indexed passages only for missing absolute sources under
an existing selected directory. It preserves sibling corpora, existing paths,
and legacy relative identities. Missing roots and inspection errors abort before
mutation. Both retrieval indexes are staged, and the persisted save completes
before live state is replaced. No document files are deleted.

Validation: 41 tests passed, including ten new pruning regression tests covering
preview, apply, path boundaries, reload, missing roots, inspection and save
failures, relative legacy identities, nested removal, recreation, and the CLI.
The bundled 20-question evaluation remains recall@5 1.0, MRR 0.975, citation
rate 1.0. These are fixture results, not production accuracy claims.

Limitations: no concurrent writers or filesystem mutations during apply;
preview and apply are separate inspections. An existing empty mount is
indistinguishable from intentional deletion, so the preview must be reviewed.

## Next phases

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
- Staging retrieval indexes before a disk save keeps the live state consistent
  if persistence fails; deletion previews expose what a reconciliation will do.
- Absolute source paths are simple for local use but require explicit handling
  of moved corpora and do not provide portable document identity.

Potential CV wording after reviewing and understanding the implementation:
"Extended an offline Python RAG engine with document replacement and scoped
deletion reconciliation, including preview mode and failure-safe index updates;
validated with 41 tests and a 20-question retrieval regression suite."
