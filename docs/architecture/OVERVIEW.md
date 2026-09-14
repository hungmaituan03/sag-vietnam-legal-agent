# Architecture — living notes

## Current stage

`sag-v0` — the retrieval spine runs end to end on the approved corpus, and SAG
expands the reranked shortlist.

### Done
- `LegalDocument` / `LegalChunk` with provenance + temporal fields
- Regex structure-aware splitter: Điều → Khoản → Điểm
- `ingestion.corpus`: approved JSON → documents, `effective_date` from force
  clauses, finance pack selection, `flatten_chunks`
- BM25, dense, hybrid RRF (fuses ranks, not scores)
- Voyage `rerank-2.5`, live, injectable client for CI
- `retrieval.embeddings`: corpus vectors encoded once and cached on disk
- `sag.index`: event/entity index, structural + semantic edges, bounded
  multi-hop expansion

Results and limitations: `docs/experiments/SAG_REPORT.md`.

### Explicitly not done yet
- LLM entity extraction, MySQL, Elasticsearch (dicts + numpy for now)
- LLM draft generation, Hindsight, citation validation
- Temporal filtering during expansion
- Real crawlers (the corpus is a fixed approved dump)

## Target pipeline

```text
User Query
  → Query analysis / router
  → Hybrid retrieval (BM25 + dense → fuse)
  → Voyage AI reranker
  → SAG (event–entity hyperedge expansion)
  → Evidence selection
  → LLM draft
  → Hindsight (mandatory)
  → Citation validation
  → Structured answer
```

## Module map (`src/sag_legal/`)

| Package | Responsibility |
|---------|----------------|
| `ingestion` | Fetch/normalize legal docs + provenance |
| `chunking` | Structure-aware Điều/Khoản/Điểm chunks |
| `retrieval` | BM25, dense, hybrid fusion, cached corpus embeddings |
| `reranking` | Voyage client |
| `sag` | Event–entity index + hyperedge expansion |
| `hindsight` | Self-verification before user sees answer |
| `generation` | Draft answer from evidence |
| `citation` | Format/validate citations |
| `contract_review` | Constrained contract issues (later weeks) |

Update this file whenever an architectural decision lands (and link the PR).
