# Architecture — living notes

## Current stage

`week01-schema-ingest` — legal document/chunk schemas + structure-aware chunker v0 + fixture ingest.

### Done in this stage
- `sag_legal.models.LegalDocument` / `LegalChunk` with provenance + temporal fields
- Regex structure-aware splitter: Điều → Khoản → Điểm
- `ingest_document` / `ingest_text_file` over synthetic fixtures (not official law text)
- Demo: `python scripts/demo_ingest.py`

### Explicitly not done yet
- Real crawlers / official corpus
- BM25 / dense / hybrid retrieval
- Voyage, SAG, Hindsight, Q&A agent

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
| `retrieval` | BM25, dense, hybrid fusion |
| `reranking` | Voyage client |
| `sag` | Event–entity index + hyperedge queries |
| `hindsight` | Self-verification before user sees answer |
| `generation` | Draft answer from evidence |
| `citation` | Format/validate citations |
| `contract_review` | Constrained contract issues (later weeks) |
| `evaluation` | Metrics helpers |

Update this file whenever an architectural decision lands (and link the PR).
