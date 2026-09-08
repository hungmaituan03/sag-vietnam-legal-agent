# Architecture — living notes

## Current stage

`week01-scaffold` — repository and process only. No retrieval agent yet.

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
