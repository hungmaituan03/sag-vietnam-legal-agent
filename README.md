# SAG Vietnam Legal Agent

Internal research/prototype: **Vietnamese legal information agent** using SQL-Retrieval Augmented Generation (SAG), hybrid retrieval, Voyage AI reranking, and a mandatory Hindsight self-verification layer.

> Not a lawyer replacement. Answers must be citation-backed, temporally aware, and honest about uncertainty.

## Repo layout


| Path                 | Purpose                                                               |
| -------------------- | --------------------------------------------------------------------- |
| `src/sag_legal/`     | Implementation (ingestion → retrieval → SAG → generation → Hindsight) |
| `configs/`           | Experiment / runtime configs (no secrets)                             |
| `data/`              | Raw + processed corpora (see `data/README.md`)                        |
| `tests/`             | Unit + integration tests                                              |
| `docs/process/`      | How we work in public (commits, PRs, weekly cadence)                  |
| `docs/architecture/` | Architecture notes as they evolve                                     |
| `../SAG-docs/`       | Optional sibling folder for long-form notebooks/reports/demos         |




## Pipeline (mandatory order)

```text
Query → Hybrid (BM25 + dense) → Voyage reranker → SAG multi-hop
     → LLM draft → Hindsight → Citation validation → Answer
```



## Quick start (local)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # fill keys locally — never commit .env
pytest
```



### Chat UI (optional)

```bash
pip install -e ".[web,dev]"
python -m uvicorn sag_legal.chat.app:app --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000
```

Needs `VOYAGE_API_KEY`, `OPENAI_API_KEY`, and `data/raw/uts_vlc_processed.json`.  
First run rebuilds `data/processed/khung1_embeddings.npz` for the expanded Khung 1 pack (~14 laws).

## Research question

Can **SAG + hybrid retrieval + Voyage reranking + Hindsight** produce more accurate, traceable, reliable Vietnamese legal answers than conventional RAG?