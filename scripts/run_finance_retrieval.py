#!/usr/bin/env python3
"""Run hybrid → Voyage on the approved finance JSON (live, not a smoke pack).

Requires:
  - data/raw/uts_vlc_processed.json
  - VOYAGE_API_KEY in .env (real rerank; no fake client)
  - sentence-transformers (dense stage uses the multilingual MiniLM)

Usage (repo root, venv on):
  python scripts/run_finance_retrieval.py
  python scripts/run_finance_retrieval.py --query "kỳ kế toán năm"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sag_legal.ingestion import FINANCE_DOC_IDS, flatten_chunks, ingest_corpus
from sag_legal.reranking import rerank
from sag_legal.retrieval.bm25 import ScoreChunk, search_bm25
from sag_legal.retrieval.hybrid import search_hybrid
from sag_legal.settings import get_settings

ROOT = Path(__file__).resolve().parents[1]
RAW_JSON = ROOT / "data" / "raw" / "uts_vlc_processed.json"
DEFAULT_QUERY = "cấp giấy phép thành lập tổ chức tín dụng"


def print_hits(title: str, hits: list[ScoreChunk], titles: dict[str, str]) -> None:
    print(f"\n=== {title} ({len(hits)} hits) ===")
    if not hits:
        print("  (none)")
        return
    for i, hit in enumerate(hits, start=1):
        preview = hit.chunk.text.replace("\n", " ")[:80]
        law = titles.get(hit.chunk.document_id, hit.chunk.document_id)
        eff = hit.chunk.effective_date or "unknown"
        print(
            f"  {i}. score={hit.score:.4f} | {law} | {hit.chunk.citation_path} | "
            f"eff={eff} | {preview}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finance corpus: BM25 → dense → hybrid → live Voyage"
    )
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument(
        "--hybrid-k",
        type=int,
        default=20,
        help="Hybrid shortlist size passed to Voyage (keep small for token budget).",
    )
    parser.add_argument("--voyage-k", type=int, default=5)
    args = parser.parse_args()

    settings = get_settings()
    if not settings.voyage_configured:
        raise SystemExit(
            "VOYAGE_API_KEY is not set. Put it in .env at the repo root."
        )
    if not RAW_JSON.is_file():
        raise SystemExit(f"Missing corpus: {RAW_JSON}")

    print("Ingesting finance pack from", RAW_JSON.name)
    results = ingest_corpus(RAW_JSON, doc_ids=FINANCE_DOC_IDS)
    titles: dict[str, str] = {}
    for result in results:
        doc = result.document
        titles[doc.document_id] = doc.title
        eff = doc.effective_date.isoformat() if doc.effective_date else "unknown"
        print(
            f"  {doc.document_id}: {len(result.chunks)} chunks | "
            f"effective_date={eff}"
        )

    chunks = flatten_chunks(results)
    print(f"CHUNKS: {len(chunks)}")
    print(f"QUERY: {args.query}")
    print("Dense: real MiniLM inside hybrid (first run downloads the model).")
    print("Voyage: live API", flush=True)

    bm25_hits = search_bm25(args.query, chunks, k=5)
    print_hits("1) BM25", bm25_hits, titles)

    print(
        "Embedding full corpus for hybrid dense (this can take a few minutes)…",
        flush=True,
    )
    hybrid_hits = search_hybrid(args.query, chunks, k=args.hybrid_k)
    print_hits("2) HYBRID RRF (shortlist → Voyage)", hybrid_hits, titles)

    voyage_hits = rerank(
        args.query,
        [hit.chunk for hit in hybrid_hits],
        top_k=args.voyage_k,
    )
    print_hits("3) VOYAGE (live rerank-2.5)", voyage_hits, titles)
    print("\nDone. Evidence above is the Voyage shortlist for SAG next.")


if __name__ == "__main__":
    main()
