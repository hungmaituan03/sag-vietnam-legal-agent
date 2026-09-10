#!/usr/bin/env python3
"""Demo the retrieval pipeline for Friday / mentor walkthroughs.

Default: offline-friendly (fake dense embedder + fake Voyage).
Optional: --live-voyage to call real Voyage (needs VOYAGE_API_KEY in .env).

Usage (from repo root, venv active):
  python scripts/demo_retrieval.py
  python scripts/demo_retrieval.py --live-voyage
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from sag_legal.ingestion import ingest_text_file
from sag_legal.models import DocumentStatus, DocumentType, LegalDocument
from sag_legal.reranking import rerank
from sag_legal.retrieval.bm25 import ScoreChunk, search_bm25
from sag_legal.retrieval.dense import search_dense
from sag_legal.retrieval.hybrid import search_hybrid

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
QUERY = "cấp giấy phép"


def fake_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic vectors so the demo never downloads a local embedding model."""
    vectors: list[list[float]] = []
    for text in texts:
        lower = text.lower()
        if "phép" in lower or "giấy" in lower:
            vectors.append([1.0, 0.0])
        else:
            vectors.append([0.0, 1.0])
    return vectors


class FakeVoyageClient:
    """Offline stand-in for voyageai.Client.rerank."""

    def rerank(self, query, documents, model, top_k=None, truncation=True):
        scored = []
        for index, doc in enumerate(documents):
            lower = doc.lower()
            score = 0.9 if ("phép" in lower or "giấy" in lower) else 0.1
            scored.append(
                SimpleNamespace(index=index, document=doc, relevance_score=score)
            )
        scored.sort(key=lambda item: item.relevance_score, reverse=True)
        if top_k is not None:
            scored = scored[:top_k]
        return SimpleNamespace(results=scored)


def print_hits(title: str, hits: list[ScoreChunk]) -> None:
    print(f"\n=== {title} ({len(hits)} hits) ===")
    if not hits:
        print("  (none)")
        return
    for i, hit in enumerate(hits, start=1):
        preview = hit.chunk.text.replace("\n", " ")[:70]
        cite = hit.chunk.citation_path
        print(
            f"  {i}. score={hit.score:.4f} | {hit.chunk.chunk_id} | {cite} | {preview}"
        )


def load_fixture_chunks():
    meta = LegalDocument(
        document_id="fixture-luat-tctd-01",
        title="Luật mẫu TCTD (fixture)",
        document_type=DocumentType.LUAT,
        document_number="01/2024/QH15",
        effective_date=date(2024, 7, 1),
        status=DocumentStatus.ACTIVE,
        source_url="https://example.local/fixtures/luat-tctd",
        content="",
    )
    result = ingest_text_file(FIXTURES / "luat_tctd_fixture.txt", meta)
    return result.chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo BM25 → dense → hybrid → Voyage")
    parser.add_argument(
        "--live-voyage",
        action="store_true",
        help="Call real Voyage API (requires VOYAGE_API_KEY). Default uses a fake client.",
    )
    args = parser.parse_args()

    chunks = load_fixture_chunks()
    print("SAG Legal — retrieval demo")
    print(f"QUERY: {QUERY}")
    print(f"CHUNKS: {len(chunks)} from Luật fixture")
    print(
        "Voyage mode:",
        "LIVE API" if args.live_voyage else "FAKE client (offline demo)",
    )

    bm25_hits = search_bm25(QUERY, chunks, k=3)
    print_hits("1) BM25", bm25_hits)

    dense_hits = search_dense(QUERY, chunks, k=3, model=fake_embed)
    print_hits("2) DENSE (fake embedder)", dense_hits)

    hybrid_hits = search_hybrid(QUERY, chunks, k=3, model=fake_embed)
    print_hits("3) HYBRID RRF", hybrid_hits)

    hybrid_chunks = [hit.chunk for hit in hybrid_hits]
    if args.live_voyage:
        voyage_hits = rerank(QUERY, hybrid_chunks, top_k=2)
        voyage_title = "4) VOYAGE RERANK (live)"
    else:
        voyage_hits = rerank(
            QUERY,
            hybrid_chunks,
            top_k=2,
            client=FakeVoyageClient(),
        )
        voyage_title = "4) VOYAGE RERANK (fake)"
    print_hits(voyage_title, voyage_hits)

    print("\nDone. These top chunks are the evidence candidates for SAG / LLM next.")
    print("Narrate: BM25=words → dense=meaning → hybrid=fair merge → Voyage=pairwise rerank.")


if __name__ == "__main__":
    main()  