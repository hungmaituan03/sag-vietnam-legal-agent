#!/usr/bin/env python3
"""Run hybrid → Voyage → SAG (+ optional LLM draft) on the finance JSON.

Requires:
  - data/raw/uts_vlc_processed.json
  - VOYAGE_API_KEY in .env (real rerank; no fake client)
  - sentence-transformers (dense stage uses the multilingual MiniLM)
  - QWEN_API_KEY in .env when using --qwen / --qwen-all / --answer

Usage (repo root, venv on):
  python scripts/run_finance_retrieval.py
  python scripts/run_finance_retrieval.py --query "kỳ kế toán năm" --qwen
  python scripts/run_finance_retrieval.py --query "kỳ kế toán năm" --answer
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sag_legal.generation import generate_draft
from sag_legal.ingestion import FINANCE_DOC_IDS, flatten_chunks, ingest_corpus
from sag_legal.models import LegalChunk
from sag_legal.reranking import rerank
from sag_legal.retrieval.bm25 import ScoreChunk, search_bm25
from sag_legal.retrieval.embeddings import embed_chunks
from sag_legal.retrieval.hybrid import search_hybrid
from sag_legal.sag import (
    ChunkExtraction,
    build_index,
    concept_keys_by_chunk,
    expand,
    extract_chunks,
)
from sag_legal.settings import get_settings

ROOT = Path(__file__).resolve().parents[1]
RAW_JSON = ROOT / "data" / "raw" / "uts_vlc_processed.json"
CACHE_NPZ = ROOT / "data" / "processed" / "finance_embeddings.npz"
QWEN_CACHE = ROOT / "data" / "processed" / "qwen_extract_finance.json"
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


def print_context(
    title: str,
    chunks: list[LegalChunk],
    seed_ids: set[str],
    titles: dict[str, str],
) -> None:
    """SAG output has no score — these chunks were never ranked against the query."""
    print(f"\n=== {title} ({len(chunks)} chunks) ===")
    for chunk in chunks:
        tag = "SEED" if chunk.chunk_id in seed_ids else "+SAG"
        law = titles.get(chunk.document_id, chunk.document_id)
        preview = chunk.text.replace("\n", " ")[:80]
        print(f"  {tag} | {law} | {chunk.citation_path} | {preview}")


def print_extractions(
    title: str,
    seeds: list[LegalChunk],
    extractions: dict[str, ChunkExtraction],
    titles: dict[str, str],
) -> None:
    print(f"\n=== {title} ===")
    for chunk in seeds:
        extraction = extractions.get(chunk.chunk_id)
        law = titles.get(chunk.document_id, chunk.document_id)
        print(f"\n  {law} | {chunk.citation_path}")
        if extraction is None:
            print("    (not extracted)")
            continue
        for event in extraction.events:
            print(
                f"    event: {event.summary} "
                f"[actor={event.actor}, action={event.action}, "
                f"modality={event.modality}, trigger={event.trigger}]"
            )
        for entity in extraction.entities:
            print(f"    entity: {entity.name} ({entity.type})")
        print(f"    concept keys: {extraction.concept_keys()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finance corpus: BM25 → dense → hybrid → live Voyage → SAG"
    )
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument(
        "--hybrid-k",
        type=int,
        default=20,
        help="Hybrid shortlist size passed to Voyage (keep small for token budget).",
    )
    parser.add_argument("--voyage-k", type=int, default=5)
    parser.add_argument(
        "--sag-extra",
        type=int,
        default=10,
        help="Max chunks SAG may add behind the reranked seeds.",
    )
    parser.add_argument("--hops", type=int, default=1, help="Frontier depth.")
    parser.add_argument(
        "--min-sim",
        type=float,
        default=0.6,
        help="Cosine floor for a semantic edge.",
    )
    parser.add_argument(
        "--no-sag",
        action="store_true",
        help="Stop after Voyage — the ablation baseline.",
    )
    parser.add_argument(
        "--qwen",
        action="store_true",
        help=(
            "Extract events/entities with Qwen on the hybrid shortlist "
            "(cached), then expand via concept:: joins as well."
        ),
    )
    parser.add_argument(
        "--qwen-all",
        action="store_true",
        help=(
            "Extract the whole finance pack with Qwen (slow first run, then "
            "cached). Implies --qwen."
        ),
    )
    parser.add_argument(
        "--answer",
        action="store_true",
        help=(
            "After SAG (or Voyage with --no-sag), call Qwen to draft a "
            "Vietnamese answer from the evidence pack."
        ),
    )
    args = parser.parse_args()
    use_qwen = args.qwen or args.qwen_all
    need_qwen = use_qwen or args.answer

    settings = get_settings()
    if not settings.voyage_configured:
        raise SystemExit(
            "VOYAGE_API_KEY is not set. Put it in .env at the repo root."
        )
    if need_qwen and not settings.qwen_configured:
        raise SystemExit(
            "QWEN_API_KEY is not set. Put it in .env, or drop "
            "--qwen / --qwen-all / --answer."
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
    print("Voyage: live API", flush=True)
    if use_qwen:
        scope = "full corpus" if args.qwen_all else "hybrid shortlist"
        print(f"Qwen: {settings.qwen_model} on {scope}", flush=True)

    print("Embedding corpus (cached after the first run)…", flush=True)
    vectors = embed_chunks(chunks, cache_path=CACHE_NPZ)

    bm25_hits = search_bm25(args.query, chunks, k=5)
    print_hits("1) BM25", bm25_hits, titles)

    hybrid_hits = search_hybrid(args.query, chunks, k=args.hybrid_k, vectors=vectors)
    print_hits("2) HYBRID RRF (shortlist → Voyage)", hybrid_hits, titles)

    voyage_hits = rerank(
        args.query,
        [hit.chunk for hit in hybrid_hits],
        top_k=args.voyage_k,
    )
    print_hits("3) VOYAGE (live rerank-2.5)", voyage_hits, titles)

    evidence: list[LegalChunk]
    if args.no_sag:
        evidence = [hit.chunk for hit in voyage_hits]
        print("\n(--no-sag): Voyage shortlist is the whole context.")
    else:
        concepts: dict[str, list[str]] | None = None
        if use_qwen:
            to_extract = (
                chunks if args.qwen_all else [hit.chunk for hit in hybrid_hits]
            )
            print(
                f"\nQwen extracting {len(to_extract)} chunks "
                f"(cache: {QWEN_CACHE.name})…",
                flush=True,
            )
            extractions = extract_chunks(to_extract, cache_path=QWEN_CACHE)
            concepts = concept_keys_by_chunk(extractions)
            print_extractions(
                f"3.5) QWEN extract ({settings.qwen_model})",
                [hit.chunk for hit in voyage_hits],
                extractions,
                titles,
            )
            concept_entity_count = len(
                {k for keys in concepts.values() for k in keys}
            )
            print(
                f"\n  concept keys on extracted chunks: "
                f"{concept_entity_count} unique",
                flush=True,
            )

        index = build_index(
            chunks, vectors=vectors, min_sim=args.min_sim, concepts=concepts
        )
        concept_buckets = sum(
            1 for key in index.events_by_entity if key.startswith("concept::")
        )
        print(
            f"\nSAG index: {len(index.events_by_id)} events, "
            f"{len(index.events_by_entity)} entities "
            f"({concept_buckets} concept), "
            f"{sum(len(v) for v in index.neighbours.values())} semantic edges"
        )

        seeds = [hit.chunk for hit in voyage_hits]
        seed_ids = {chunk.chunk_id for chunk in seeds}
        evidence = expand(
            seeds,
            index,
            max_extra=args.sag_extra,
            hops=args.hops,
            min_sim=args.min_sim,
            use_concepts=use_qwen,
        )
        stage = (
            "4) SAG expansion (+Qwen concepts)" if use_qwen else "4) SAG expansion"
        )
        print_context(stage, evidence, seed_ids, titles)

        added = [c for c in evidence if c.chunk_id not in seed_ids]
        cross = {c.document_id for c in added} - {c.document_id for c in seeds}
        print(
            f"\n{len(seeds)} seeds → {len(evidence)} context chunks "
            f"({len(added)} added)"
        )
        print(f"Laws reached only through SAG: {sorted(cross) or 'none'}")

    if not args.answer:
        return

    print(
        f"\n=== 5) LLM DRAFT ({settings.qwen_model}, "
        f"{len(evidence)} evidence chunks) ===",
        flush=True,
    )
    draft = generate_draft(args.query, evidence)
    status = "ABSTAINED" if draft.abstained else "ANSWER"
    print(f"\n[{status}]\n{draft.answer}\n")
    if draft.cited_chunk_ids:
        print("Cited chunk_ids:")
        for cid in draft.cited_chunk_ids:
            print(f"  - {cid}")


if __name__ == "__main__":
    main()
