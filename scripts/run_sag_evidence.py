#!/usr/bin/env python3
"""Measure what SAG adds over the Voyage shortlist, on the real finance corpus.

For each query it runs the full spine twice over: the Voyage top-k alone
(the `--no-sag` baseline) and the same seeds after SAG expansion. It reports
two things the baseline cannot do:

1. orphan repair — a seed that is a điểm/khoản fragment whose parent text was
   missing from the shortlist, recovered by a structural edge;
2. cross-law reach — a chunk from a law none of the seeds came from, reached
   by a semantic edge.

Writes a markdown table to docs/experiments/sag_evidence.md.

Usage (repo root, venv on, VOYAGE_API_KEY set):
  python scripts/run_sag_evidence.py
"""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path

from sag_legal.ingestion import KHUNG1_DOC_IDS, flatten_chunks, ingest_corpus
from sag_legal.models import LegalChunk
from sag_legal.reranking import rerank
from sag_legal.retrieval.embeddings import embed_chunks
from sag_legal.retrieval.hybrid import search_hybrid
from sag_legal.sag import EventEntityIndex, build_index, expand
from sag_legal.settings import get_settings

ROOT = Path(__file__).resolve().parents[1]
RAW_JSON = ROOT / "data" / "raw" / "uts_vlc_processed.json"
CACHE_NPZ = ROOT / "data" / "processed" / "khung1_embeddings.npz"
REPORT_MD = ROOT / "docs" / "experiments" / "sag_evidence.md"

QUERIES = [
    "kỳ kế toán năm",
    "điều kiện cấp giấy phép thành lập tổ chức tín dụng",
    "kiểm kê tài sản",
    "báo cáo tài chính hằng năm",
    "nhiệm vụ và quyền hạn của Ngân hàng Nhà nước",
    "vốn điều lệ của tổ chức tín dụng",
]


def parent_id(chunk: LegalChunk) -> str | None:
    """Id of the text this fragment hangs off, or None if it stands alone."""
    if not chunk.article:
        return None
    parts = [chunk.document_id, chunk.article.replace(" ", "")]
    if chunk.point and chunk.clause:
        parts.append(chunk.clause.replace(" ", ""))
    elif not chunk.clause:
        return None
    return "::".join(parts)


def rerank_with_retry(
    query: str,
    candidates: list[LegalChunk],
    top_k: int,
    pause: float,
    attempts: int = 5,
) -> list:
    """Voyage's free tier is 3 requests and 10K tokens per minute."""
    for attempt in range(1, attempts + 1):
        try:
            return rerank(query, candidates, top_k=top_k)
        except Exception as exc:  # noqa: BLE001 - library raises its own error type
            if "rate limit" not in str(exc).lower() or attempt == attempts:
                raise
            wait = pause * attempt
            print(f"    rate limited, retrying in {wait:.0f}s…", flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def measure(
    seeds: list[LegalChunk],
    context: list[LegalChunk],
    index: EventEntityIndex,
) -> dict[str, object]:
    seed_ids = {c.chunk_id for c in seeds}
    context_ids = {c.chunk_id for c in context}
    added = [c for c in context if c.chunk_id not in seed_ids]

    orphans, repaired = 0, 0
    for seed in seeds:
        pid = parent_id(seed)
        if pid is None or pid not in index.events_by_id or pid in seed_ids:
            continue
        orphans += 1
        if pid in context_ids:
            repaired += 1

    seed_laws = {c.document_id for c in seeds}
    new_laws = sorted({c.document_id for c in added} - seed_laws)
    return {
        "seeds": len(seeds),
        "added": len(added),
        "orphans": orphans,
        "repaired": repaired,
        "new_laws": new_laws,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantify what SAG adds.")
    parser.add_argument("--hybrid-k", type=int, default=20)
    parser.add_argument("--voyage-k", type=int, default=5)
    parser.add_argument("--sag-extra", type=int, default=30)
    parser.add_argument("--min-sim", type=float, default=0.7)
    parser.add_argument("--max-sim", type=float, default=0.99)
    parser.add_argument(
        "--pause",
        type=float,
        default=25.0,
        help="Seconds between Voyage calls; the free tier allows 3 per minute.",
    )
    args = parser.parse_args()

    if not get_settings().voyage_configured:
        raise SystemExit("VOYAGE_API_KEY is not set. Put it in .env at the repo root.")
    if not RAW_JSON.is_file():
        raise SystemExit(f"Missing corpus: {RAW_JSON}")

    chunks = flatten_chunks(ingest_corpus(RAW_JSON, doc_ids=KHUNG1_DOC_IDS))
    vectors = embed_chunks(chunks, cache_path=CACHE_NPZ)
    index = build_index(
        chunks, vectors=vectors, min_sim=args.min_sim, max_sim=args.max_sim
    )
    edges = sum(len(v) for v in index.neighbours.values())
    print(f"{len(chunks)} chunks | {len(index.events_by_entity)} entities | {edges} edges")

    rows: list[tuple[str, dict[str, object]]] = []
    for position, query in enumerate(QUERIES):
        if position:
            time.sleep(args.pause)
        hybrid_hits = search_hybrid(query, chunks, k=args.hybrid_k, vectors=vectors)
        voyage_hits = rerank_with_retry(
            query,
            [hit.chunk for hit in hybrid_hits],
            top_k=args.voyage_k,
            pause=args.pause,
        )
        seeds = [hit.chunk for hit in voyage_hits]
        context = expand(
            seeds, index, max_extra=args.sag_extra, min_sim=args.min_sim
        )
        stats = measure(seeds, context, index)
        rows.append((query, stats))
        print(
            f"  {query!r}: {stats['seeds']}→{stats['seeds'] + stats['added']} chunks, "
            f"orphans {stats['repaired']}/{stats['orphans']} repaired, "
            f"new laws {stats['new_laws'] or '—'}",
            flush=True,
        )

    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# What SAG adds over the Voyage shortlist",
        "",
        f"Generated {stamp} by `scripts/run_sag_evidence.py` against the approved",
        f"finance corpus ({len(chunks)} chunks, {edges} semantic edges).",
        "",
        f"Settings: hybrid-k={args.hybrid_k}, voyage-k={args.voyage_k}, "
        f"sag-extra={args.sag_extra}, min-sim={args.min_sim}, max-sim={args.max_sim}.",
        "",
        "`Orphans` counts reranked seeds that are fragments whose parent text was",
        "absent from the shortlist. `Repaired` counts how many SAG recovered.",
        "`New laws` are statutes no seed came from, reached by a semantic edge.",
        "",
        "| Query | Seeds | With SAG | Orphans | Repaired | New laws |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for query, s in rows:
        total = int(s["seeds"]) + int(s["added"])  # type: ignore[call-overload]
        laws = ", ".join(s["new_laws"]) or "—"  # type: ignore[arg-type]
        lines.append(
            f"| {query} | {s['seeds']} | {total} | {s['orphans']} | "
            f"{s['repaired']} | {laws} |"
        )

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {REPORT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
