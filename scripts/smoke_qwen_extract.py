#!/usr/bin/env python3
"""Smoke-test Qwen extraction on a few real finance-pack chunks.

Requires QWEN_API_KEY in .env. Writes/reads a small cache so reruns are free.

Usage (repo root, venv on):
  python scripts/smoke_qwen_extract.py
  python scripts/smoke_qwen_extract.py --limit 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sag_legal.ingestion import FINANCE_DOC_IDS, flatten_chunks, ingest_corpus
from sag_legal.sag import build_index, concept_keys_by_chunk, expand, extract_chunks
from sag_legal.settings import get_settings

ROOT = Path(__file__).resolve().parents[1]
RAW_JSON = ROOT / "data" / "raw" / "uts_vlc_processed.json"
CACHE = ROOT / "data" / "processed" / "qwen_extract_smoke.json"

# Fragments that only make sense with concept/structure repair.
SEED_IDS = (
    "luat-ke-toan::Điều40::Khoản2::Điểma",
    "luat-ke-toan::Điều12::Khoản1::Điểma",
    "law-2020-luat-doanh-nghiep::Điều128::Khoản3::Điểmb",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Qwen extract smoke test")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    if not get_settings().qwen_configured:
        raise SystemExit("QWEN_API_KEY is not set. Put it in .env at the repo root.")
    if not RAW_JSON.is_file():
        raise SystemExit(f"Missing corpus: {RAW_JSON}")

    chunks = flatten_chunks(ingest_corpus(RAW_JSON, doc_ids=FINANCE_DOC_IDS))
    by_id = {c.chunk_id: c for c in chunks}
    sample = [by_id[cid] for cid in SEED_IDS if cid in by_id][: args.limit]
    if not sample:
        raise SystemExit("None of the seed chunk ids were found in the finance pack.")

    print(f"Extracting {len(sample)} chunks with {get_settings().qwen_model}…")
    extractions = extract_chunks(sample, cache_path=CACHE)
    concepts = concept_keys_by_chunk(extractions)

    for chunk in sample:
        extraction = extractions[chunk.chunk_id]
        print(f"\n=== {chunk.citation_path} ({chunk.document_id}) ===")
        print(chunk.text.replace("\n", " ")[:120])
        print("events:")
        for event in extraction.events:
            print(
                f"  - {event.event_id}: {event.summary} "
                f"| actor={event.actor} action={event.action} "
                f"modality={event.modality} trigger={event.trigger}"
            )
        print("entities:")
        for entity in extraction.entities:
            print(f"  - {entity.name} ({entity.type}) aliases={entity.aliases}")
        print("concept keys:", concepts[chunk.chunk_id])

    index = build_index(sample, concepts=concepts)
    seed = sample[0]
    context = expand([seed], index, max_extra=5, use_semantic=False)
    print(f"\nexpand({seed.citation_path}) via concepts/structure:")
    for chunk in context:
        tag = "SEED" if chunk.chunk_id == seed.chunk_id else "+SAG"
        print(f"  {tag} | {chunk.document_id} | {chunk.citation_path}")

    print(f"\nCache: {CACHE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
