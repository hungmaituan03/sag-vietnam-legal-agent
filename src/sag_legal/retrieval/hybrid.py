"""Hybrid retrieval via Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

from sag_legal.models import LegalChunk
from sag_legal.retrieval.bm25 import ScoreChunk, search_bm25
from sag_legal.retrieval.dense import search_dense


def rrf_contribution(rank: int, k_rrf: int = 60) -> float:
    return 1.0 / (k_rrf + rank)


def hits_to_ranks(hits: list[ScoreChunk]) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for index, hit in enumerate(hits):
        ranks[hit.chunk.chunk_id] = index + 1
    return ranks


def fuse_rank_maps(
    bm25_ranks: dict[str, int], 
    dense_ranks: dict[str, int], 
    k_rrf: int = 60,
    ) -> dict[str, float]:
    all_ids = set(bm25_ranks) | set(dense_ranks)
    fused: dict[str, float] = {}
    for chunk_id in all_ids:
        score = 0.0
        if chunk_id in bm25_ranks:
            score += rrf_contribution(bm25_ranks[chunk_id], k_rrf=k_rrf)
        if chunk_id in dense_ranks:
            score += rrf_contribution(dense_ranks[chunk_id], k_rrf=k_rrf)
        fused[chunk_id] = score
    return fused


def top_k_score_chunks(
    fused_scores: dict[str, float],
    chunks: list[LegalChunk],
    k: int,
) -> list[ScoreChunk]:
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    ranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    results: list[ScoreChunk] = []
    for chunk_id, score in ranked[:k]:
        results.append(ScoreChunk(chunk=chunks_by_id[chunk_id], score=score))
    return results


def search_hybrid(
    query: str,
    chunks: list[LegalChunk],
    k: int = 5,
    model=None,
) -> list[ScoreChunk]:
    if not chunks or k <= 0:
        return []

    pool = len(chunks)
    bm25_hits = search_bm25(query, chunks, k=pool)
    dense_hits = search_dense(query, chunks, k=pool, model=model)

    bm25_ranks = hits_to_ranks(bm25_hits)
    dense_ranks = hits_to_ranks(dense_hits)
    fused_scores = fuse_rank_maps(bm25_ranks, dense_ranks)

    return top_k_score_chunks(fused_scores, chunks, k=k)
