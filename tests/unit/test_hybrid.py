"""Hybrid RRF tests — use fake dense embedder so CI stays offline."""

from sag_legal.models import DocumentStatus, LegalChunk
from sag_legal.retrieval.bm25 import ScoreChunk
from sag_legal.retrieval.hybrid import (
    fuse_rank_maps,
    hits_to_ranks,
    rrf_contribution,
    search_hybrid,
    top_k_score_chunks,
)


def _chunk(chunk_id: str, text: str) -> LegalChunk:
    return LegalChunk(
        chunk_id=chunk_id,
        document_id="fixture-hybrid",
        text=text,
        status=DocumentStatus.ACTIVE,
    )


def fake_embed(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        lower = text.lower()
        if "phép" in lower or "giấy" in lower:
            vectors.append([1.0, 0.0])
        else:
            vectors.append([0.0, 1.0])
    return vectors


def test_rrf_contribution_prefers_better_rank():
    assert rrf_contribution(1) > rrf_contribution(2)


def test_fuse_rank_maps_adds_both_sides():
    fused = fuse_rank_maps({"a": 1}, {"a": 1, "b": 2})
    assert fused["a"] == rrf_contribution(1) + rrf_contribution(1)
    assert fused["b"] == rrf_contribution(2)


def test_top_k_score_chunks_orders_and_cuts():
    chunks = [_chunk("a", "a"), _chunk("b", "b"), _chunk("c", "c")]
    fused = {"a": 0.1, "b": 0.9, "c": 0.5}
    hits = top_k_score_chunks(fused, chunks, k=2)
    assert [h.chunk.chunk_id for h in hits] == ["b", "c"]
    assert hits[0].score >= hits[1].score


def test_search_hybrid_returns_sorted_hits():
    chunks = [
        _chunk("c1", "Hoạt động ngân hàng bao gồm nhận tiền gửi."),
        _chunk("c2", "Điều kiện cấp giấy phép thành lập tổ chức tín dụng."),
        _chunk("c3", "Tổ chức tín dụng là doanh nghiệp."),
    ]
    hits = search_hybrid("cấp giấy phép", chunks, k=2, model=fake_embed)
    assert len(hits) == 2
    assert hits[0].score >= hits[1].score

def test_hits_to_ranks_is_one_based_in_list_order():
    hits = [
        ScoreChunk(chunk=_chunk("dieu-10", "Điều kiện cấp giấy phép"), score=9.0),
        ScoreChunk(chunk=_chunk("dieu-2", "Giải thích từ ngữ"), score=3.0),
        ScoreChunk(chunk=_chunk("dieu-1", "Phạm vi điều chỉnh"), score=1.0),
    ]

    ranks = hits_to_ranks(hits)

    assert ranks == {"dieu-10": 1, "dieu-2": 2,"dieu-1": 3,}

