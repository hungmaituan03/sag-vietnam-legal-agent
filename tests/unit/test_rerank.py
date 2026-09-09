"""Voyage rerank tests — fake client so CI needs no API key."""

from types import SimpleNamespace

from sag_legal.models import DocumentStatus, LegalChunk
from sag_legal.reranking.rerank import rerank
from sag_legal.retrieval.bm25 import ScoreChunk


def _chunk(chunk_id: str, text: str) -> LegalChunk:
    return LegalChunk(
        chunk_id=chunk_id,
        document_id="fixture-rerank",
        text=text,
        status=DocumentStatus.ACTIVE,
    )


class FakeVoyageClient:
    """Mimics voyageai.Client.rerank: returns sorted results with index + score."""

    def rerank(self, query, documents, model, top_k=None, truncation=True):
        scored = []
        for index, doc in enumerate(documents):
            lower = doc.lower()
            # Higher score if doc looks permit-related (same teaching trick as dense)
            score = 0.9 if ("phép" in lower or "giấy" in lower) else 0.1
            scored.append(SimpleNamespace(index=index, document=doc, relevance_score=score))
        scored.sort(key=lambda item: item.relevance_score, reverse=True)
        if top_k is not None:
            scored = scored[:top_k]
        return SimpleNamespace(results=scored)


def test_rerank_empty_chunks():
    assert rerank("query", [], top_k=5, client=FakeVoyageClient()) == []


def test_rerank_orders_by_fake_relevance():
    chunks = [
        _chunk("c1", "Hoạt động ngân hàng bao gồm nhận tiền gửi."),
        _chunk("c2", "Điều kiện cấp giấy phép thành lập tổ chức tín dụng."),
        _chunk("c3", "Tổ chức tín dụng là doanh nghiệp."),
    ]
    hits = rerank("cấp giấy phép", chunks, top_k=2, client=FakeVoyageClient())

    assert len(hits) == 2
    assert all(isinstance(h, ScoreChunk) for h in hits)
    assert hits[0].chunk.chunk_id == "c2"
    assert hits[0].score >= hits[1].score


def test_rerank_top_k_not_larger_than_input():
    chunks = [_chunk("a", "a"), _chunk("b", "giấy phép")]
    hits = rerank("giấy phép", chunks, top_k=10, client=FakeVoyageClient())
    assert len(hits) == 2
