"""Dense retrieval tests — use a fake embedder so CI never downloads a model."""

from sag_legal.models import DocumentStatus, LegalChunk
from sag_legal.retrieval.dense import cosine, search_dense


def _chunk(chunk_id: str, text: str) -> LegalChunk:
    """Tiny factory: only fields we need for ranking tests."""
    return LegalChunk(
        chunk_id=chunk_id,
        document_id="fixture-dense",
        text=text,
        status=DocumentStatus.ACTIVE,
    )


def fake_embed(texts: list[str]) -> list[list[float]]:
    """Map text → 2D vectors by hand.

    Idea:
    - texts about permits/licenses ("phép") → near [1, 0]
    - unrelated banking texts → near [0, 1]
    - query "cấp giấy phép" → near [1, 0] so it should match permit chunks
    """
    vectors: list[list[float]] = []
    for text in texts:
        lower = text.lower()
        if "phép" in lower or "giấy" in lower:
            vectors.append([1.0, 0.0])
        else:
            vectors.append([0.0, 1.0])
    return vectors


def test_cosine_identical_vectors():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_search_dense_ranks_permit_chunk_higher_with_fake_embed():
    chunks = [
        _chunk("c1", "Hoạt động ngân hàng bao gồm nhận tiền gửi."),
        _chunk("c2", "Điều kiện cấp giấy phép thành lập tổ chức tín dụng."),
        _chunk("c3", "Tổ chức tín dụng là doanh nghiệp."),
    ]
    query = "cấp giấy phép"

    hits = search_dense(query, chunks, k=2, model=fake_embed)

    assert len(hits) == 2
    assert hits[0].score >= hits[1].score
    # top hit should be the permit-related chunk (c2)
    assert "giấy phép" in hits[0].chunk.text
    assert hits[0].chunk.chunk_id == "c2"
