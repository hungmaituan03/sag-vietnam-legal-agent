"""Embedding cache tests — a fake embedder keeps CI offline."""

from pathlib import Path

import numpy as np

from sag_legal.models import LegalChunk
from sag_legal.retrieval.dense import search_dense
from sag_legal.retrieval.embeddings import embed_chunks, encode


def _chunk(chunk_id: str, text: str) -> LegalChunk:
    return LegalChunk(chunk_id=chunk_id, document_id="fixture-embed", text=text)


class CountingEmbedder:
    """Fake encoder that records how many texts it was asked to embed."""

    def __init__(self) -> None:
        self.calls = 0
        self.texts_seen = 0

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.texts_seen += len(texts)
        return [[1.0, 0.0] if "phép" in t.lower() else [0.0, 1.0] for t in texts]


def test_encode_empty_returns_empty_matrix():
    assert encode([]).shape == (0, 0)


def test_embed_chunks_maps_every_id():
    chunks = [_chunk("c1", "cấp giấy phép"), _chunk("c2", "nhận tiền gửi")]
    vectors = embed_chunks(chunks, model=CountingEmbedder())

    assert set(vectors) == {"c1", "c2"}
    assert np.allclose(vectors["c1"], [1.0, 0.0])


def test_embed_chunks_writes_then_reuses_cache(tmp_path: Path):
    chunks = [_chunk("c1", "cấp giấy phép"), _chunk("c2", "nhận tiền gửi")]
    cache = tmp_path / "vectors.npz"
    embedder = CountingEmbedder()

    first = embed_chunks(chunks, cache_path=cache, model=embedder)
    assert cache.is_file()
    assert embedder.texts_seen == 2

    second = embed_chunks(chunks, cache_path=cache, model=embedder)
    assert embedder.texts_seen == 2, "second call must not re-encode"
    assert np.allclose(first["c1"], second["c1"])


def test_embed_chunks_ignores_cache_when_ids_change(tmp_path: Path):
    cache = tmp_path / "vectors.npz"
    embedder = CountingEmbedder()
    embed_chunks([_chunk("c1", "cấp giấy phép")], cache_path=cache, model=embedder)

    # A different corpus must not silently reuse the previous matrix.
    vectors = embed_chunks(
        [_chunk("c1", "cấp giấy phép"), _chunk("c2", "nhận tiền gửi")],
        cache_path=cache,
        model=embedder,
    )

    assert set(vectors) == {"c1", "c2"}
    assert embedder.texts_seen == 3


def test_search_dense_with_precomputed_vectors_skips_the_embedder():
    chunks = [
        _chunk("c1", "Hoạt động ngân hàng bao gồm nhận tiền gửi."),
        _chunk("c2", "Điều kiện cấp giấy phép thành lập tổ chức tín dụng."),
    ]
    embedder = CountingEmbedder()
    vectors = embed_chunks(chunks, model=embedder)
    embedder.texts_seen = 0

    hits = search_dense("cấp giấy phép", chunks, k=1, model=embedder, vectors=vectors)

    assert hits[0].chunk.chunk_id == "c2"
    assert embedder.texts_seen == 1, "only the query should be embedded"
