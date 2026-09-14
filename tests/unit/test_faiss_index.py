"""FAISS dense-index tests — requires faiss-cpu (installed via .[dev])."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sag_legal.models import DocumentStatus, LegalChunk
from sag_legal.retrieval.dense import search_dense
from sag_legal.retrieval.faiss_index import DenseFaissIndex


def _chunk(chunk_id: str, text: str) -> LegalChunk:
    return LegalChunk(
        chunk_id=chunk_id,
        document_id="fixture-faiss",
        text=text,
        status=DocumentStatus.ACTIVE,
    )


def test_faiss_from_vectors_ranks_nearest_neighbour():
    vectors = {
        "a": np.array([1.0, 0.0], dtype=np.float32),
        "b": np.array([0.0, 1.0], dtype=np.float32),
        "c": np.array([0.9, 0.1], dtype=np.float32),
    }
    index = DenseFaissIndex.from_vectors(vectors)
    hits = index.search(np.array([1.0, 0.0], dtype=np.float32), k=2)
    assert [chunk_id for chunk_id, _ in hits] == ["a", "c"]


def test_faiss_save_and_load_roundtrip(tmp_path: Path):
    vectors = {
        "a": np.array([1.0, 0.0], dtype=np.float32),
        "b": np.array([0.0, 1.0], dtype=np.float32),
    }
    path = tmp_path / "mini.faiss"
    DenseFaissIndex.from_vectors(vectors).save(path)
    loaded = DenseFaissIndex.load(path)
    hits = loaded.search(np.array([0.0, 1.0], dtype=np.float32), k=1)
    assert hits[0][0] == "b"


def test_search_dense_uses_faiss_index():
    chunks = [
        _chunk("a", "permit giấy phép"),
        _chunk("b", "banking tiền gửi"),
    ]
    vectors = {
        "a": np.array([1.0, 0.0], dtype=np.float32),
        "b": np.array([0.0, 1.0], dtype=np.float32),
    }
    index = DenseFaissIndex.from_vectors(vectors)

    def fake_embed(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    hits = search_dense(
        "giấy phép",
        chunks,
        k=1,
        model=fake_embed,
        faiss_index=index,
    )
    assert hits[0].chunk.chunk_id == "a"
