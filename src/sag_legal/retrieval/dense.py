"""Dense (embedding) retrieval — numpy for small/CI, FAISS for corpus scale."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from sag_legal.models import LegalChunk
from sag_legal.retrieval.embeddings import encode
from sag_legal.retrieval.faiss_index import DenseFaissIndex


@dataclass
class ScoreChunk:
    chunk: LegalChunk
    score: float


def cosine(a, b) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def embed_texts(texts: list[str], model=None) -> list[list[float]]:
    if not texts:
        return []
    return [list(v) for v in encode(texts, model=model)]


def search_dense(
    query: str,
    chunks: Sequence[LegalChunk],
    k: int = 5,
    model=None,
    vectors: Mapping[str, np.ndarray] | None = None,
    faiss_index: DenseFaissIndex | None = None,
) -> list[ScoreChunk]:
    """Rank chunks by cosine / IP similarity.

    Prefer ``faiss_index=`` for large corpora. Without it, falls back to a
    full numpy scan (fine for fixtures and the Khung 1 pack).
    """
    if not chunks:
        return []
    k = max(0, k)
    if k == 0:
        return []

    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    query_vec = np.asarray(embed_texts([query], model=model)[0], dtype=np.float32)

    if faiss_index is not None:
        hits = faiss_index.search(query_vec, k)
        return [
            ScoreChunk(chunk=by_id[chunk_id], score=score)
            for chunk_id, score in hits
            if chunk_id in by_id
        ]

    if vectors is None:
        chunk_vecs = embed_texts([c.text for c in chunks], model=model)
    else:
        chunk_vecs = [vectors[c.chunk_id] for c in chunks]

    paired = [
        ScoreChunk(chunk=c, score=cosine(query_vec, v))
        for c, v in zip(chunks, chunk_vecs, strict=True)
    ]
    paired.sort(key=lambda x: x.score, reverse=True)
    return paired[:k]
