"""Corpus embeddings with an on-disk cache.

Encoding 4k chunks takes minutes, and every query used to pay that cost again
because the encoder was rebuilt per call. Encode once, keep the vectors next to
their chunk ids, reuse them for dense search and for SAG semantic edges.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np

from sag_legal.models import LegalChunk

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

Embedder = Callable[[list[str]], Sequence[Sequence[float]]]


@lru_cache(maxsize=2)
def get_encoder(name: str = DEFAULT_MODEL):
    """Load the sentence-transformer once per process."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def encode(texts: Sequence[str], model: Embedder | None = None) -> np.ndarray:
    """Rows of unit-norm vectors, one per text."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    if callable(model):
        return np.asarray(model(list(texts)), dtype=np.float32)
    vectors = get_encoder().encode(list(texts), normalize_embeddings=True)
    return np.asarray(vectors, dtype=np.float32)


def _load_cache(path: Path, ids: list[str]) -> np.ndarray | None:
    """Return the cached matrix only if it matches these ids exactly."""
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        cached_ids = list(data["ids"])
        matrix = data["vectors"]
    if cached_ids != ids:
        return None
    return matrix


def _save_cache(path: Path, ids: list[str], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=np.array(ids, dtype=object).astype(str), vectors=matrix)


def embed_chunks(
    chunks: Sequence[LegalChunk],
    cache_path: Path | str | None = None,
    model: Embedder | None = None,
) -> dict[str, np.ndarray]:
    """Map chunk_id -> vector, reading or filling the cache when given a path."""
    ids = [chunk.chunk_id for chunk in chunks]
    path = Path(cache_path) if cache_path is not None else None

    matrix = _load_cache(path, ids) if path is not None else None
    if matrix is None:
        matrix = encode([chunk.text for chunk in chunks], model=model)
        if path is not None:
            _save_cache(path, ids, matrix)

    return dict(zip(ids, matrix, strict=True))
