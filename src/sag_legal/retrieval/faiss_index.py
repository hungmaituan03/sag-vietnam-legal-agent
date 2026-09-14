"""FAISS-backed dense index for corpus-scale top-k search.

Vectors are assumed L2-normalised (as produced by ``embed_chunks``), so
inner-product search equals cosine. CI stays offline: tests can inject a
pre-built index or fall back to the numpy path in ``search_dense``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _require_faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - exercised when dep missing
        raise ImportError(
            "faiss-cpu is required for DenseFaissIndex. "
            'Install with: pip install -e ".[dev]"'
        ) from exc
    return faiss


@dataclass
class DenseFaissIndex:
    """chunk_id list aligned with FAISS row order + an IndexFlatIP."""

    ids: list[str]
    index: Any

    @classmethod
    def from_vectors(
        cls,
        vectors: Mapping[str, np.ndarray],
        ids: Sequence[str] | None = None,
    ) -> DenseFaissIndex:
        """Build an exact IP index. ``ids`` defaults to ``list(vectors)`` order."""
        faiss = _require_faiss()
        ordered = list(ids) if ids is not None else list(vectors.keys())
        if not ordered:
            dim = 1
            index = faiss.IndexFlatIP(dim)
            return cls(ids=[], index=index)

        missing = [chunk_id for chunk_id in ordered if chunk_id not in vectors]
        if missing:
            raise KeyError(f"Missing vectors for chunk ids: {missing[:5]}")

        matrix = np.vstack(
            [np.asarray(vectors[chunk_id], dtype=np.float32) for chunk_id in ordered]
        )
        # Guard against non-unit rows so IP ≈ cosine.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.where(norms == 0.0, 1.0, norms)

        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        return cls(ids=ordered, index=index)

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Return up to k (chunk_id, score) pairs, best first."""
        if k <= 0 or not self.ids:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        norm = float(np.linalg.norm(q))
        if norm > 0.0:
            q = q / norm
        keep = min(k, len(self.ids))
        scores, indices = self.index.search(q, keep)
        out: list[tuple[str, float]] = []
        for score, row in zip(scores[0], indices[0], strict=True):
            if row < 0:
                continue
            out.append((self.ids[int(row)], float(score)))
        return out

    def save(self, path: Path | str) -> None:
        faiss = _require_faiss()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        ids_path = path.with_suffix(path.suffix + ".ids.npy")
        np.save(ids_path, np.array(self.ids, dtype=object))

    @classmethod
    def load(cls, path: Path | str) -> DenseFaissIndex:
        faiss = _require_faiss()
        path = Path(path)
        index = faiss.read_index(str(path))
        ids_path = path.with_suffix(path.suffix + ".ids.npy")
        ids = [str(x) for x in np.load(ids_path, allow_pickle=True)]
        return cls(ids=ids, index=index)
