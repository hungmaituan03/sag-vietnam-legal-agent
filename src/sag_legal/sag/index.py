"""SAG v0: event/entity index over legal chunks.

Every chunk is an *event*. Entities are legal coordinates (law / Điều / Khoản).
Chunks sharing an entity key form a local hyperedge — the in-memory stand-in
for the SQL join the full SAG design does at query time.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from sag_legal.models import LegalChunk


def entity_keys(chunk: LegalChunk) -> list[str]:
    """Legal coordinates a chunk belongs to, coarse -> fine."""
    keys = [f"law::{chunk.document_id}"]
    if chunk.article:
        keys.append(f"art::{chunk.document_id}::{chunk.article}")
        if chunk.clause:
            keys.append(f"cls::{chunk.document_id}::{chunk.article}::{chunk.clause}")
    return keys


@dataclass
class EventEntityIndex:
    events_by_id: dict[str, LegalChunk] = field(default_factory=dict)
    events_by_entity: dict[str, list[str]] = field(default_factory=dict)
    # chunk_id -> [(neighbour_id, cosine), ...], best first. Empty without vectors.
    neighbours: dict[str, list[tuple[str, float]]] = field(default_factory=dict)


def build_semantic_edges(
    chunks: Sequence[LegalChunk],
    vectors: Mapping[str, np.ndarray],
    top_n: int = 5,
    min_sim: float = 0.6,
) -> dict[str, list[tuple[str, float]]]:
    """Link each event to its nearest neighbours by meaning, across documents."""
    ids: list[str] = []
    for chunk in chunks:
        if chunk.chunk_id in vectors and chunk.chunk_id not in ids:
            ids.append(chunk.chunk_id)
    if len(ids) < 2 or top_n < 1:
        return {}

    matrix = np.vstack([np.asarray(vectors[i], dtype=np.float32) for i in ids])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.where(norms == 0.0, 1.0, norms)

    sims = matrix @ matrix.T
    np.fill_diagonal(sims, -1.0)

    keep = min(top_n, len(ids) - 1)
    edges: dict[str, list[tuple[str, float]]] = {}
    for row, chunk_id in enumerate(ids):
        scores = sims[row]
        candidates = np.argpartition(-scores, keep - 1)[:keep]
        candidates = candidates[np.argsort(-scores[candidates])]
        best = [(ids[j], float(scores[j])) for j in candidates if scores[j] >= min_sim]
        if best:
            edges[chunk_id] = best
    return edges


def build_index(
    chunks: Iterable[LegalChunk],
    vectors: Mapping[str, np.ndarray] | None = None,
    top_n: int = 5,
    min_sim: float = 0.6,
) -> EventEntityIndex:
    index = EventEntityIndex()
    for chunk in chunks:
        if chunk.chunk_id in index.events_by_id:
            continue
        index.events_by_id[chunk.chunk_id] = chunk
        for key in entity_keys(chunk):
            index.events_by_entity.setdefault(key, []).append(chunk.chunk_id)

    if vectors:
        index.neighbours = build_semantic_edges(
            list(index.events_by_id.values()), vectors, top_n=top_n, min_sim=min_sim
        )
    return index


def _structural_ids(chunk: LegalChunk, index: EventEntityIndex) -> list[str]:
    """Events sharing this chunk's Điều — the hierarchy edge."""
    if not chunk.article:
        return []
    return index.events_by_entity.get(f"art::{chunk.document_id}::{chunk.article}", [])


def _semantic_ids(chunk_id: str, index: EventEntityIndex, min_sim: float) -> list[str]:
    """Events close in meaning, possibly in another law."""
    return [nid for nid, sim in index.neighbours.get(chunk_id, []) if sim >= min_sim]


def expand(
    seed_chunks: Sequence[LegalChunk],
    index: EventEntityIndex,
    max_extra: int = 10,
    hops: int = 1,
    min_sim: float = 0.0,
    use_semantic: bool = True,
) -> list[LegalChunk]:
    """Walk the hyperedges out from the seeds, keeping seed order in front.

    Each hop follows two edge types: same-Điều siblings and semantic
    neighbours. `max_extra` bounds the whole walk, so a seed in a long article
    can starve later seeds — that is the trade-off of a global budget.
    """
    seen = {c.chunk_id for c in seed_chunks}
    extras: list[LegalChunk] = []
    frontier: list[LegalChunk] = list(seed_chunks)

    for _ in range(max(hops, 0)):
        next_frontier: list[LegalChunk] = []
        for chunk in frontier:
            candidates = _structural_ids(chunk, index)
            if use_semantic:
                candidates = candidates + _semantic_ids(chunk.chunk_id, index, min_sim)
            for cid in candidates:
                if len(extras) >= max_extra:
                    return list(seed_chunks) + extras
                if cid in seen:
                    continue
                neighbour = index.events_by_id.get(cid)
                if neighbour is None:
                    continue
                seen.add(cid)
                extras.append(neighbour)
                next_frontier.append(neighbour)
        if not next_frontier:
            break
        frontier = next_frontier

    return list(seed_chunks) + extras