"""SAG v0: event/entity index over legal chunks.

Every chunk is an *event*. Entities are legal coordinates (law / Điều / Khoản).
Chunks sharing an entity key form a local hyperedge — the in-memory stand-in
for the SQL join the full SAG design does at query time.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import zip_longest

import numpy as np

from sag_legal.models import LegalChunk


def entity_keys(
    chunk: LegalChunk,
    extra: Sequence[str] | None = None,
) -> list[str]:
    """Legal coordinates a chunk belongs to, coarse -> fine, plus optional concepts."""
    keys = [f"law::{chunk.document_id}"]
    if chunk.article:
        keys.append(f"art::{chunk.document_id}::{chunk.article}")
        if chunk.clause:
            keys.append(f"cls::{chunk.document_id}::{chunk.article}::{chunk.clause}")
    if extra:
        for key in extra:
            if key and key not in keys:
                keys.append(key)
    return keys


@dataclass
class EventEntityIndex:
    events_by_id: dict[str, LegalChunk] = field(default_factory=dict)
    events_by_entity: dict[str, list[str]] = field(default_factory=dict)
    # chunk_id -> entity keys (structural + concept), for reverse lookup at expand time
    keys_by_event: dict[str, list[str]] = field(default_factory=dict)
    # chunk_id -> [(neighbour_id, cosine), ...], best first. Empty without vectors.
    neighbours: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
 

def build_semantic_edges(
    chunks: Sequence[LegalChunk],
    vectors: Mapping[str, np.ndarray],
    top_n: int = 5,
    min_sim: float = 0.6,
    max_sim: float = 0.99,
) -> dict[str, list[tuple[str, float]]]:
    """Link each event to its nearest neighbours by meaning, across documents.

    `max_sim` drops near-duplicates: legal texts repeat clauses verbatim, and a
    restatement fills a slot without adding anything for the reader.
    """
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
    # Mask before selecting, so a dropped duplicate frees its slot for a real
    # neighbour instead of shrinking the list.
    sims[sims > max_sim] = -1.0

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
    max_sim: float = 0.99,
    concepts: Mapping[str, Sequence[str]] | None = None,
) -> EventEntityIndex:
    index = EventEntityIndex()
    for chunk in chunks:
        if chunk.chunk_id in index.events_by_id:
            continue
        index.events_by_id[chunk.chunk_id] = chunk
        extras = list(concepts.get(chunk.chunk_id, [])) if concepts else None
        keys = entity_keys(chunk, extra=extras)
        index.keys_by_event[chunk.chunk_id] = keys
        for key in keys:
            index.events_by_entity.setdefault(key, []).append(chunk.chunk_id)

    if vectors:
        index.neighbours = build_semantic_edges(
            list(index.events_by_id.values()),
            vectors,
            top_n=top_n,
            min_sim=min_sim,
            max_sim=max_sim,
        )
    return index


def _kinship(candidate: LegalChunk, seed: LegalChunk) -> int:
    """How closely a candidate encloses the seed. Lower is nearer."""
    if candidate.clause is None and candidate.point is None:
        return 0  # the Điều heading
    if candidate.clause == seed.clause and candidate.point is None:
        return 1  # the khoản this fragment hangs off
    if candidate.clause == seed.clause:
        return 2  # điểm siblings inside that khoản
    return 3  # the rest of the article


def _structural_ids(chunk: LegalChunk, index: EventEntityIndex) -> list[str]:
    """Events sharing this chunk's Điều, nearest ancestors first.

    Buckets are in document order, so a seed's own khoản can sit 39 entries
    deep in a long Điều and never be reached before the budget runs out.
    Sorting is stable, so document order still breaks ties within a rank.
    """
    if not chunk.article:
        return []
    bucket = index.events_by_entity.get(f"art::{chunk.document_id}::{chunk.article}", [])
    if len(bucket) < 2:
        return list(bucket)
    return sorted(bucket, key=lambda cid: _kinship(index.events_by_id[cid], chunk))


def _semantic_ids(chunk_id: str, index: EventEntityIndex, min_sim: float) -> list[str]:
    """Events close in meaning, possibly in another law."""
    return [nid for nid, sim in index.neighbours.get(chunk_id, []) if sim >= min_sim]


def _concept_ids(chunk_id: str, index: EventEntityIndex) -> list[str]:
    """Events sharing a concept:: entity with this chunk — the paper's SQL join."""
    ids: list[str] = []
    seen: set[str] = set()
    for key in index.keys_by_event.get(chunk_id, []):
        if not key.startswith("concept::"):
            continue
        for neighbour_id in index.events_by_entity.get(key, []):
            if neighbour_id not in seen:
                seen.add(neighbour_id)
                ids.append(neighbour_id)
    return ids


def _candidate_ids(
    chunk: LegalChunk,
    index: EventEntityIndex,
    use_semantic: bool,
    min_sim: float,
    use_concepts: bool,
) -> list[str]:
    ids = list(_structural_ids(chunk, index))
    if use_concepts:
        ids += _concept_ids(chunk.chunk_id, index)
    if use_semantic:
        ids += _semantic_ids(chunk.chunk_id, index, min_sim)
    return ids


def expand(
    seed_chunks: Sequence[LegalChunk],
    index: EventEntityIndex,
    max_extra: int = 10,
    hops: int = 1,
    min_sim: float = 0.0,
    use_semantic: bool = True,
    use_concepts: bool = True,
) -> list[LegalChunk]:
    """Walk the hyperedges out from the seeds, keeping seed order in front.

    Each hop follows three edge types: same-Điều siblings, shared concept
    entities (LLM-extracted), and semantic neighbours. Candidates are taken
    round-robin across the frontier, so a seed sitting in a 46-clause Điều
    cannot spend the whole budget before the later seeds contribute anything.
    """
    seen = {c.chunk_id for c in seed_chunks}
    extras: list[LegalChunk] = []
    frontier: list[LegalChunk] = list(seed_chunks)

    for _ in range(max(hops, 0)):
        candidate_lists = [
            _candidate_ids(chunk, index, use_semantic, min_sim, use_concepts)
            for chunk in frontier
        ]
        next_frontier: list[LegalChunk] = []
        for column in zip_longest(*candidate_lists):
            for cid in column:
                if cid is None or cid in seen:
                    continue
                if len(extras) >= max_extra:
                    return list(seed_chunks) + extras
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