"""SAG v0: event/entity index over legal chunks.

Every chunk is an *event*. Entities are legal coordinates (law / Điều / Khoản).
Chunks sharing an entity key form a local hyperedge — the in-memory stand-in
for the SQL join the full SAG design does at query time.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

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


def build_index(chunks: Iterable[LegalChunk]) -> EventEntityIndex:
    index = EventEntityIndex()
    for chunk in chunks:
        if chunk.chunk_id in index.events_by_id:
            continue
        index.events_by_id[chunk.chunk_id] = chunk
        for key in entity_keys(chunk):
            index.events_by_entity.setdefault(key, []).append(chunk.chunk_id)
    return index


def expand(
    seed_chunks: Sequence[LegalChunk],
    index: EventEntityIndex,
    max_extra: int = 10,
) -> list[LegalChunk]:
    """Add same-Điều siblings of the seeds, keeping seed order in front."""
    seen = {c.chunk_id for c in seed_chunks}
    extras: list[LegalChunk] = []
    for chunk in seed_chunks:
        if not chunk.article:
            continue
        art_key = f"art::{chunk.document_id}::{chunk.article}"
        for cid in index.events_by_entity.get(art_key, []):
            if len(extras) >= max_extra:
                return list(seed_chunks) + extras
            if cid in seen:
                continue
            seen.add(cid)
            extras.append(index.events_by_id[cid])
    return list(seed_chunks) + extras