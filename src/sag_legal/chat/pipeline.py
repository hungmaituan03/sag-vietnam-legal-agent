"""Finance-corpus ask path for the chat UI.

Warm-loads chunks + embeddings + SAG index once per process, then each query
runs hybrid → Voyage → expand → generate_draft.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from sag_legal.generation import DraftAnswer, generate_draft
from sag_legal.ingestion import KHUNG1_DOC_IDS, flatten_chunks, ingest_corpus
from sag_legal.models import LegalChunk
from sag_legal.reranking import rerank
from sag_legal.retrieval.embeddings import embed_chunks
from sag_legal.retrieval.hybrid import search_hybrid
from sag_legal.sag import (
    ChunkExtraction,
    EventEntityIndex,
    build_index,
    concept_keys_by_chunk,
    expand,
    extract_query,
    load_extractions,
    seeds_with_query_concepts,
)
from sag_legal.settings import get_settings


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return here.parents[3]


ROOT = _repo_root()
RAW_JSON = ROOT / "data" / "raw" / "uts_vlc_processed.json"
CACHE_NPZ = ROOT / "data" / "processed" / "khung1_embeddings.npz"
CONCEPT_CACHE = ROOT / "data" / "processed" / "qwen_extract_khung1.json"

RetrieveFn = Callable[..., list[LegalChunk]]
GenerateFn = Callable[[str, Sequence[LegalChunk]], DraftAnswer]
QueryExtractFn = Callable[[str], ChunkExtraction]


@dataclass
class CitedChunk:
    chunk_id: str
    document_id: str
    title: str
    citation_path: str
    text: str


@dataclass
class ChatStats:
    seed_count: int
    context_count: int
    sag_added: int


@dataclass
class ChatResult:
    answer: str
    abstained: bool
    cited: list[CitedChunk] = field(default_factory=list)
    stats: ChatStats = field(default_factory=lambda: ChatStats(0, 0, 0))
    use_sag: bool = True


@dataclass
class CorpusBundle:
    chunks: list[LegalChunk]
    titles: dict[str, str]
    vectors: Mapping[str, np.ndarray]
    index: EventEntityIndex


_bundle: CorpusBundle | None = None
_bundle_lock = Lock()


def corpus_ready() -> bool:
    return RAW_JSON.is_file()


def get_corpus(
    *,
    raw_json: Path | None = None,
    cache_npz: Path | None = None,
    min_sim: float = 0.6,
    force_reload: bool = False,
) -> CorpusBundle:
    """Lazy singleton: ingest + embed + SAG index once."""
    global _bundle
    with _bundle_lock:
        if _bundle is not None and not force_reload:
            return _bundle

        path = raw_json or RAW_JSON
        if not path.is_file():
            raise FileNotFoundError(f"Missing corpus: {path}")

        results = ingest_corpus(path, doc_ids=KHUNG1_DOC_IDS)
        titles = {r.document.document_id: r.document.title for r in results}
        chunks = flatten_chunks(results)
        vectors = embed_chunks(chunks, cache_path=cache_npz or CACHE_NPZ)
        concepts = None
        if CONCEPT_CACHE.is_file():
            concepts = concept_keys_by_chunk(load_extractions(CONCEPT_CACHE))
        index = build_index(
            chunks, vectors=vectors, min_sim=min_sim, concepts=concepts
        )
        _bundle = CorpusBundle(
            chunks=chunks, titles=titles, vectors=vectors, index=index
        )
        return _bundle


def _augment_seeds_with_query(
    query: str,
    seeds: list[LegalChunk],
    index: EventEntityIndex,
    *,
    query_extract_fn: QueryExtractFn | None = None,
) -> list[LegalChunk]:
    """LLM-extract the query, join concept:: keys into the Voyage seed list."""
    if query_extract_fn is None and not get_settings().openai_configured:
        return seeds
    extraction = extract_query(query, extract_fn=query_extract_fn)
    return seeds_with_query_concepts(extraction.concept_keys(), seeds, index)


def _default_retrieve(
    query: str,
    bundle: CorpusBundle,
    *,
    hybrid_k: int,
    voyage_k: int,
    sag_extra: int,
    hops: int,
    min_sim: float,
    use_sag: bool,
    voyage_client: Any | None,
    query_extract_fn: QueryExtractFn | None = None,
) -> tuple[list[LegalChunk], list[LegalChunk]]:
    """Return (seeds, evidence). Seeds are Voyage top-k; evidence is after SAG."""
    hybrid_hits = search_hybrid(
        query, bundle.chunks, k=hybrid_k, vectors=bundle.vectors
    )
    voyage_hits = rerank(
        query,
        [hit.chunk for hit in hybrid_hits],
        top_k=voyage_k,
        client=voyage_client,
    )
    seeds = [hit.chunk for hit in voyage_hits]
    if not use_sag:
        return seeds, list(seeds)
    seeds = _augment_seeds_with_query(
        query, seeds, bundle.index, query_extract_fn=query_extract_fn
    )
    evidence = expand(
        seeds,
        bundle.index,
        max_extra=sag_extra,
        hops=hops,
        min_sim=min_sim,
    )
    return seeds, evidence


def answer_query(
    query: str,
    *,
    bundle: CorpusBundle | None = None,
    hybrid_k: int = 20,
    voyage_k: int = 5,
    sag_extra: int = 10,
    rag_k: int | None = None,
    hops: int = 1,
    min_sim: float = 0.6,
    use_sag: bool = True,
    voyage_client: Any | None = None,
    generate_fn: GenerateFn | None = None,
    retrieve_fn: RetrieveFn | None = None,
    query_extract_fn: QueryExtractFn | None = None,
) -> ChatResult:
    """Run retrieval → draft. Inject retrieve_fn / generate_fn for offline tests.

    ``use_sag=False`` is the RAG baseline: Voyage shortlist only (no expand).
    Pack size defaults to ``voyage_k + sag_extra`` (15) so RAG matches SAG's
    typical evidence budget; override with ``rag_k=``.
    """
    cleaned = query.strip()
    if not cleaned:
        return ChatResult(
            answer="Vui lòng nhập câu hỏi.",
            abstained=True,
            stats=ChatStats(0, 0, 0),
            use_sag=use_sag,
        )

    active = bundle if bundle is not None else get_corpus(min_sim=min_sim)
    effective_k = (
        voyage_k
        if use_sag
        else (rag_k if rag_k is not None else voyage_k + sag_extra)
    )

    if retrieve_fn is not None:
        evidence = retrieve_fn(cleaned, active)
        seeds = evidence[:effective_k]
    else:
        seeds, evidence = _default_retrieve(
            cleaned,
            active,
            hybrid_k=hybrid_k,
            voyage_k=effective_k,
            sag_extra=sag_extra,
            hops=hops,
            min_sim=min_sim,
            use_sag=use_sag,
            voyage_client=voyage_client,
            query_extract_fn=query_extract_fn,
        )

    seed_ids = {c.chunk_id for c in seeds}
    draft = generate_draft(
        cleaned, evidence, generate_fn=generate_fn, seed_ids=seed_ids
    )
    by_id = {c.chunk_id: c for c in evidence}
    cite_order = list(draft.cited_chunk_ids)
    if not cite_order and not draft.abstained:
        cite_order = [c.chunk_id for c in evidence[: min(5, len(evidence))]]
    cited: list[CitedChunk] = []
    for cid in cite_order:
        chunk = by_id.get(cid)
        if chunk is None:
            continue
        cited.append(
            CitedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                title=active.titles.get(chunk.document_id, chunk.document_id),
                citation_path=chunk.citation_path,
                text=chunk.text[:280],
            )
        )

    return ChatResult(
        answer=draft.answer,
        abstained=draft.abstained,
        cited=cited,
        stats=ChatStats(
            seed_count=len(seeds),
            context_count=len(evidence),
            sag_added=len([c for c in evidence if c.chunk_id not in seed_ids]),
        ),
        use_sag=use_sag,
    )


def compare_query(
    query: str,
    *,
    bundle: CorpusBundle | None = None,
    hybrid_k: int = 20,
    voyage_k: int = 5,
    sag_extra: int = 10,
    rag_k: int | None = None,
    hops: int = 1,
    min_sim: float = 0.6,
    voyage_client: Any | None = None,
    generate_fn: GenerateFn | None = None,
    query_extract_fn: QueryExtractFn | None = None,
) -> tuple[ChatResult, ChatResult]:
    """K-matched ablation: RAG = Voyage top-(voyage_k+sag_extra); SAG = top-voyage_k + expand.

    One hybrid → one Voyage call (depth = RAG budget). RAG keeps the deep
    shortlist; SAG keeps only the first ``voyage_k`` as seeds then expands.
    """
    cleaned = query.strip()
    if not cleaned:
        empty = ChatResult(
            answer="Vui lòng nhập câu hỏi.",
            abstained=True,
            stats=ChatStats(0, 0, 0),
            use_sag=False,
        )
        return empty, ChatResult(
            answer=empty.answer,
            abstained=True,
            stats=ChatStats(0, 0, 0),
            use_sag=True,
        )

    active = bundle if bundle is not None else get_corpus(min_sim=min_sim)
    rag_budget = rag_k if rag_k is not None else voyage_k + sag_extra
    hybrid_hits = search_hybrid(
        cleaned, active.chunks, k=hybrid_k, vectors=active.vectors
    )
    voyage_hits = rerank(
        cleaned,
        [hit.chunk for hit in hybrid_hits],
        top_k=max(rag_budget, voyage_k),
        client=voyage_client,
    )
    ranked = [hit.chunk for hit in voyage_hits]
    rag_evidence = ranked[:rag_budget]
    sag_seeds = ranked[:voyage_k]
    sag_seeds = _augment_seeds_with_query(
        cleaned, sag_seeds, active.index, query_extract_fn=query_extract_fn
    )
    sag_evidence = expand(
        sag_seeds,
        active.index,
        max_extra=sag_extra,
        hops=hops,
        min_sim=min_sim,
    )

    def _finish(
        evidence: list[LegalChunk],
        use_sag: bool,
        seed_list: list[LegalChunk],
    ) -> ChatResult:
        seed_ids = {c.chunk_id for c in seed_list}
        draft = generate_draft(
            cleaned, evidence, generate_fn=generate_fn, seed_ids=seed_ids
        )
        by_id = {c.chunk_id: c for c in evidence}
        cite_order = list(draft.cited_chunk_ids)
        if not cite_order and not draft.abstained:
            cite_order = [c.chunk_id for c in evidence[: min(5, len(evidence))]]
        cited = [
            CitedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                title=active.titles.get(chunk.document_id, chunk.document_id),
                citation_path=chunk.citation_path,
                text=chunk.text[:280],
            )
            for cid in cite_order
            if (chunk := by_id.get(cid)) is not None
        ]
        return ChatResult(
            answer=draft.answer,
            abstained=draft.abstained,
            cited=cited,
            stats=ChatStats(
                seed_count=len(seed_list),
                context_count=len(evidence),
                sag_added=len([c for c in evidence if c.chunk_id not in seed_ids]),
            ),
            use_sag=use_sag,
        )

    return (
        _finish(rag_evidence, False, rag_evidence),
        _finish(sag_evidence, True, sag_seeds),
    )


def keys_configured() -> tuple[bool, bool]:
    settings = get_settings()
    return settings.voyage_configured, settings.openai_configured
