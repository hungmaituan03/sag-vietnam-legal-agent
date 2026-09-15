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
from sag_legal.ingestion import flatten_chunks, ingest_corpus
from sag_legal.models import LegalChunk
from sag_legal.org import fetch_org_docs, merge_evidence, resolve_orgs
from sag_legal.org.resolve import lookup_org
from sag_legal.reranking import rerank
from sag_legal.retrieval.embeddings import embed_chunks
from sag_legal.retrieval.faiss_index import DenseFaissIndex
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
# Full dump embeddings (separate from the smaller Khung 1 cache).
CACHE_NPZ = ROOT / "data" / "processed" / "full_corpus_embeddings.npz"
FAISS_PATH = ROOT / "data" / "processed" / "full_corpus.faiss"
CONCEPT_CACHE = ROOT / "data" / "processed" / "qwen_extract_khung1.json"
# All-pairs cosine for SAG semantic edges is O(n²); skip above this size.
SEMANTIC_EDGE_MAX_CHUNKS = 20_000

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
class OrgCandidate:
    org_id: str
    display_name: str


@dataclass
class OrgClarify:
    """Ambiguous org match — caller should abstain and show candidates."""

    prompt: str
    candidates: list[OrgCandidate]


@dataclass
class ChatResult:
    answer: str
    abstained: bool
    cited: list[CitedChunk] = field(default_factory=list)
    stats: ChatStats = field(default_factory=lambda: ChatStats(0, 0, 0))
    use_sag: bool = True
    needs_org_clarify: bool = False
    org_candidates: list[OrgCandidate] = field(default_factory=list)


@dataclass
class CorpusBundle:
    chunks: list[LegalChunk]
    titles: dict[str, str]
    vectors: Mapping[str, np.ndarray]
    index: EventEntityIndex
    faiss_index: DenseFaissIndex | None = None


def _load_or_build_faiss(
    vectors: Mapping[str, np.ndarray],
    cache_path: Path,
) -> DenseFaissIndex:
    """Reuse on-disk FAISS when present; otherwise build and save."""
    ids_sidecar = Path(str(cache_path) + ".ids.npy")
    if cache_path.is_file() and ids_sidecar.is_file():
        print(f"Loading FAISS index: {cache_path.name}", flush=True)
        return DenseFaissIndex.load(cache_path)

    print(f"Building FAISS index ({len(vectors)} vectors)…", flush=True)
    index = DenseFaissIndex.from_vectors(vectors)
    index.save(cache_path)
    print(f"Wrote {cache_path.name}", flush=True)
    return index


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

        results = ingest_corpus(path, all_docs=True)
        titles = {r.document.document_id: r.document.title for r in results}
        chunks = flatten_chunks(results)
        print(
            f"Corpus: {len(results)} docs → {len(chunks)} chunks "
            f"(embedding cache: {(cache_npz or CACHE_NPZ).name})",
            flush=True,
        )
        vectors = embed_chunks(chunks, cache_path=cache_npz or CACHE_NPZ)
        faiss_path = FAISS_PATH
        if cache_npz is not None:
            faiss_path = cache_npz.with_suffix(".faiss")
        faiss_index = _load_or_build_faiss(vectors, faiss_path)
        concepts = None
        if CONCEPT_CACHE.is_file():
            concepts = concept_keys_by_chunk(load_extractions(CONCEPT_CACHE))
        # Dense retrieval still uses `vectors`; SAG semantic edges need all-pairs
        # cosine and are only safe on the smaller Khung 1-sized packs.
        sag_vectors = (
            vectors if len(chunks) <= SEMANTIC_EDGE_MAX_CHUNKS else None
        )
        if sag_vectors is None:
            print(
                f"SAG semantic edges skipped "
                f"({len(chunks)} chunks > {SEMANTIC_EDGE_MAX_CHUNKS})",
                flush=True,
            )
        index = build_index(
            chunks, vectors=sag_vectors, min_sim=min_sim, concepts=concepts
        )
        print(
            f"SAG index: {len(index.events_by_id)} events, "
            f"{len(index.events_by_entity)} entities",
            flush=True,
        )
        _bundle = CorpusBundle(
            chunks=chunks,
            titles=titles,
            vectors=vectors,
            index=index,
            faiss_index=faiss_index,
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
        query,
        bundle.chunks,
        k=hybrid_k,
        vectors=bundle.vectors,
        faiss_index=bundle.faiss_index,
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


def _apply_org_evidence(
    query: str,
    evidence: list[LegalChunk],
    *,
    org_id: str | None = None,
) -> list[LegalChunk] | OrgClarify:
    """Merge org docs into evidence, or return clarify signal if ambiguous.

    When ``org_id`` is set (user re-run after clarify), skip resolve and fetch
    that org. Unknown ``org_id`` leaves evidence unchanged.
    """
    if org_id is not None:
        ref = lookup_org(org_id)
        if ref is None:
            return evidence
        return merge_evidence(evidence, fetch_org_docs(ref))

    refs = resolve_orgs(query)
    if any(r.needs_clarify for r in refs):
        return OrgClarify(
            prompt=refs[0].clarify_prompt,
            candidates=[
                OrgCandidate(org_id=r.org_id, display_name=r.display_name)
                for r in refs
            ],
        )
    if refs:
        return merge_evidence(evidence, fetch_org_docs(refs[0]))
    return evidence


def _clarify_result(
    clarify: OrgClarify,
    *,
    seeds: list[LegalChunk],
    evidence: list[LegalChunk],
    use_sag: bool,
) -> ChatResult:
    return ChatResult(
        answer=clarify.prompt,
        abstained=True,
        stats=ChatStats(len(seeds), len(evidence), 0),
        use_sag=use_sag,
        needs_org_clarify=True,
        org_candidates=list(clarify.candidates),
    )


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
    org_id: str | None = None,
) -> ChatResult:
    """Run retrieval → draft. Inject retrieve_fn / generate_fn for offline tests.

    ``use_sag=False`` is the RAG baseline: Voyage shortlist only (no expand).
    Pack size defaults to ``voyage_k + sag_extra`` (15) so RAG matches SAG's
    typical evidence budget; override with ``rag_k=``.

    ``org_id`` locks org fetch after a clarify re-run (skips ambiguous resolve).
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
    applied = _apply_org_evidence(cleaned, evidence, org_id=org_id)
    if isinstance(applied, OrgClarify):
        return _clarify_result(
            applied, seeds=seeds, evidence=evidence, use_sag=use_sag
        )
    evidence = applied

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
    org_id: str | None = None,
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
        cleaned,
        active.chunks,
        k=hybrid_k,
        vectors=active.vectors,
        faiss_index=active.faiss_index,
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

    # Shared org gate: ambiguous → both arms abstain with the same prompt.
    org_probe = _apply_org_evidence(cleaned, [], org_id=org_id)
    if isinstance(org_probe, OrgClarify):
        clarify_rag = _clarify_result(
            org_probe, seeds=[], evidence=[], use_sag=False
        )
        clarify_sag = _clarify_result(
            org_probe, seeds=[], evidence=[], use_sag=True
        )
        return clarify_rag, clarify_sag

    def _finish(
        evidence: list[LegalChunk],
        use_sag: bool,
        seed_list: list[LegalChunk],
    ) -> ChatResult:
        before = len(evidence)
        applied = _apply_org_evidence(cleaned, evidence, org_id=org_id)
        # Clarify already handled above; applied is always a list here.
        evidence = applied if isinstance(applied, list) else evidence
        org_extra = len(evidence) - before
        seed_ids = {c.chunk_id for c in seed_list}
        # Same draft budget as the K-matched pack (default 15), not the
        # global DEFAULT_MAX_EVIDENCE=10, so RAG and SAG see equal depth.
        # Room for appended org docs so they are not truncated away.
        draft = generate_draft(
            cleaned,
            evidence,
            generate_fn=generate_fn,
            seed_ids=seed_ids,
            max_evidence_chunks=rag_budget + org_extra,
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
