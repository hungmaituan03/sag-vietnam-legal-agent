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
from sag_legal.ingestion import FINANCE_DOC_IDS, flatten_chunks, ingest_corpus
from sag_legal.models import LegalChunk
from sag_legal.reranking import rerank
from sag_legal.retrieval.embeddings import embed_chunks
from sag_legal.retrieval.hybrid import search_hybrid
from sag_legal.sag import EventEntityIndex, build_index, expand
from sag_legal.settings import get_settings


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return here.parents[3]


ROOT = _repo_root()
RAW_JSON = ROOT / "data" / "raw" / "uts_vlc_processed.json"
CACHE_NPZ = ROOT / "data" / "processed" / "finance_embeddings.npz"

RetrieveFn = Callable[..., list[LegalChunk]]
GenerateFn = Callable[[str, Sequence[LegalChunk]], DraftAnswer]


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

        results = ingest_corpus(path, doc_ids=FINANCE_DOC_IDS)
        titles = {r.document.document_id: r.document.title for r in results}
        chunks = flatten_chunks(results)
        vectors = embed_chunks(chunks, cache_path=cache_npz or CACHE_NPZ)
        index = build_index(chunks, vectors=vectors, min_sim=min_sim)
        _bundle = CorpusBundle(
            chunks=chunks, titles=titles, vectors=vectors, index=index
        )
        return _bundle


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
    hops: int = 1,
    min_sim: float = 0.6,
    use_sag: bool = True,
    voyage_client: Any | None = None,
    generate_fn: GenerateFn | None = None,
    retrieve_fn: RetrieveFn | None = None,
) -> ChatResult:
    """Run retrieval → draft. Inject retrieve_fn / generate_fn for offline tests.

    ``use_sag=False`` is the traditional RAG baseline: Voyage shortlist only
    (same as ``--no-sag`` on the finance script).
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

    if retrieve_fn is not None:
        evidence = retrieve_fn(cleaned, active)
        seeds = evidence[:voyage_k]
    else:
        seeds, evidence = _default_retrieve(
            cleaned,
            active,
            hybrid_k=hybrid_k,
            voyage_k=voyage_k,
            sag_extra=sag_extra,
            hops=hops,
            min_sim=min_sim,
            use_sag=use_sag,
            voyage_client=voyage_client,
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
    hops: int = 1,
    min_sim: float = 0.6,
    voyage_client: Any | None = None,
    generate_fn: GenerateFn | None = None,
) -> tuple[ChatResult, ChatResult]:
    """Same Voyage seeds → RAG baseline + SAG expansion (fair ablation)."""
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
    seeds, _rag_evidence = _default_retrieve(
        cleaned,
        active,
        hybrid_k=hybrid_k,
        voyage_k=voyage_k,
        sag_extra=sag_extra,
        hops=hops,
        min_sim=min_sim,
        use_sag=False,
        voyage_client=voyage_client,
    )
    sag_evidence = expand(
        seeds,
        active.index,
        max_extra=sag_extra,
        hops=hops,
        min_sim=min_sim,
    )

    def _finish(evidence: list[LegalChunk], use_sag: bool) -> ChatResult:
        seed_ids = {c.chunk_id for c in seeds}
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
                seed_count=len(seeds),
                context_count=len(evidence),
                sag_added=len([c for c in evidence if c.chunk_id not in seed_ids]),
            ),
            use_sag=use_sag,
        )

    return _finish(list(seeds), False), _finish(sag_evidence, True)


def keys_configured() -> tuple[bool, bool]:
    settings = get_settings()
    return settings.voyage_configured, settings.qwen_configured
