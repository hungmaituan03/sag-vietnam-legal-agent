"""Voyage AI reranker — runs after hybrid, before SAG.

Design:
- Inject `client` for tests (no API key / network in CI).
- Real path uses voyageai.Client + VOYAGE_API_KEY from env/settings.
"""

from __future__ import annotations

import os
from typing import Any, Protocol

from sag_legal.models import LegalChunk
from sag_legal.retrieval.bm25 import ScoreChunk

DEFAULT_MODEL = "rerank-2.5"


class RerankClient(Protocol):
    def rerank(
        self,
        query: str,
        documents: list[str],
        model: str,
        top_k: int | None = None,
        truncation: bool = True,
    ) -> Any:
        """Must return an object with `.results` items: index, relevance_score."""
        ...


def get_voyage_client() -> Any:
    """Build a real Voyage client from env (never hard-code the key)."""
    import voyageai

    from sag_legal.settings import get_settings

    settings = get_settings()
    api_key = settings.voyage_api_key or os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "VOYAGE_API_KEY is not set. Add it to .env or the environment."
        )
    return voyageai.Client(api_key=api_key)


def rerank(
    query: str,
    chunks: list[LegalChunk],
    top_k: int = 5,
    *,
    client: RerankClient | None = None,
    model: str = DEFAULT_MODEL,
) -> list[ScoreChunk]:
    """Rerank candidate chunks for a query; return top_k ScoreChunks."""
    if not chunks or top_k <= 0:
        return []

    top_k = min(top_k, len(chunks))
    active_client = client if client is not None else get_voyage_client()
    documents = [chunk.text for chunk in chunks]

    response = active_client.rerank(
        query=query,
        documents=documents,
        model=model,
        top_k=top_k,
        truncation=True,
    )

    results: list[ScoreChunk] = []
    for item in response.results:
        chunk = chunks[item.index]
        results.append(ScoreChunk(chunk=chunk, score=float(item.relevance_score)))
    return results
