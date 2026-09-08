"""Minimal ingestion: metadata + raw text → LegalDocument + chunks."""

from __future__ import annotations

from pathlib import Path

from sag_legal.chunking.structure import chunk_document
from sag_legal.models import IngestResult, LegalDocument


def ingest_document(document: LegalDocument) -> IngestResult:
    """Parse one in-memory document into structure-aware chunks."""
    chunks = chunk_document(document)
    return IngestResult(document=document, chunks=chunks)


def ingest_text_file(path: Path, document: LegalDocument) -> IngestResult:
    """Load file contents into `document.content`, then ingest.

    `document` should already carry provenance fields (id, type, dates, url).
    The file supplies only the body text for Week-1 simplicity.
    """
    text = path.read_text(encoding="utf-8")
    filled = document.model_copy(update={"content": text})
    return ingest_document(filled)
