"""Public ingestion API."""

from sag_legal.ingestion.corpus import (
    FINANCE_DOC_IDS,
    flatten_chunks,
    ingest_corpus,
    parse_effective_date,
    row_to_document,
)
from sag_legal.ingestion.pipeline import ingest_document, ingest_text_file

__all__ = [
    "FINANCE_DOC_IDS",
    "flatten_chunks",
    "ingest_corpus",
    "ingest_document",
    "ingest_text_file",
    "parse_effective_date",
    "row_to_document",
]
