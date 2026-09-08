"""Public ingestion API."""

from sag_legal.ingestion.pipeline import ingest_document, ingest_text_file

__all__ = ["ingest_document", "ingest_text_file"]
