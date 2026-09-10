"""Corpus load + metadata parse for the approved JSON working set.

Tests: tests/unit/test_corpus.py (real snippets from uts_vlc_processed.json).
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

from sag_legal.ingestion.pipeline import ingest_document
from sag_legal.models import DocumentStatus, DocumentType, IngestResult, LegalDocument

""" Applicable laws for demos only, delete later """
FINANCE_DOC_IDS = (
  "luat-cac-to-chuc-tin-dung",
  "luat-ngan-hang-nha-nuoc",
  "luat-ke-toan",
  "law-2020-luat-doanh-nghiep",
  "law-2005-luat-thuong-mai",
)

JSON_TYPE_TO_DOCUMENT_TYPE = {
  "law": DocumentType.LUAT,
  "code": DocumentType.OTHER,
  "constitution": DocumentType.OTHER,
}

_EFFECTIVE_DATE_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"Luật này có hiệu lực thi hành từ ngày\s+"
            r"(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
            re.IGNORECASE,
        ),
        "dmy",
    ),
)

def parse_effective_date(content: str) -> date | None:
    """Return the document force date when exactly one clear match exists.

    Prefer phrases like:
      Luật này có hiệu lực thi hành từ ngày DD tháng MM năm YYYY
    Return None if missing, invalid, or ambiguous (multiple distinct dates).
    """
    if not content:
      return None 
    
    found: list[date] = []
    for pattern,kind in _EFFECTIVE_DATE_RULES:
      for match in pattern.finditer(content):
        a, b, c = (int(x) for x in match.groups())
        try: 
          if kind == "dmy":
            found.append(date(c,b,a))
        except ValueError:
          continue
    
    unique = sorted(set(found))
    if len(unique) == 1:
      return unique[0]
    return None 

def infer_status(effective_date: date | None, as_of: date | None = None) -> DocumentStatus:
    as_of = as_of or date.today()
    if effective_date is None:
        return DocumentStatus.UNKNOWN
    if effective_date > as_of:
        return DocumentStatus.FUTURE
    return DocumentStatus.UNKNOWN

def row_to_document(row: dict) -> LegalDocument:
    content = row.get("content") or ""
    if not content.strip():
        raise ValueError(f"Empty content for document id={row.get('id')!r}")
    raw_type = (row.get("type") or "").strip().lower()
    document_type = JSON_TYPE_TO_DOCUMENT_TYPE.get(raw_type, DocumentType.OTHER)
    effective = parse_effective_date(content)
    return LegalDocument(
        document_id=row["id"],
        title=row.get("title") or row["id"],
        document_type=document_type,
        effective_date=effective,
        status=DocumentStatus.UNKNOWN,
        source_url=None,
        content=content,
    )

def ingest_corpus(
    path: Path | str,
    doc_ids: tuple[str, ...] | list[str] | None = None,
) -> list[IngestResult]:
    """Load selected JSON rows, map to LegalDocument, then chunk via pipeline."""
    
    if doc_ids is None:
        doc_ids = FINANCE_DOC_IDS

    corpus_path = Path(path)
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Missing corpus: {corpus_path}")

    with corpus_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list, got {type(payload).__name__}")

    by_id = {row["id"]: row for row in payload if isinstance(row, dict) and "id" in row}
    missing = [doc_id for doc_id in doc_ids if doc_id not in by_id]
    if missing:
        raise KeyError(f"Document ids not found in corpus: {missing}")

    results: list[IngestResult] = []
    for doc_id in doc_ids:
        document = row_to_document(by_id[doc_id])
        results.append(ingest_document(document))
    return results
