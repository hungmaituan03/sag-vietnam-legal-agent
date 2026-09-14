"""Corpus load + metadata parse for the approved JSON working set.

Tests: tests/unit/test_corpus.py (real snippets from uts_vlc_processed.json).
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

from sag_legal.ingestion.pipeline import ingest_document
from sag_legal.models import (
    DocumentStatus,
    DocumentType,
    IngestResult,
    LegalChunk,
    LegalDocument,
)

# Khung 1 — tổ chức tài chính (Vietcredit-style), using ids present in
# data/raw/uts_vlc_processed.json. Keep the longer demo aliases for TCTD /
# NHNN / Kế toán that the live finance runs already use.
#
# Covered vs brief:
#   TCTD, NHNN, Kế toán (+ sửa đổi), Doanh nghiệp, Thương mại,
#   Chứng khoán (+ sửa đổi), Bảo vệ DLCN, SHTT/nhãn hiệu (+ sửa đổi),
#   Bộ luật Lao động, Phòng chống rửa tiền.
# Not in this JSON dump (follow-up corpus work):
#   Thông tư / Nghị định hướng dẫn; separate "niêm yết" instruments.
# Rows to skip when loading the full dump (corrupt / empty content).
SKIP_DOC_IDS: frozenset[str] = frozenset(
    {
        "code-2019-bo-luat-lao-dong",  # corrupt body in the dump
        "luat-ban-hanh-van-ban-quy-pham-phap-luat",  # empty content
    }
)

KHUNG1_DOC_IDS: tuple[str, ...] = (
    # Core banking / credit / accounting (demo aliases already in use)
    "luat-cac-to-chuc-tin-dung",
    "law-2025-luat-cac-to-chuc-tin-dung-sua-doi",
    "luat-ngan-hang-nha-nuoc",
    "luat-ke-toan",
    "law-2025-luat-ke-toan-sua-doi",
    "law-2020-luat-doanh-nghiep",
    "law-2005-luat-thuong-mai",
    # Securities
    "law-2019-luat-chung-khoan",
    "law-2025-luat-chung-khoan-sua-doi",
    # Personal data
    "law-2025-luat-bao-ve-du-lieu-ca-nhan",
    # Trademark / brand protection via IP law
    "law-2005-luat-so-huu-tri-tue",
    "law-2022-luat-so-huu-tri-tue-sua-doi",
    # Labour (usable copy; not the corrupt code-2019-* row)
    "bo-luat-lao-dong",
    # AML
    "law-2012-luat-phong-chong-rua-tien",
)

# Backward-compatible name used by scripts / chat.
FINANCE_DOC_IDS = KHUNG1_DOC_IDS

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
    for pattern, kind in _EFFECTIVE_DATE_RULES:
        for match in pattern.finditer(content):
            a, b, c = (int(x) for x in match.groups())
            try:
                if kind == "dmy":
                    found.append(date(c, b, a))
            except ValueError:
                continue

    unique = sorted(set(found))
    if len(unique) == 1:
        return unique[0]
    return None


def infer_status(
    effective_date: date | None, as_of: date | None = None
) -> DocumentStatus:
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


def list_corpus_doc_ids(
    path: Path | str,
    *,
    skip: frozenset[str] | set[str] | None = None,
) -> tuple[str, ...]:
    """Return every usable document id in the JSON dump (stable file order).

    Skips ``SKIP_DOC_IDS`` and rows with empty ``content`` so ingest does not
    fail halfway through a full-corpus load.
    """
    corpus_path = Path(path)
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Missing corpus: {corpus_path}")

    with corpus_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list, got {type(payload).__name__}")

    skip_ids = SKIP_DOC_IDS if skip is None else frozenset(skip)
    ids: list[str] = []
    seen: set[str] = set()
    for row in payload:
        if not isinstance(row, dict):
            continue
        doc_id = row.get("id")
        if not doc_id or doc_id in seen or doc_id in skip_ids:
            continue
        content = row.get("content") or ""
        if not str(content).strip():
            continue
        seen.add(doc_id)
        ids.append(doc_id)
    return tuple(ids)


def ingest_corpus(
    path: Path | str,
    doc_ids: tuple[str, ...] | list[str] | None = None,
    *,
    all_docs: bool = False,
) -> list[IngestResult]:
    """Load selected JSON rows, map to LegalDocument, then chunk via pipeline.

    Default is the Khung 1 pack. Pass ``all_docs=True`` (or ``doc_ids`` from
    ``list_corpus_doc_ids``) to ingest the full dump.
    """
    corpus_path = Path(path)
    if all_docs:
        if doc_ids is not None:
            raise ValueError("Pass either all_docs=True or doc_ids=, not both.")
        doc_ids = list_corpus_doc_ids(corpus_path)
    elif doc_ids is None:
        doc_ids = KHUNG1_DOC_IDS

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


def flatten_chunks(results: list[IngestResult]) -> list[LegalChunk]:
    chunks: list[LegalChunk] = []
    for result in results:
        chunks.extend(result.chunks)
    return chunks
