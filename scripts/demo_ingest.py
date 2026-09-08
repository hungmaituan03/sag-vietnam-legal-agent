#!/usr/bin/env python3
"""Tiny demo: ingest fixtures and print citation paths.

Usage (from repo root, venv active):
  python scripts/demo_ingest.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from sag_legal.ingestion import ingest_text_file
from sag_legal.models import DocumentStatus, DocumentType, LegalDocument

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def main() -> None:
    docs = [
        (
            FIXTURES / "luat_tctd_fixture.txt",
            LegalDocument(
                document_id="fixture-luat-tctd-01",
                title="Luật mẫu TCTD (fixture)",
                document_type=DocumentType.LUAT,
                document_number="01/2024/QH15",
                effective_date=date(2024, 7, 1),
                status=DocumentStatus.ACTIVE,
                source_url="https://example.local/fixtures/luat-tctd",
                content="",
            ),
        ),
        (
            FIXTURES / "thong_tu_fixture.txt",
            LegalDocument(
                document_id="fixture-tt-02",
                title="Thông tư mẫu (fixture)",
                document_type=DocumentType.THONG_TU,
                document_number="02/2024/TT-NHNN",
                effective_date=date(2024, 8, 15),
                status=DocumentStatus.ACTIVE,
                source_url="https://example.local/fixtures/thong-tu",
                content="",
            ),
        ),
    ]

    for path, meta in docs:
        result = ingest_text_file(path, meta)
        print(f"\n=== {result.document.title} ({len(result.chunks)} chunks) ===")
        for chunk in result.chunks:
            preview = chunk.text.replace("\n", " ")[:80]
            print(f"- {chunk.citation_path:30} | {preview}")


if __name__ == "__main__":
    main()
