from datetime import date
from pathlib import Path

from sag_legal.ingestion import ingest_document, ingest_text_file
from sag_legal.retrieval.bm25 import search_bm25, tokenize
from sag_legal.models import DocumentStatus, DocumentType, LegalDocument


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

def _luat_meta(**kwargs) -> LegalDocument:
    base = dict(
        document_id="fixture-luat-tctd-01",
        title="Luật mẫu về tổ chức tín dụng (fixture)",
        document_type=DocumentType.LUAT,
        document_number="01/2024/QH15",
        issuing_authority="Fixture National Assembly",
        issued_date=date(2024, 1, 15),
        effective_date=date(2024, 7, 1),
        status=DocumentStatus.ACTIVE,
        domain="financial_institution",
        source_url="https://example.local/fixtures/luat-tctd",
        content="",
    )
    base.update(kwargs)
    return LegalDocument(**base)

def test_search_bm25():
    query = "cấp giấy phép"
    path = FIXTURES / "luat_tctd_fixture.txt"
    meta = _luat_meta()
    ingest_result = ingest_text_file(path, meta)
    chunks = ingest_result.chunks
    hits = search_bm25(query, chunks, k=3)
    assert len(hits) > 0
    assert hits[0].score >= hits[-1].score

def test_tokenize_strips_punctuation():
    assert "!" not in tokenize("hello!")    