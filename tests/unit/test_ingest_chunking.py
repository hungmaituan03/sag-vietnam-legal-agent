"""Week-1 ingest + structure-aware chunking tests."""

from datetime import date
from pathlib import Path

from sag_legal.ingestion import ingest_document, ingest_text_file
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


def test_chunk_preserves_dieu_khoan_diem():
    path = FIXTURES / "luat_tctd_fixture.txt"
    result = ingest_text_file(path, _luat_meta())

    assert result.document.document_id == "fixture-luat-tctd-01"
    assert len(result.chunks) >= 5

    diem = [c for c in result.chunks if c.point == "Điểm a"]
    assert diem, "expected Điểm a under Điều 2 Khoản 2"
    assert diem[0].article == "Điều 2"
    assert diem[0].clause == "Khoản 2"
    assert "Nhận tiền gửi" in diem[0].text
    assert "Điều 2, Khoản 2, Điểm a" == diem[0].citation_path


def test_ingest_thong_tu_links_conceptually_to_luat_article():
    """Fixture designed for later SAG multi-hop (Thông tư → Luật Điều 10)."""
    meta = LegalDocument(
        document_id="fixture-tt-02",
        title="Thông tư mẫu hướng dẫn Luật TCTD (fixture)",
        document_type=DocumentType.THONG_TU,
        document_number="02/2024/TT-NHNN",
        issuing_authority="Fixture SBV",
        issued_date=date(2024, 8, 1),
        effective_date=date(2024, 8, 15),
        status=DocumentStatus.ACTIVE,
        domain="financial_institution",
        source_url="https://example.local/fixtures/thong-tu",
        content="",
    )
    result = ingest_text_file(FIXTURES / "thong_tu_fixture.txt", meta)
    joined = "\n".join(c.text for c in result.chunks)
    assert "Điều 10 Luật mẫu" in joined
    assert any(c.article == "Điều 3" for c in result.chunks)


def test_ingest_document_without_file():
    doc = _luat_meta(
        content=(
            "Điều 5. Kiểm tra\n"
            "1. Nội dung khoản một.\n"
            "2. Nội dung khoản hai.\n"
        )
    )
    result = ingest_document(doc)
    assert [c.clause for c in result.chunks if c.clause] == ["Khoản 1", "Khoản 2"]


def test_no_article_falls_back_to_body_chunk():
    doc = _luat_meta(content="Văn bản không có đánh số Điều.")
    result = ingest_document(doc)
    assert len(result.chunks) == 1
    assert result.chunks[0].article is None
    assert result.chunks[0].citation_path == "(document-level)"
