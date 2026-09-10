"""Unit tests for corpus metadata parse + row mapping.

Snippets for effective_date are copied from data/raw/uts_vlc_processed.json
(finance pack). Do not load the full JSON here — keep CI fast.
"""

import json
from datetime import date
from pathlib import Path

import pytest

from sag_legal.ingestion.corpus import (
    flatten_chunks,
    ingest_corpus,
    parse_effective_date,
    row_to_document,
)
from sag_legal.models import DocumentStatus, DocumentType, LegalChunk
from sag_legal.retrieval.bm25 import search_bm25

# --- parse_effective_date -------------------------------------------------


@pytest.mark.parametrize(
    ("content", "expected", "source"),
    [
        (
            "Luật này có hiệu lực thi hành từ ngày 01 tháng 01 năm 2011.",
            date(2011, 1, 1),
            "luat-cac-to-chuc-tin-dung",
        ),
        (
            "Luật này có hiệu lực thi hành từ ngày 01 tháng 01 năm 2011.",
            date(2011, 1, 1),
            "luat-ngan-hang-nha-nuoc",
        ),
        (
            "Luật này có hiệu lực thi hành từ ngày 01 tháng 01 năm 2017.",
            date(2017, 1, 1),
            "luat-ke-toan",
        ),
        (
            "Luật này có hiệu lực thi hành từ ngày 01 tháng 01 năm 2021.",
            date(2021, 1, 1),
            "law-2020-luat-doanh-nghiep",
        ),
        (
            "1. Luật này có hiệu lực thi hành từ ngày 01 tháng 01 năm 2017.\n"
            "2. Luật kế toán số 03/2003/QH11 hết hiệu lực kể từ ngày "
            "Luật này có hiệu lực thi hành.",
            date(2017, 1, 1),
            "luat-ke-toan (force + hết hiệu lực)",
        ),
    ],
)
def test_parse_effective_date_accepts_clear_force_phrase(content, expected, source):
    assert parse_effective_date(content) == expected, source


@pytest.mark.parametrize(
    ("content", "source"),
    [
        (
            "Kỳ kế toán đầu tiên của đơn vị kế toán khác tính từ đầu ngày "
            "quyết định thành lập đơn vị kế toán có hiệu lực đến hết ngày "
            "cuối cùng của kỳ kế toán năm.",
            "luat-ke-toan (kỳ kế toán)",
        ),
        (
            "Bị kết tội bằng bản án của Tòa án đã có hiệu lực pháp luật.",
            "luat-ke-toan (bản án)",
        ),
        (
            "g) Khi hợp đồng thuê Tổng giám đốc (Giám đốc) hết hiệu lực;",
            "luat-cac-to-chuc-tin-dung (hợp đồng)",
        ),
        (
            "Luật Ngân hàng Nhà nước Việt Nam số 10/2003/QH11 hết hiệu lực "
            "kể từ ngày Luật này có hiệu lực.",
            "luat-ngan-hang-nha-nuoc (repeal, no target date)",
        ),
        (
            "Đối với các hợp đồng cấp tín dụng được ký kết trước ngày "
            "Luật này có hiệu lực thi hành, tổ chức tín dụng và khách hàng "
            "được tiếp tục thực hiện.",
            "luat-cac-to-chuc-tin-dung (trước ngày Luật này)",
        ),
        (
            "Luật này có hiệu lực thi hành từ ngày 01 tháng 01 năm 2011.\n"
            "Luật này có hiệu lực thi hành từ ngày 01 tháng 01 năm 2017.",
            "ambiguous (TCTD 2011 + kế toán 2017)",
        ),
        ("", "empty"),
    ],
)
def test_parse_effective_date_rejects_missing_or_ambiguous(content, source):
    assert parse_effective_date(content) is None, source


# --- row_to_document ------------------------------------------------------


def test_row_to_document_maps_law_and_effective_date():
    row = {
        "id": "luat-ngan-hang-nha-nuoc",
        "title": "Luat Ngan Hang Nha Nuoc",
        "type": "law",
        "content": (
            "Điều 1. Phạm vi\n"
            "Luật này có hiệu lực thi hành từ ngày 01 tháng 01 năm 2011.\n"
        ),
    }
    doc = row_to_document(row)
    assert doc.document_id == "luat-ngan-hang-nha-nuoc"
    assert doc.title == "Luat Ngan Hang Nha Nuoc"
    assert doc.document_type == DocumentType.LUAT
    assert doc.effective_date == date(2011, 1, 1)
    assert doc.status == DocumentStatus.UNKNOWN
    assert doc.source_url is None
    assert "Điều 1" in doc.content


def test_row_to_document_unknown_type_and_missing_date():
    row = {
        "id": "fixture-other-01",
        "title": "Văn bản khác",
        "type": "circular",
        "content": "Không có cụm từ hiệu lực thi hành từ ngày.",
    }
    doc = row_to_document(row)
    assert doc.document_type == DocumentType.OTHER
    assert doc.effective_date is None


def test_row_to_document_title_falls_back_to_id():
    row = {
        "id": "law-without-title",
        "type": "law",
        "content": "Luật này có hiệu lực thi hành từ ngày 01 tháng 01 năm 2017.",
    }
    doc = row_to_document(row)
    assert doc.title == "law-without-title"
    assert doc.effective_date == date(2017, 1, 1)


def test_row_to_document_rejects_empty_content():
    with pytest.raises(ValueError):
        row_to_document({"id": "empty", "title": "x", "type": "law", "content": ""})


# --- ingest_corpus --------------------------------------------------------


def _write_mini_corpus(path: Path, rows: list[dict]) -> Path:
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return path


def test_ingest_corpus_chunks_dieu_khoan_diem_and_copies_date(tmp_path: Path):
    rows = [
        {
            "id": "mini-tctd",
            "title": "Luật mẫu TCTD",
            "type": "law",
            "content": (
                "Điều 1. Phạm vi\n"
                "Luật này quy định về tổ chức tín dụng.\n"
                "Điều 2. Giấy phép\n"
                "1. Điều kiện cấp giấy phép.\n"
                "a) Nhận tiền gửi.\n"
                "Luật này có hiệu lực thi hành từ ngày 01 tháng 01 năm 2011.\n"
            ),
        },
        {
            "id": "skip-me",
            "title": "Not selected",
            "type": "law",
            "content": "Điều 9. Không được chọn.\n",
        },
    ]
    corpus = _write_mini_corpus(tmp_path / "mini.json", rows)

    results = ingest_corpus(corpus, doc_ids=("mini-tctd",))

    assert len(results) == 1
    result = results[0]
    assert result.document.document_id == "mini-tctd"
    assert result.document.effective_date == date(2011, 1, 1)
    assert result.document.status == DocumentStatus.UNKNOWN

    articles = {c.article for c in result.chunks}
    assert "Điều 1" in articles
    assert "Điều 2" in articles

    diem = [c for c in result.chunks if c.point == "Điểm a"]
    assert diem, "expected Điểm a under Điều 2 Khoản 1"
    assert diem[0].article == "Điều 2"
    assert diem[0].clause == "Khoản 1"
    assert diem[0].effective_date == date(2011, 1, 1)
    assert diem[0].citation_path == "Điều 2, Khoản 1, Điểm a"


def test_ingest_corpus_missing_id_raises(tmp_path: Path):
    corpus = _write_mini_corpus(
        tmp_path / "mini.json",
        [{"id": "only-this", "title": "x", "type": "law", "content": "Điều 1. A\n"}],
    )
    with pytest.raises(KeyError, match="not found"):
        ingest_corpus(corpus, doc_ids=("missing-id",))


def test_ingest_corpus_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        ingest_corpus(tmp_path / "no-such.json", doc_ids=("mini-tctd",))


# --- flatten_chunks -------------------------------------------------------


_MINI_ROWS = [
    {
        "id": "mini-tctd",
        "title": "Luật mẫu TCTD",
        "type": "law",
        "content": (
            "Điều 1. Phạm vi\n"
            "Luật này quy định về tổ chức tín dụng.\n"
            "Điều 2. Giấy phép\n"
            "1. Điều kiện cấp giấy phép.\n"
            "a) Nhận tiền gửi.\n"
            "Luật này có hiệu lực thi hành từ ngày 01 tháng 01 năm 2011.\n"
        ),
    },
    {
        "id": "mini-other",
        "title": "Luật mẫu khác",
        "type": "law",
        "content": "Điều 3. Không liên quan.\n",
    },
]


def test_flatten_chunks_empty():
    assert flatten_chunks([]) == []


def test_flatten_chunks_concatenates_in_result_order(tmp_path: Path):
    corpus = _write_mini_corpus(tmp_path / "mini.json", _MINI_ROWS)
    results = ingest_corpus(corpus, doc_ids=("mini-tctd", "mini-other"))
    flat = flatten_chunks(results)

    expected_len = sum(len(r.chunks) for r in results)
    assert len(flat) == expected_len
    assert all(isinstance(c, LegalChunk) for c in flat)
    assert [c.document_id for c in flat] == (
        ["mini-tctd"] * len(results[0].chunks)
        + ["mini-other"] * len(results[1].chunks)
    )


def test_search_bm25_on_flattened_mini_corpus(tmp_path: Path):
    corpus = _write_mini_corpus(tmp_path / "mini.json", _MINI_ROWS)
    chunks = flatten_chunks(ingest_corpus(corpus, doc_ids=("mini-tctd", "mini-other")))
    hits = search_bm25("cấp giấy phép", chunks, k=3)

    assert hits
    assert hits[0].score >= hits[-1].score
    assert "giấy phép" in hits[0].chunk.text.lower()
