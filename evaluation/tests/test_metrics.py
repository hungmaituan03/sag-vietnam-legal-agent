"""Unit tests for evaluation scoring helpers (offline)."""

from pathlib import Path

from evaluation.metrics import (
    GoldItem,
    ProvisionRef,
    f1,
    load_gold,
    score_answer,
    score_retrieval,
)
from sag_legal.models import LegalChunk


def _chunk(
    doc: str,
    article: str,
    text: str,
    *,
    clause: str | None = None,
    point: str | None = None,
) -> LegalChunk:
    return LegalChunk(
        chunk_id=f"{doc}::{article}",
        document_id=doc,
        text=text,
        article=article,
        clause=clause,
        point=point,
    )


def test_load_gold_khung1():
    path = Path(__file__).resolve().parents[1] / "datasets" / "khung1_gold_v0.jsonl"
    items = load_gold(path)
    assert len(items) >= 3
    assert items[0].query
    assert items[0].must_provisions


def test_score_retrieval_orphan_case():
    gold = GoldItem(
        id="t",
        query="q",
        must_docs=["luat-ke-toan"],
        must_provisions=[
            ProvisionRef("luat-ke-toan", "Điều 40", "Khoản 2"),
        ],
    )
    rag = [
        _chunk(
            "luat-ke-toan",
            "Điều 40",
            "a) Cuối kỳ kế toán năm;",
            clause="Khoản 2",
            point="Điểm a",
        )
    ]
    # Điểm a matches article but not exact clause-only ref if we require clause
    # Our matcher: clause must equal — Điểm a has clause Khoản 2, so it matches!
    rag_score = score_retrieval(rag, gold)
    assert rag_score.provision_recall == 1.0

    # Missing stem article heading still ok if khoản present
    sag = rag + [
        _chunk(
            "luat-ke-toan",
            "Điều 40",
            "2. Đơn vị kế toán phải kiểm kê tài sản...",
            clause="Khoản 2",
        )
    ]
    assert score_retrieval(sag, gold).provision_recall == 1.0


def test_score_retrieval_missing_doc():
    gold = GoldItem(
        id="t",
        query="q",
        must_docs=["law-2019-luat-chung-khoan"],
        must_provisions=[
            ProvisionRef("law-2019-luat-chung-khoan", "Điều 118"),
        ],
    )
    only_dn = [_chunk("law-2020-luat-doanh-nghiep", "Điều 176", "công bố")]
    score = score_retrieval(only_dn, gold)
    assert score.doc_recall == 0.0
    assert score.provision_recall == 0.0


def test_score_answer_phrases():
    gold = GoldItem(
        id="t",
        query="q",
        must_phrases=["kiểm kê", "12 tháng"],
    )
    out = score_answer("Phải kiểm kê tài sản vào cuối kỳ.", gold)
    assert out.phrase_recall == 0.5
    assert out.missing_phrases == ["12 tháng"]


def test_f1():
    assert f1(1.0, 1.0) == 1.0
    assert f1(0.0, 1.0) == 0.0
    assert abs(f1(0.5, 0.5) - 0.5) < 1e-9
