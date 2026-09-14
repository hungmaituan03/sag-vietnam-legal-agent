"""Draft-answer tests — fake generate_fn / client keep CI offline."""

from collections.abc import Sequence
from types import SimpleNamespace

from sag_legal.generation import DraftAnswer, InvalidDraftError, generate_draft
from sag_legal.generation.draft import select_evidence_for_draft
from sag_legal.models import LegalChunk


def _chunk(doc: str, article: str, text: str) -> LegalChunk:
    return LegalChunk(
        chunk_id=f"{doc}::{article.replace(' ', '')}",
        document_id=doc,
        text=text,
        article=article,
    )


def test_empty_evidence_abstains():
    out = generate_draft("kỳ kế toán năm là gì?", [])
    assert out.abstained is True
    assert out.cited_chunk_ids == []
    assert "Không có" in out.answer


def test_generate_fn_bypass():
    evidence = [_chunk("luat-ke-toan", "Điều 12", "Kỳ kế toán gồm năm, quý, tháng.")]

    def fake(query: str, chunks: Sequence[LegalChunk]) -> DraftAnswer:
        assert "kỳ kế toán" in query
        assert len(chunks) == 1
        return DraftAnswer(
            answer="Kỳ kế toán gồm năm, quý và tháng.",
            cited_chunk_ids=[chunks[0].chunk_id],
            abstained=False,
        )

    out = generate_draft("kỳ kế toán năm?", evidence, generate_fn=fake)
    assert out.abstained is False
    assert out.cited_chunk_ids == [evidence[0].chunk_id]
    assert "năm" in out.answer


def test_client_json_parsed_and_unknown_ids_dropped():
    evidence = [
        _chunk("luat-ke-toan", "Điều 12", "1. Kỳ kế toán gồm kỳ kế toán năm."),
    ]
    payload = {
        "answer": "Kỳ kế toán năm là một loại kỳ kế toán.",
        "cited_chunk_ids": [evidence[0].chunk_id, "invented::id"],
        "abstained": False,
    }

    class _Msg:
        content = __import__("json").dumps(payload, ensure_ascii=False)

    class _FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**_kwargs):
                    return SimpleNamespace(choices=[SimpleNamespace(message=_Msg())])

    out = generate_draft(
        "kỳ kế toán năm?",
        evidence,
        client=_FakeClient(),
        model="fake-model",
    )
    assert out.abstained is False
    assert out.cited_chunk_ids == [evidence[0].chunk_id]
    assert "Kỳ kế toán năm" in out.answer


def test_client_markdown_fence_and_empty_answer_raises():
    evidence = [_chunk("luat-ke-toan", "Điều 12", "text")]
    raw = '```json\n{"answer": "", "cited_chunk_ids": [], "abstained": false}\n```'

    class _Msg:
        content = raw

    class _FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**_kwargs):
                    return SimpleNamespace(choices=[SimpleNamespace(message=_Msg())])

    try:
        generate_draft("q?", evidence, client=_FakeClient(), model="fake")
        raise AssertionError("expected InvalidDraftError")
    except InvalidDraftError as exc:
        assert "empty" in str(exc).lower()


def test_select_evidence_caps_seed_first_order():
    chunks = [_chunk("d", f"Điều {i}", f"text {i}") for i in range(12)]
    selected = select_evidence_for_draft(chunks, max_chunks=8)
    assert len(selected) == 8
    assert selected[0].chunk_id == chunks[0].chunk_id
    assert selected[-1].chunk_id == chunks[7].chunk_id


def test_select_evidence_prefers_same_article_repairs():
    """SAG parents must beat other-seed headings when the draft budget is tight."""
    seed_a = LegalChunk(
        chunk_id="doc::A::điểma",
        document_id="doc",
        text="a) Cuối kỳ kế toán năm;",
        article="Điều 40",
        clause="Khoản 2",
        point="Điểm a",
    )
    seed_b = LegalChunk(
        chunk_id="doc::B::điểma",
        document_id="doc",
        text="a) Lập báo cáo tài chính;",
        article="Điều 29",
        clause="Khoản 2",
        point="Điểm a",
    )
    other_heading = LegalChunk(
        chunk_id="doc::B",
        document_id="doc",
        text="Điều 29. Báo cáo tài chính",
        article="Điều 29",
    )
    repair_heading = LegalChunk(
        chunk_id="doc::A",
        document_id="doc",
        text="Điều 40. Kiểm kê tài sản",
        article="Điều 40",
    )
    repair_stem = LegalChunk(
        chunk_id="doc::A::khoản2",
        document_id="doc",
        text="2. Đơn vị kế toán phải kiểm kê tài sản trong các trường hợp sau đây:",
        article="Điều 40",
        clause="Khoản 2",
    )
    # Expand order: seeds, then other heading before the useful repairs.
    evidence = [seed_b, seed_a, other_heading, repair_heading, repair_stem]
    selected = select_evidence_for_draft(
        evidence,
        max_chunks=4,
        seed_ids={seed_a.chunk_id, seed_b.chunk_id},
    )
    ids = [c.chunk_id for c in selected]
    assert seed_a.chunk_id in ids and seed_b.chunk_id in ids
    assert repair_heading.chunk_id in ids
    assert repair_stem.chunk_id in ids
    assert other_heading.chunk_id not in ids


def test_truncated_json_raises_instead_of_soft_repair():
    evidence = [_chunk("luat-ke-toan", "Điều 12", "Kỳ kế toán năm.")]
    raw = (
        '{\n  "answer": "Kỳ kế toán năm là 12 tháng.",\n'
        '  "cited_chunk_ids": ["luat-ke-toan::Điều12",\n  "absta'
    )

    class _Msg:
        content = raw

    class _FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**_kwargs):
                    return SimpleNamespace(choices=[SimpleNamespace(message=_Msg())])

    try:
        generate_draft("q?", evidence, client=_FakeClient(), model="fake")
        raise AssertionError("expected InvalidDraftError")
    except InvalidDraftError as exc:
        assert "invalid json" in str(exc).lower()


def test_answer_without_valid_citations_raises():
    evidence = [_chunk("luat-ke-toan", "Điều 12", "text")]
    payload = {
        "answer": "Một câu trả lời không trích dẫn.",
        "cited_chunk_ids": ["invented::id"],
        "abstained": False,
    }

    class _Msg:
        content = __import__("json").dumps(payload, ensure_ascii=False)

    class _FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**_kwargs):
                    return SimpleNamespace(choices=[SimpleNamespace(message=_Msg())])

    try:
        generate_draft("q?", evidence, client=_FakeClient(), model="fake")
        raise AssertionError("expected InvalidDraftError")
    except InvalidDraftError as exc:
        assert "cite" in str(exc).lower()
