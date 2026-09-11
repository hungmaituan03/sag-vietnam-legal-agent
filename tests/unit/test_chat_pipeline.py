"""Chat pipeline tests — injectable retrieve/generate keep CI offline."""

from collections.abc import Sequence

from sag_legal.chat.pipeline import (
    ChatResult,
    CorpusBundle,
    answer_query,
)
from sag_legal.generation import DraftAnswer
from sag_legal.models import LegalChunk
from sag_legal.sag import EventEntityIndex


def _chunk(doc: str, article: str, text: str) -> LegalChunk:
    return LegalChunk(
        chunk_id=f"{doc}::{article.replace(' ', '')}",
        document_id=doc,
        text=text,
        article=article,
    )


def _bundle(chunks: list[LegalChunk]) -> CorpusBundle:
    titles = {c.document_id: c.document_id for c in chunks}
    return CorpusBundle(
        chunks=chunks,
        titles=titles,
        vectors={},
        index=EventEntityIndex(),
    )


def test_empty_query_abstains():
    out = answer_query("   ", bundle=_bundle([]))
    assert out.abstained is True
    assert "Vui lòng" in out.answer


def test_answer_query_uses_injectable_fns():
    a = _chunk("luat-ke-toan", "Điều 12", "Kỳ kế toán gồm năm, quý, tháng.")
    b = _chunk("luat-ke-toan", "Điều 40", "Cuối kỳ kế toán năm phải kiểm kê.")
    bundle = _bundle([a, b])

    def retrieve(query: str, _bundle: CorpusBundle) -> list[LegalChunk]:
        assert "kỳ kế toán" in query
        return [a, b]

    def generate(query: str, evidence: Sequence[LegalChunk]) -> DraftAnswer:
        assert len(evidence) == 2
        return DraftAnswer(
            answer="Kỳ kế toán năm là một loại kỳ kế toán.",
            cited_chunk_ids=[a.chunk_id],
            abstained=False,
        )

    out = answer_query(
        "kỳ kế toán năm?",
        bundle=bundle,
        retrieve_fn=retrieve,
        generate_fn=generate,
    )
    assert isinstance(out, ChatResult)
    assert out.abstained is False
    assert out.cited[0].chunk_id == a.chunk_id
    assert out.stats.context_count == 2
    assert "Kỳ kế toán năm" in out.answer
    assert out.use_sag is True


def test_use_sag_false_flag_on_result():
    a = _chunk("luat-ke-toan", "Điều 12", "Kỳ kế toán gồm năm.")
    bundle = _bundle([a])

    def retrieve(query: str, _bundle: CorpusBundle) -> list[LegalChunk]:
        return [a]

    def generate(query: str, evidence: Sequence[LegalChunk]) -> DraftAnswer:
        return DraftAnswer(
            answer="Baseline RAG.",
            cited_chunk_ids=[a.chunk_id],
            abstained=False,
        )

    out = answer_query(
        "kỳ kế toán?",
        bundle=bundle,
        use_sag=False,
        retrieve_fn=retrieve,
        generate_fn=generate,
    )
    assert out.use_sag is False
    assert out.stats.sag_added == 0


def test_compare_query_shares_seed_counts():
    a = _chunk("luat-ke-toan", "Điều 12", "Kỳ kế toán gồm năm.")
    b = LegalChunk(
        chunk_id="luat-ke-toan::Điều12::Khoản1",
        document_id="luat-ke-toan",
        text="Khoản chi tiết về kỳ kế toán.",
        article="Điều 12",
        clause="Khoản 1",
    )
    from sag_legal.sag import build_index

    chunks = [a, b]
    bundle = CorpusBundle(
        chunks=chunks,
        titles={"luat-ke-toan": "Luật Kế toán"},
        vectors={},
        index=build_index(chunks),
    )

    calls = {"n": 0}

    def generate(query: str, evidence: Sequence[LegalChunk]) -> DraftAnswer:
        calls["n"] += 1
        return DraftAnswer(
            answer=f"arm-{calls['n']} len={len(evidence)}",
            cited_chunk_ids=[evidence[0].chunk_id],
            abstained=False,
        )

    import sag_legal.chat.pipeline as pipe

    def fake_retrieve(query, active_bundle, **kwargs):
        return [a], [a]

    original = pipe._default_retrieve
    pipe._default_retrieve = fake_retrieve
    try:
        rag, sag = pipe.compare_query(
            "kỳ kế toán?", bundle=bundle, generate_fn=generate
        )
    finally:
        pipe._default_retrieve = original

    assert rag.use_sag is False
    assert sag.use_sag is True
    assert rag.stats.seed_count == sag.stats.seed_count == 1
    assert rag.stats.context_count == 1
    assert sag.stats.context_count > 1
    assert sag.stats.sag_added >= 1
    assert calls["n"] == 2
