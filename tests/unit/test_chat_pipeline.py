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


def test_compare_query_k_matched_rag_budget():
    """RAG keeps Voyage depth voyage_k+sag_extra; SAG keeps shallow seeds + expand."""
    a = _chunk("luat-ke-toan", "Điều 12", "Kỳ kế toán gồm năm.")
    b = LegalChunk(
        chunk_id="luat-ke-toan::Điều12::Khoản1",
        document_id="luat-ke-toan",
        text="Khoản chi tiết về kỳ kế toán.",
        article="Điều 12",
        clause="Khoản 1",
    )
    c = _chunk("luat-ke-toan", "Điều 40", "Phải kiểm kê tài sản.")
    from sag_legal.sag import build_index

    chunks = [a, b, c]
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
    from sag_legal.sag import ChunkExtraction

    class _Hit:
        def __init__(self, chunk: LegalChunk) -> None:
            self.chunk = chunk

    def fake_hybrid(query, corpus, k=20, vectors=None, **_kwargs):
        return [_Hit(a), _Hit(b), _Hit(c)]

    def fake_rerank(query, documents, top_k=None, client=None, model=None):
        return [_Hit(ch) for ch in documents[:top_k]]

    def fake_query_extract(query: str) -> ChunkExtraction:
        return ChunkExtraction(chunk_id="query")

    original_hybrid = pipe.search_hybrid
    original_rerank = pipe.rerank
    pipe.search_hybrid = fake_hybrid
    pipe.rerank = fake_rerank
    try:
        rag, sag = pipe.compare_query(
            "kỳ kế toán?",
            bundle=bundle,
            voyage_k=1,
            sag_extra=1,
            rag_k=2,
            generate_fn=generate,
            query_extract_fn=fake_query_extract,
        )
    finally:
        pipe.search_hybrid = original_hybrid
        pipe.rerank = original_rerank

    assert rag.use_sag is False
    assert sag.use_sag is True
    assert rag.stats.context_count == 2  # K-matched deep shortlist
    assert sag.stats.seed_count == 1  # shallow Voyage seeds
    assert sag.stats.context_count > 1  # expand added siblings
    assert calls["n"] == 2


def test_answer_query_rag_defaults_to_voyage_plus_extra():
    a = _chunk("luat-ke-toan", "Điều 12", "Kỳ kế toán gồm năm.")
    b = _chunk("luat-ke-toan", "Điều 40", "Phải kiểm kê.")
    chunks = [a, b]
    from sag_legal.sag import build_index

    bundle = CorpusBundle(
        chunks=chunks,
        titles={"luat-ke-toan": "Luật Kế toán"},
        vectors={},
        index=build_index(chunks),
    )

    seen = {}

    def retrieve(query: str, _bundle: CorpusBundle) -> list[LegalChunk]:
        return [a, b]

    def generate(query: str, evidence: Sequence[LegalChunk]) -> DraftAnswer:
        seen["n"] = len(evidence)
        return DraftAnswer(answer="ok", cited_chunk_ids=[a.chunk_id], abstained=False)

    out = answer_query(
        "kỳ kế toán?",
        bundle=bundle,
        use_sag=False,
        voyage_k=1,
        sag_extra=1,
        retrieve_fn=retrieve,
        generate_fn=generate,
    )
    # retrieve_fn returns both; seeds sliced to rag_k = voyage_k + sag_extra = 2
    assert out.use_sag is False
    assert out.stats.seed_count == 2
    assert seen["n"] == 2


def test_answer_query_merges_org_docs_for_clear_org():
    statute = _chunk("luat-tctd", "Điều 1", "Quy định về tổ chức tín dụng.")
    bundle = _bundle([statute])
    seen: dict[str, int | list[str]] = {}

    def retrieve(query: str, _bundle: CorpusBundle) -> list[LegalChunk]:
        return [statute]

    def generate(query: str, evidence: Sequence[LegalChunk]) -> DraftAnswer:
        seen["n"] = len(evidence)
        seen["org_ids"] = [c.document_id for c in evidence if c.document_id.startswith("org:")]
        return DraftAnswer(
            answer="Draft with org context.",
            cited_chunk_ids=[statute.chunk_id],
            abstained=False,
        )

    out = answer_query(
        "VietCredit vốn điều lệ?",
        bundle=bundle,
        retrieve_fn=retrieve,
        generate_fn=generate,
    )
    assert out.abstained is False
    assert seen["n"] == 3  # 1 statute + dkkd + dieu_le stubs
    assert set(seen["org_ids"]) == {"org:VCC:dkkd", "org:VCC:dieu_le"}


def test_answer_query_clarify_ambiguous_org():
    statute = _chunk("luat-tctd", "Điều 1", "Quy định chung.")
    bundle = _bundle([statute])
    calls = {"generate": 0}

    def retrieve(query: str, _bundle: CorpusBundle) -> list[LegalChunk]:
        return [statute]

    def generate(query: str, evidence: Sequence[LegalChunk]) -> DraftAnswer:
        calls["generate"] += 1
        return DraftAnswer(answer="should not run", cited_chunk_ids=[], abstained=False)

    out = answer_query(
        "VietCredit và Waka",
        bundle=bundle,
        retrieve_fn=retrieve,
        generate_fn=generate,
    )
    assert out.abstained is True
    assert "tổ chức nào" in out.answer
    assert calls["generate"] == 0


def test_answer_query_no_org_leaves_evidence_unchanged():
    statute = _chunk("luat-ke-toan", "Điều 12", "Kỳ kế toán gồm năm.")
    bundle = _bundle([statute])
    seen: dict[str, int] = {}

    def retrieve(query: str, _bundle: CorpusBundle) -> list[LegalChunk]:
        return [statute]

    def generate(query: str, evidence: Sequence[LegalChunk]) -> DraftAnswer:
        seen["n"] = len(evidence)
        return DraftAnswer(
            answer="Statute only.",
            cited_chunk_ids=[statute.chunk_id],
            abstained=False,
        )

    out = answer_query(
        "kỳ kế toán năm?",
        bundle=bundle,
        retrieve_fn=retrieve,
        generate_fn=generate,
    )
    assert out.abstained is False
    assert seen["n"] == 1
