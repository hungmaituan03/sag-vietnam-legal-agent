"""Draft-answer tests — fake generate_fn / client keep CI offline."""

from collections.abc import Sequence
from types import SimpleNamespace

from sag_legal.generation import DraftAnswer, generate_draft
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


def test_client_markdown_fence_and_empty_answer_abstains():
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

    out = generate_draft("q?", evidence, client=_FakeClient(), model="fake")
    assert out.abstained is True
    assert "Không đủ căn cứ" in out.answer
