"""Extraction tests — fake extract_fn keeps CI offline (no Qwen, no key)."""

import json
from pathlib import Path

from sag_legal.models import LegalChunk
from sag_legal.sag import (
    ChunkExtraction,
    ExtractedEntity,
    ExtractedEvent,
    build_index,
    concept_key,
    concept_keys_by_chunk,
    expand,
    extract_chunk,
    extract_chunks,
    normalize_entity_name,
)


def _chunk(doc: str, article: str, text: str, clause: str | None = None) -> LegalChunk:
    parts = [doc, article.replace(" ", "")]
    if clause:
        parts.append(clause.replace(" ", ""))
    return LegalChunk(
        chunk_id="::".join(parts),
        document_id=doc,
        text=text,
        article=article,
        clause=clause,
    )


def fake_extract(chunk: LegalChunk) -> ChunkExtraction:
    if "kiểm kê" in chunk.text.lower() or "kỳ kế toán" in chunk.text.lower():
        return ChunkExtraction(
            chunk_id=chunk.chunk_id,
            events=[
                ExtractedEvent(
                    event_id="inventory_year_end",
                    summary="Phải kiểm kê tài sản cuối kỳ kế toán năm",
                    actor="đơn vị kế toán",
                    action="kiểm kê tài sản",
                    modality="phải",
                    trigger="cuối kỳ kế toán năm",
                )
            ],
            entities=[
                ExtractedEntity(name="Đơn vị kế toán", type="organization"),
                ExtractedEntity(
                    name="kiểm kê tài sản",
                    type="obligation",
                    aliases=["kiểm kê"],
                ),
                ExtractedEntity(name="cuối kỳ kế toán năm", type="time"),
            ],
        )
    return ChunkExtraction(chunk_id=chunk.chunk_id)


def test_normalize_entity_name_casefolds_for_joining():
    assert normalize_entity_name("Đơn vị kế toán") == normalize_entity_name(
        "đơn vị kế toán"
    )


def test_concept_key_shape():
    assert concept_key("kiểm kê tài sản") == "concept::kiểm kê tài sản"
    assert concept_key("  ") is None


def test_extract_chunk_uses_injectable_fn():
    chunk = _chunk("doc-a", "Điều 40", "a) Cuối kỳ kế toán năm;", "Khoản 2")
    out = extract_chunk(chunk, extract_fn=fake_extract)

    assert out.events[0].modality == "phải"
    assert "concept::kiểm kê tài sản" in out.concept_keys()
    assert "concept::kiểm kê" in out.concept_keys()


def test_extract_chunks_writes_and_reuses_cache(tmp_path: Path):
    chunks = [
        _chunk("doc-a", "Điều 40", "a) Cuối kỳ kế toán năm;", "Khoản 2"),
        _chunk("doc-b", "Điều 1", "Không liên quan."),
    ]
    cache = tmp_path / "extract.json"
    calls = {"n": 0}

    def counting(chunk: LegalChunk) -> ChunkExtraction:
        calls["n"] += 1
        return fake_extract(chunk)

    first = extract_chunks(chunks, cache_path=cache, extract_fn=counting)
    assert cache.is_file()
    assert calls["n"] == 2

    second = extract_chunks(chunks, cache_path=cache, extract_fn=counting)
    assert calls["n"] == 2, "second call must not re-extract"
    assert first["doc-a::Điều40::Khoản2"].events[0].event_id == (
        second["doc-a::Điều40::Khoản2"].events[0].event_id
    )
    raw = json.loads(cache.read_text(encoding="utf-8"))
    assert "doc-a::Điều40::Khoản2" in raw


def test_build_index_joins_chunks_that_share_a_concept():
    a = _chunk("luat-ke-toan", "Điều 40", "a) Cuối kỳ kế toán năm;", "Khoản 2")
    b = _chunk(
        "law-2020-luat-doanh-nghiep",
        "Điều 128",
        "b) Có báo cáo liên quan kiểm kê tài sản.",
        "Khoản 3",
    )
    extractions = extract_chunks([a, b], extract_fn=fake_extract)
    concepts = concept_keys_by_chunk(extractions)
    index = build_index([a, b], concepts=concepts)

    shared = index.events_by_entity["concept::kiểm kê tài sản"]
    assert set(shared) == {a.chunk_id, b.chunk_id}

    out = expand([a], index, max_extra=5, use_semantic=False)
    assert b.chunk_id in {c.chunk_id for c in out}
