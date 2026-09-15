from sag_legal.models import LegalChunk
from sag_legal.org.fetch import OrgDocument
from sag_legal.org.merge import merge_evidence, org_docs_to_chunks


def test_org_docs_to_chunks_smoke():
    docs = [
        OrgDocument(org_id="COA", doc_type="dkkd", text="ĐKKD stub text."),
        OrgDocument(org_id="COA", doc_type="dieu_le", text="Điều lệ stub."),
    ]
    chunks = org_docs_to_chunks(docs)

    assert len(chunks) == 2
    assert chunks[0].document_id == "org:COA:dkkd"
    assert chunks[0].chunk_id == "org:COA:dkkd::body"
    assert chunks[0].text == "ĐKKD stub text."
    assert chunks[0].entity_refs == ["COA"]
    assert chunks[0].article is None

    assert chunks[1].document_id == "org:COA:dieu_le"
    assert chunks[1].chunk_id == "org:COA:dieu_le::body"


def _statute(doc: str, article: str, text: str) -> LegalChunk:
    return LegalChunk(
        chunk_id=f"{doc}::{article.replace(' ', '')}",
        document_id=doc,
        text=text,
        article=article,
    )


def test_org_docs_to_chunks_empty():
    assert org_docs_to_chunks([]) == []


def test_merge_appends_org_after_statute():
    statutes = [
        _statute("luat-tctd", "Điều 1", "statute A"),
        _statute("luat-tctd", "Điều 2", "statute B"),
    ]
    org_docs = [
        OrgDocument("COA", "dkkd", "org A"),
        OrgDocument("COA", "dieu_le", "org B"),
    ]
    merged = merge_evidence(statutes, org_docs)

    assert len(merged) == 4
    assert merged[0].chunk_id == statutes[0].chunk_id
    assert merged[1].chunk_id == statutes[1].chunk_id
    assert merged[2].document_id == "org:COA:dkkd"
    assert merged[3].document_id == "org:COA:dieu_le"


def test_merge_max_org_truncates_org_only():
    statutes = [_statute("luat-tctd", "Điều 1", "only statute")]
    org_docs = [
        OrgDocument("COA", "a", "1"),
        OrgDocument("COA", "b", "2"),
        OrgDocument("COA", "c", "3"),
    ]
    merged = merge_evidence(statutes, org_docs, max_org=1)

    assert len(merged) == 2
    assert merged[0].chunk_id == statutes[0].chunk_id
    assert merged[1].document_id == "org:COA:a"


def test_merge_empty_org_docs():
    statutes = [_statute("luat-tctd", "Điều 1", "x")]
    assert merge_evidence(statutes, []) == statutes
