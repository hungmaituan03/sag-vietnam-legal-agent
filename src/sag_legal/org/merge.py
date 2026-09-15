from sag_legal.models import LegalChunk
from sag_legal.org.fetch import OrgDocument


def org_docs_to_chunks(docs: list[OrgDocument]) -> list[LegalChunk]:
    results = []
    for doc in docs:
        document_id = f"org:{doc.org_id}:{doc.doc_type}"
        chunk = LegalChunk(
            chunk_id=f"{document_id}::body",
            document_id=document_id,
            text=doc.text,
            entity_refs=[doc.org_id],
        )
        results.append(chunk)
    return results

def merge_evidence(
    statute_chunks: list[LegalChunk],
    org_docs: list[OrgDocument],
    *,
    max_org: int = 4,
) -> list[LegalChunk]:
    org_chunks = org_docs_to_chunks(org_docs)
    org_chunks = org_chunks[:max_org]
    return statute_chunks + org_chunks