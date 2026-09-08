"""Structure-aware chunking for Vietnamese legal hierarchy (v0).

Week-1 strategy (simple on purpose):
1. Split the document on `Điều …` headings.
2. Inside each Điều, split on numbered khoản lines (`1.`, `2.`, …).
3. Inside each Khoản, split on điểm markers (`a)`, `b)`, …).
4. Every chunk keeps parent Điều/Khoản labels for citations.

Not a production layout parser — a measurable baseline we can improve.
"""

from __future__ import annotations

import re

from sag_legal.models import LegalChunk, LegalDocument

_ARTICLE_SPLIT = re.compile(r"(?=^Điều\s+\d+[A-Za-z]?)", re.MULTILINE)
_CLAUSE_SPLIT = re.compile(r"(?=^\d+\.\s+)", re.MULTILINE)
_POINT_SPLIT = re.compile(r"(?=^[a-zđ]\))", re.MULTILINE)

_ARTICLE_LABEL = re.compile(r"^(Điều\s+\d+[A-Za-z]?)")
_CLAUSE_LABEL = re.compile(r"^(\d+)\.")
_POINT_LABEL = re.compile(r"^([a-zđ])\)")


def chunk_document(document: LegalDocument) -> list[LegalChunk]:
    content = document.content.strip()
    if not content:
        return []

    articles = [p.strip() for p in _ARTICLE_SPLIT.split(content) if p.strip()]
    has_article = any(_ARTICLE_LABEL.match(a) for a in articles)

    if not has_article:
        return [_chunk(document, content, None, None, None)]

    chunks: list[LegalChunk] = []
    for article_text in articles:
        if not _ARTICLE_LABEL.match(article_text):
            # Front matter before the first Điều — keep as document-level context.
            chunks.append(_chunk(document, article_text, None, None, None))
            continue

        article = _ARTICLE_LABEL.match(article_text).group(1)  # type: ignore[union-attr]
        clause_parts = [p.strip() for p in _CLAUSE_SPLIT.split(article_text) if p.strip()]

        # Only the Điều header (no numbered khoản)
        if len(clause_parts) == 1:
            chunks.append(_chunk(document, article_text, article, None, None))
            continue

        for part in clause_parts:
            if not _CLAUSE_LABEL.match(part):
                # Điều title line before "1."
                chunks.append(_chunk(document, part, article, None, None))
                continue

            clause = f"Khoản {_CLAUSE_LABEL.match(part).group(1)}"  # type: ignore[union-attr]
            point_parts = [p.strip() for p in _POINT_SPLIT.split(part) if p.strip()]

            if len(point_parts) == 1:
                chunks.append(_chunk(document, part, article, clause, None))
                continue

            for point_part in point_parts:
                m = _POINT_LABEL.match(point_part)
                if not m:
                    chunks.append(_chunk(document, point_part, article, clause, None))
                    continue
                point = f"Điểm {m.group(1)}"
                chunks.append(_chunk(document, point_part, article, clause, point))

    return chunks


def _chunk(
    document: LegalDocument,
    text: str,
    article: str | None,
    clause: str | None,
    point: str | None,
) -> LegalChunk:
    parts = [document.document_id]
    if article:
        parts.append(article.replace(" ", ""))
    if clause:
        parts.append(clause.replace(" ", ""))
    if point:
        parts.append(point.replace(" ", ""))
    if len(parts) == 1:
        parts.append("body")

    return LegalChunk(
        chunk_id="::".join(parts),
        document_id=document.document_id,
        text=text.strip(),
        article=article,
        clause=clause,
        point=point,
        effective_date=document.effective_date,
        expiration_date=document.expiration_date,
        status=document.status,
        source_url=document.source_url,
    )
