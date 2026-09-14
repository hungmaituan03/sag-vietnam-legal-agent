"""Query-time draft answer from post-SAG evidence chunks.

Index-time OpenAI extraction (sag.extract) builds concept joins.
This module is a different LLM job: turn the evidence pack into a
Vietnamese answer with citations. Hindsight stays out of scope for v0.

Invalid model output raises ``InvalidDraftError`` so callers can stop
(no soft fake answer, no follow-up LLM spend on garbage).

CI stays offline via injectable `client=` / `generate_fn=` fakes.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from sag_legal.models import LegalChunk

GenerateFn = Callable[[str, Sequence[LegalChunk]], "DraftAnswer"]

# Keep prompts under control: SAG can expand to 15+ chunks and the model then
# truncates JSON or writes overly short answers.
DEFAULT_MAX_EVIDENCE = 10
DEFAULT_MAX_CHUNK_CHARS = 600


class InvalidDraftError(ValueError):
    """Raised when the draft LLM returns unusable JSON / citations."""


@dataclass
class DraftAnswer:
    answer: str
    cited_chunk_ids: list[str] = field(default_factory=list)
    abstained: bool = False


def select_evidence_for_draft(
    evidence: Sequence[LegalChunk],
    *,
    max_chunks: int = DEFAULT_MAX_EVIDENCE,
    seed_ids: set[str] | None = None,
) -> list[LegalChunk]:
    """Pick chunks for the answer LLM.

    Naive ``evidence[:max_chunks]`` drops SAG's most useful adds: same-Điều
    parents often land *after* headings of other seeds (round-robin expand).
    Prefer: seeds → repairs for điểm-orphan articles → other same-article
    extras → everything else.
    """
    if max_chunks <= 0:
        return []
    items = list(evidence)
    if len(items) <= max_chunks:
        return items

    seeds = {sid for sid in (seed_ids or set()) if sid}
    if not seeds:
        return items[:max_chunks]

    seed_chunks = [c for c in items if c.chunk_id in seeds]
    extras = [c for c in items if c.chunk_id not in seeds]
    seed_articles = {
        (c.document_id, c.article) for c in seed_chunks if c.article
    }
    orphan_articles = {
        (c.document_id, c.article) for c in seed_chunks if c.point and c.article
    }

    def article_key(chunk: LegalChunk) -> tuple[str, str] | None:
        if not chunk.article:
            return None
        return (chunk.document_id, chunk.article)

    repair_orphan = [c for c in extras if article_key(c) in orphan_articles]
    repair_other = [
        c
        for c in extras
        if article_key(c) in seed_articles
        and article_key(c) not in orphan_articles
    ]
    other = [c for c in extras if article_key(c) not in seed_articles]

    def orphan_priority(chunk: LegalChunk) -> tuple[int, int]:
        key = article_key(chunk)
        clause_match = any(
            s.point
            and s.clause
            and article_key(s) == key
            and chunk.clause == s.clause
            for s in seed_chunks
        )
        if clause_match:
            return (0, 0)
        is_heading = chunk.clause is None and chunk.point is None
        has_clause_repair = any(
            article_key(e) == key and e.clause for e in extras
        )
        if is_heading and has_clause_repair:
            return (1, 0)
        if is_heading:
            return (2, 0)
        return (3, 0)

    repair_orphan = [
        c
        for _, c in sorted(
            enumerate(repair_orphan),
            key=lambda iv: (*orphan_priority(iv[1]), iv[0]),
        )
    ]
    ordered = seed_chunks + repair_orphan + repair_other + other
    seen: set[str] = set()
    out: list[LegalChunk] = []
    for chunk in ordered:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        out.append(chunk)
        if len(out) >= max_chunks:
            break
    return out


def _format_evidence(
    evidence: Sequence[LegalChunk],
    *,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> str:
    blocks: list[str] = []
    for i, chunk in enumerate(evidence, start=1):
        eff = chunk.effective_date.isoformat() if chunk.effective_date else "unknown"
        text = chunk.text.strip()
        if max_chunk_chars > 0 and len(text) > max_chunk_chars:
            text = text[: max_chunk_chars - 1] + "…"
        blocks.append(
            "\n".join(
                [
                    f"[{i}] chunk_id={chunk.chunk_id}",
                    f"    document_id={chunk.document_id}",
                    f"    citation={chunk.citation_path}",
                    f"    effective_date={eff}",
                    f"    text={text}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _build_prompt(
    query: str,
    evidence: Sequence[LegalChunk],
    *,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> str:
    return f"""Bạn là trợ lý pháp lý Việt Nam. Chỉ dùng bằng chứng bên dưới.

Yêu cầu trả lời:
- Viết bằng tiếng Việt, đủ ý để người dùng hiểu quy định (không chỉ một câu quá ngắn).
- Có: (1) kết luận ngắn, (2) nội dung/điều kiện chính, (3) trích dẫn document_id + citation.
- Bám sát văn bản trong evidence; không bịa điều luật.
- Nếu evidence không đủ: nói rõ và đặt abstained=true.

Trả về ĐÚNG một JSON object, không markdown, không giải thích:
{{
  "answer": "văn bản trả lời đầy đủ cho người dùng",
  "cited_chunk_ids": ["chunk_id đã dùng — copy đúng từ evidence"],
  "abstained": false
}}

Câu hỏi:
{query}

Evidence:
{_format_evidence(evidence, max_chunk_chars=max_chunk_chars)}
"""


def _strip_fences(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return text


def _parse_draft(content: str, allowed_ids: set[str]) -> DraftAnswer:
    """Parse and validate draft JSON. Raises ``InvalidDraftError`` if unusable."""
    text = _strip_fences(content)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InvalidDraftError(
            "Draft model returned invalid JSON; refusing to spend more tokens "
            f"on a bad answer ({exc})."
        ) from exc

    if not isinstance(data, dict):
        raise InvalidDraftError("Draft JSON must be an object.")

    if "answer" not in data or "abstained" not in data:
        raise InvalidDraftError(
            "Draft JSON missing required keys: answer, abstained."
        )

    answer = str(data.get("answer") or "").strip()
    abstained = bool(data.get("abstained"))
    raw_ids = data.get("cited_chunk_ids")
    if raw_ids is None:
        raw_ids = []
    if not isinstance(raw_ids, list):
        raise InvalidDraftError("cited_chunk_ids must be a list.")

    cited = [str(cid) for cid in raw_ids if str(cid) in allowed_ids]
    unknown = [str(cid) for cid in raw_ids if str(cid) not in allowed_ids]

    if not answer:
        raise InvalidDraftError("Draft answer is empty.")

    if not abstained and not cited:
        detail = (
            f" unknown ids dropped={unknown!r}" if unknown else " no citations given"
        )
        raise InvalidDraftError(
            "Non-abstained draft must cite at least one evidence chunk_id;"
            f"{detail}."
        )

    return DraftAnswer(answer=answer, cited_chunk_ids=cited, abstained=abstained)


def generate_draft(
    query: str,
    evidence: Sequence[LegalChunk],
    *,
    client: Any | None = None,
    model: str | None = None,
    generate_fn: GenerateFn | None = None,
    max_evidence_chunks: int = DEFAULT_MAX_EVIDENCE,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    seed_ids: set[str] | None = None,
) -> DraftAnswer:
    """Draft a user-facing answer from SAG (or Voyage) evidence chunks.

    Raises ``InvalidDraftError`` when the model output fails validation
    (bad JSON, empty answer, or answer without evidence citations).
    """
    if generate_fn is not None:
        return generate_fn(query, evidence)

    if not evidence:
        return DraftAnswer(
            answer="Không có điều khoản nào được truy xuất để trả lời câu hỏi.",
            cited_chunk_ids=[],
            abstained=True,
        )

    selected = select_evidence_for_draft(
        evidence, max_chunks=max_evidence_chunks, seed_ids=seed_ids
    )
    from sag_legal.sag.extract import get_openai_client
    from sag_legal.settings import get_settings

    settings = get_settings()
    active = client if client is not None else get_openai_client()
    active_model = model or settings.openai_model
    # gpt-5+ rejects max_tokens; older chat models still want it.
    token_kw = (
        {"max_completion_tokens": 1200}
        if active_model.lower().startswith(("gpt-5", "o1", "o3", "o4"))
        else {"max_tokens": 1200}
    )
    response = active.chat.completions.create(
        model=active_model,
        messages=[
            {"role": "system", "content": "Output valid JSON only."},
            {
                "role": "user",
                "content": _build_prompt(
                    query, selected, max_chunk_chars=max_chunk_chars
                ),
            },
        ],
        temperature=0,
        **token_kw,
    )
    content = response.choices[0].message.content or "{}"
    allowed = {chunk.chunk_id for chunk in selected}
    return _parse_draft(content, allowed)
