"""Query-time draft answer from post-SAG evidence chunks.

Index-time Qwen extraction (sag.extract) builds concept joins.
This module is a different LLM job: turn the evidence pack into a
Vietnamese answer with citations. Hindsight stays out of scope for v0.

CI stays offline via injectable `client=` / `generate_fn=` fakes.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from sag_legal.models import LegalChunk

GenerateFn = Callable[[str, Sequence[LegalChunk]], "DraftAnswer"]


@dataclass
class DraftAnswer:
    answer: str
    cited_chunk_ids: list[str] = field(default_factory=list)
    abstained: bool = False


def _format_evidence(evidence: Sequence[LegalChunk]) -> str:
    blocks: list[str] = []
    for i, chunk in enumerate(evidence, start=1):
        eff = chunk.effective_date.isoformat() if chunk.effective_date else "unknown"
        blocks.append(
            "\n".join(
                [
                    f"[{i}] chunk_id={chunk.chunk_id}",
                    f"    document_id={chunk.document_id}",
                    f"    citation={chunk.citation_path}",
                    f"    effective_date={eff}",
                    f"    text={chunk.text}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _build_prompt(query: str, evidence: Sequence[LegalChunk]) -> str:
    return f"""Bạn là trợ lý pháp lý Việt Nam. Chỉ dùng bằng chứng bên dưới.

Yêu cầu:
- Trả lời ngắn gọn bằng tiếng Việt.
- Mọi kết luận phải bám đúng văn bản trong evidence.
- Trích dẫn bằng document_id + citation (Điều/Khoản/Điểm).
- Nếu evidence không đủ: nói rõ và đặt abstained=true, không bịa điều luật.

Trả về ĐÚNG một JSON object, không markdown, không giải thích:
{{
  "answer": "văn bản trả lời cho người dùng",
  "cited_chunk_ids": ["chunk_id đã dùng"],
  "abstained": false
}}

Câu hỏi:
{query}

Evidence:
{_format_evidence(evidence)}
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


def _parse_draft(
    content: str, allowed_ids: set[str]
) -> DraftAnswer:
    text = _strip_fences(content)
    data = json.loads(text)
    raw_ids = data.get("cited_chunk_ids") or []
    cited = [str(cid) for cid in raw_ids if str(cid) in allowed_ids]
    abstained = bool(data.get("abstained"))
    answer = str(data.get("answer") or "").strip()
    if not answer:
        abstained = True
        answer = "Không đủ căn cứ trong evidence để trả lời."
    return DraftAnswer(answer=answer, cited_chunk_ids=cited, abstained=abstained)


def generate_draft(
    query: str,
    evidence: Sequence[LegalChunk],
    *,
    client: Any | None = None,
    model: str | None = None,
    generate_fn: GenerateFn | None = None,
) -> DraftAnswer:
    """Draft a user-facing answer from SAG (or Voyage) evidence chunks."""
    if generate_fn is not None:
        return generate_fn(query, evidence)

    if not evidence:
        return DraftAnswer(
            answer="Không có điều khoản nào được truy xuất để trả lời câu hỏi.",
            cited_chunk_ids=[],
            abstained=True,
        )

    from sag_legal.sag.extract import get_qwen_client
    from sag_legal.settings import get_settings

    settings = get_settings()
    active = client if client is not None else get_qwen_client()
    active_model = model or settings.qwen_model
    response = active.chat.completions.create(
        model=active_model,
        messages=[
            {"role": "system", "content": "Output valid JSON only."},
            {"role": "user", "content": _build_prompt(query, evidence)},
        ],
        temperature=0,
    )
    content = response.choices[0].message.content or "{}"
    allowed = {chunk.chunk_id for chunk in evidence}
    return _parse_draft(content, allowed)
