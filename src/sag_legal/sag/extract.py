"""LLM event/entity extraction for SAG (paper-shaped stage).

Index time: each chunk is one SAG event; the LLM may annotate it with at most
one structured ExtractedEvent plus many concept entities.
Query time: the user question -> the same schema, then `concept::…` keys
look up matching chunks in `events_by_entity` before expand.

CI stays offline via an injectable `client=` / `extract_fn=` fake. Live calls
use OpenAI (or any OpenAI-compatible endpoint).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sag_legal.models import LegalChunk

DEFAULT_MODEL = "gpt-5.4-mini"
QUERY_CHUNK_ID = "query"

ExtractFn = Callable[[LegalChunk], "ChunkExtraction"]
QueryExtractFn = Callable[[str], "ChunkExtraction"]

_WS = re.compile(r"\s+")


@dataclass
class ExtractedEvent:
    event_id: str
    summary: str
    actor: str | None = None
    action: str | None = None
    modality: str | None = None
    trigger: str | None = None
    source_span: str | None = None


@dataclass
class ExtractedEntity:
    name: str
    type: str = "other"
    aliases: list[str] = field(default_factory=list)


@dataclass
class ChunkExtraction:
    """One chunk (or query) annotation: ≤1 event, many entities."""

    chunk_id: str
    events: list[ExtractedEvent] = field(default_factory=list)
    entities: list[ExtractedEntity] = field(default_factory=list)

    def concept_keys(self) -> list[str]:
        keys: list[str] = []
        seen: set[str] = set()
        for entity in self.entities:
            key = concept_key(entity.name)
            if key and key not in seen:
                seen.add(key)
                keys.append(key)
            for alias in entity.aliases:
                key = concept_key(alias)
                if key and key not in seen:
                    seen.add(key)
                    keys.append(key)
        return keys


def normalize_entity_name(name: str) -> str:
    """Collapse surface form so 'Đơn vị kế toán' and 'đơn vị kế toán' join."""
    return _WS.sub(" ", name.strip().casefold())


def concept_key(name: str) -> str | None:
    normalized = normalize_entity_name(name)
    if not normalized:
        return None
    return f"concept::{normalized}"


_EXTRACTION_SCHEMA = """{
  "events": [
    {
      "event_id": "short_snake_case",
      "summary": "một câu mô tả nghĩa vụ/sự kiện chính",
      "actor": "chủ thể hoặc null",
      "action": "hành động hoặc null",
      "modality": "phải|được|cấm|null",
      "trigger": "điều kiện kích hoạt hoặc null",
      "source_span": "đoạn văn gốc liên quan"
    }
  ],
  "entities": [
    {
      "name": "tên thực thể chuẩn hóa",
      "type": "organization|concept|document|obligation|time|other",
      "aliases": ["biến thể nếu có"]
    }
  ]
}

Quy tắc: "events" có đúng 0 hoặc 1 phần tử (một sự kiện chính cho cả đoạn/câu hỏi).
"entities" có thể có nhiều phần tử gắn với sự kiện đó."""


def _build_prompt(chunk: LegalChunk) -> str:
    return f"""Bạn là bộ trích xuất sự kiện/thực thể cho hệ thống SAG
trên văn bản pháp luật Việt Nam.

Mỗi đoạn văn bản là ĐÚNG MỘT sự kiện SAG. Chỉ trích tối đa một event
(nghĩa vụ/sự kiện chính); các khái niệm liên quan để vào entities.

Trả về ĐÚNG một JSON object, không markdown, không giải thích, với schema:
{_EXTRACTION_SCHEMA}

document_id: {chunk.document_id}
citation: {chunk.citation_path}
text:
{chunk.text}
"""


def _build_query_prompt(query: str) -> str:
    return f"""Bạn là bộ trích xuất sự kiện/thực thể cho hệ thống SAG
trên câu hỏi pháp luật Việt Nam.

Câu hỏi là ĐÚNG MỘT sự kiện truy vấn. Chỉ trích tối đa một event;
các khái niệm/nghĩa vụ cần nối với điều khoản để vào entities.

Trả về ĐÚNG một JSON object, không markdown, không giải thích, với schema:
{_EXTRACTION_SCHEMA}

query:
{query}
"""


def _parse_extraction(chunk_id: str, content: str) -> ChunkExtraction:
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    data = json.loads(text)
    events = [
        ExtractedEvent(
            event_id=str(item.get("event_id") or f"event_{i}"),
            summary=str(item.get("summary") or ""),
            actor=item.get("actor"),
            action=item.get("action"),
            modality=item.get("modality"),
            trigger=item.get("trigger"),
            source_span=item.get("source_span"),
        )
        for i, item in enumerate(data.get("events") or [])
        if isinstance(item, dict)
    ][:1]
    entities = [
        ExtractedEntity(
            name=str(item.get("name") or ""),
            type=str(item.get("type") or "other"),
            aliases=[str(a) for a in (item.get("aliases") or []) if a],
        )
        for item in (data.get("entities") or [])
        if isinstance(item, dict) and item.get("name")
    ]
    return ChunkExtraction(chunk_id=chunk_id, events=events, entities=entities)


def get_openai_client() -> Any:
    from openai import OpenAI

    from sag_legal.settings import get_settings

    settings = get_settings()
    if not settings.openai_configured:
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in .env at the repo root.")
    return OpenAI(api_key=settings.openai_api_key)


def extract_chunk(
    chunk: LegalChunk,
    client: Any | None = None,
    model: str | None = None,
    extract_fn: ExtractFn | None = None,
) -> ChunkExtraction:
    """Extract events/entities for one chunk. Prefer extract_fn= in tests."""
    if extract_fn is not None:
        return extract_fn(chunk)

    from sag_legal.settings import get_settings

    settings = get_settings()
    active = client if client is not None else get_openai_client()
    active_model = model or settings.openai_model
    response = active.chat.completions.create(
        model=active_model,
        messages=[
            {"role": "system", "content": "Output valid JSON only."},
            {"role": "user", "content": _build_prompt(chunk)},
        ],
        temperature=0,
    )
    content = response.choices[0].message.content or "{}"
    return _parse_extraction(chunk.chunk_id, content)


def extract_query(
    query: str,
    client: Any | None = None,
    model: str | None = None,
    extract_fn: QueryExtractFn | None = None,
) -> ChunkExtraction:
    """Extract events/entities from a user question. Prefer extract_fn= in tests."""
    cleaned = query.strip()
    if extract_fn is not None:
        return extract_fn(cleaned)

    from sag_legal.settings import get_settings

    settings = get_settings()
    active = client if client is not None else get_openai_client()
    active_model = model or settings.openai_model
    response = active.chat.completions.create(
        model=active_model,
        messages=[
            {"role": "system", "content": "Output valid JSON only."},
            {"role": "user", "content": _build_query_prompt(cleaned)},
        ],
        temperature=0,
    )
    content = response.choices[0].message.content or "{}"
    return _parse_extraction(QUERY_CHUNK_ID, content)


def _load_cache(path: Path) -> dict[str, ChunkExtraction]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, ChunkExtraction] = {}
    for chunk_id, payload in raw.items():
        events = [ExtractedEvent(**event) for event in payload.get("events", [])][:1]
        out[chunk_id] = ChunkExtraction(
            chunk_id=chunk_id,
            events=events,
            entities=[
                ExtractedEntity(**entity) for entity in payload.get("entities", [])
            ],
        )
    return out


def load_extractions(cache_path: Path | str) -> dict[str, ChunkExtraction]:
    """Read a JSON extract cache without calling the LLM."""
    return _load_cache(Path(cache_path))


def _save_cache(path: Path, data: Mapping[str, ChunkExtraction]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        chunk_id: {
            "events": [asdict(event) for event in extraction.events],
            "entities": [asdict(entity) for entity in extraction.entities],
        }
        for chunk_id, extraction in data.items()
    }
    path.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def extract_chunks(
    chunks: Sequence[LegalChunk],
    cache_path: Path | str | None = None,
    client: Any | None = None,
    model: str | None = None,
    extract_fn: ExtractFn | None = None,
) -> dict[str, ChunkExtraction]:
    """Extract for many chunks; reuse a JSON cache keyed by chunk_id when given."""
    path = Path(cache_path) if cache_path is not None else None
    cached = _load_cache(path) if path is not None else {}
    results: dict[str, ChunkExtraction] = {}
    missing: list[LegalChunk] = []

    for chunk in chunks:
        if chunk.chunk_id in cached:
            results[chunk.chunk_id] = cached[chunk.chunk_id]
        elif chunk.chunk_id not in results:
            missing.append(chunk)

    for chunk in missing:
        results[chunk.chunk_id] = extract_chunk(
            chunk, client=client, model=model, extract_fn=extract_fn
        )

    if path is not None:
        # Merge so a shortlist extract never wipes a larger prior cache.
        cached.update(results)
        _save_cache(path, cached)
    return results


def concept_keys_by_chunk(
    extractions: Mapping[str, ChunkExtraction],
) -> dict[str, list[str]]:
    return {chunk_id: extraction.concept_keys() for chunk_id, extraction in extractions.items()}
