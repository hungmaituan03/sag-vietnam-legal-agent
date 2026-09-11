"""LLM event/entity extraction for SAG (paper-shaped stage).

Index time: each chunk -> typed events + concept entities.
Those become `concept::…` keys in `events_by_entity`, so `expand` can join
chunks that share a meaning even when they do not share a Điều.

CI stays offline via an injectable `client=` / `extract_fn=` fake. Live calls
use Qwen through the OpenAI-compatible DashScope endpoint.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sag_legal.models import LegalChunk

DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.7-flash"

ExtractFn = Callable[[LegalChunk], "ChunkExtraction"]

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


def _build_prompt(chunk: LegalChunk) -> str:
    return f"""Bạn là bộ trích xuất sự kiện/thực thể cho hệ thống SAG
trên văn bản pháp luật Việt Nam.

Trả về ĐÚNG một JSON object, không markdown, không giải thích, với schema:
{{
  "events": [
    {{
      "event_id": "short_snake_case",
      "summary": "một câu mô tả nghĩa vụ/sự kiện",
      "actor": "chủ thể hoặc null",
      "action": "hành động hoặc null",
      "modality": "phải|được|cấm|null",
      "trigger": "điều kiện kích hoạt hoặc null",
      "source_span": "đoạn văn gốc liên quan"
    }}
  ],
  "entities": [
    {{
      "name": "tên thực thể chuẩn hóa",
      "type": "organization|concept|document|obligation|time|other",
      "aliases": ["biến thể nếu có"]
    }}
  ]
}}

document_id: {chunk.document_id}
citation: {chunk.citation_path}
text:
{chunk.text}
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
    ]
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


def get_qwen_client() -> Any:
    from openai import OpenAI

    from sag_legal.settings import get_settings

    settings = get_settings()
    if not settings.qwen_configured:
        raise RuntimeError("QWEN_API_KEY is not set. Put it in .env at the repo root.")
    return OpenAI(api_key=settings.qwen_api_key, base_url=settings.qwen_base_url)


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
    active = client if client is not None else get_qwen_client()
    active_model = model or settings.qwen_model
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


def _load_cache(path: Path) -> dict[str, ChunkExtraction]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, ChunkExtraction] = {}
    for chunk_id, payload in raw.items():
        out[chunk_id] = ChunkExtraction(
            chunk_id=chunk_id,
            events=[ExtractedEvent(**event) for event in payload.get("events", [])],
            entities=[
                ExtractedEntity(**entity) for entity in payload.get("entities", [])
            ],
        )
    return out


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
