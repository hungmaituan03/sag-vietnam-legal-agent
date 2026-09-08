"""Core legal document / chunk schemas (Week 1).

Why this exists:
- Citations must point to Điều / Khoản / Điểm, not opaque text blobs.
- Provenance + temporal fields are required before retrieval quality work.
- Keep models small and explicit so experiments stay reproducible.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class DocumentType(str, Enum):
    LUAT = "luat"
    NGHI_DINH = "nghi_dinh"
    THONG_TU = "thong_tu"
    QUYET_DINH = "quyet_dinh"
    GUIDANCE = "guidance"
    OTHER = "other"


class DocumentStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    AMENDED = "amended"
    REPLACED = "replaced"
    FUTURE = "future"
    UNKNOWN = "unknown"


class LegalDocument(BaseModel):
    """One legal instrument with provenance and temporal metadata."""

    document_id: str = Field(..., description="Stable id used across chunks and citations")
    title: str
    document_type: DocumentType = DocumentType.OTHER
    document_number: str | None = None
    issuing_authority: str | None = None
    issued_date: date | None = None
    effective_date: date | None = None
    expiration_date: date | None = None
    status: DocumentStatus = DocumentStatus.UNKNOWN
    amends: list[str] = Field(default_factory=list)
    replaces: list[str] = Field(default_factory=list)
    replaced_by: list[str] = Field(default_factory=list)
    domain: str | None = None
    source_url: HttpUrl | str | None = None
    content: str = Field(..., description="Full raw text used for parsing")


class LegalChunk(BaseModel):
    """Structure-aware chunk preserving citation path + parent context."""

    chunk_id: str
    document_id: str
    text: str
    chapter: str | None = None
    section: str | None = None
    article: str | None = None  # Điều
    clause: str | None = None  # Khoản
    point: str | None = None  # Điểm
    effective_date: date | None = None
    expiration_date: date | None = None
    status: DocumentStatus = DocumentStatus.UNKNOWN
    source_url: HttpUrl | str | None = None
    # Optional free-form links for later SAG entity work
    entity_refs: list[str] = Field(default_factory=list)

    @property
    def citation_path(self) -> str:
        parts = [p for p in (self.article, self.clause, self.point) if p]
        return ", ".join(parts) if parts else "(document-level)"


class IngestResult(BaseModel):
    document: LegalDocument
    chunks: list[LegalChunk]
    parser: Literal["structure_aware_v0"] = "structure_aware_v0"
