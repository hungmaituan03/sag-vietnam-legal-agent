"""Retrieval / answer scoring helpers for Khung 1 eval notebooks.

Designed for offline unit tests (no Voyage/Qwen). The notebook wires live
retrieval; this module only scores evidence packs and draft strings.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sag_legal.models import LegalChunk


@dataclass(frozen=True)
class ProvisionRef:
    document_id: str
    article: str
    clause: str | None = None
    point: str | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ProvisionRef:
        return cls(
            document_id=str(raw["document_id"]),
            article=str(raw["article"]),
            clause=(str(raw["clause"]) if raw.get("clause") else None),
            point=(str(raw["point"]) if raw.get("point") else None),
        )


@dataclass
class GoldItem:
    id: str
    query: str
    must_docs: list[str] = field(default_factory=list)
    must_provisions: list[ProvisionRef] = field(default_factory=list)
    must_phrases: list[str] = field(default_factory=list)
    criterion: str = ""
    notes: str = ""


@dataclass
class RetrievalScore:
    doc_recall: float
    provision_recall: float
    precision_at_k: float
    hit_docs: list[str]
    hit_provisions: list[str]
    missing_docs: list[str]
    missing_provisions: list[str]
    k: int


@dataclass
class AnswerScore:
    phrase_recall: float
    hit_phrases: list[str]
    missing_phrases: list[str]
    abstained: bool


def load_gold(path: Path | str) -> list[GoldItem]:
    items: list[GoldItem] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        raw = json.loads(line)
        items.append(
            GoldItem(
                id=str(raw["id"]),
                query=str(raw["query"]),
                must_docs=[str(x) for x in raw.get("must_docs") or []],
                must_provisions=[
                    ProvisionRef.from_dict(p) for p in (raw.get("must_provisions") or [])
                ],
                must_phrases=[str(x) for x in raw.get("must_phrases") or []],
                criterion=str(raw.get("criterion") or ""),
                notes=str(raw.get("notes") or ""),
            )
        )
    return items


def _chunk_matches_provision(chunk: LegalChunk, ref: ProvisionRef) -> bool:
    if chunk.document_id != ref.document_id:
        return False
    if chunk.article != ref.article:
        return False
    if ref.clause is not None and chunk.clause != ref.clause:
        return False
    if ref.point is not None and chunk.point != ref.point:
        return False
    return True


def provision_hit(evidence: Sequence[LegalChunk], ref: ProvisionRef) -> bool:
    return any(_chunk_matches_provision(c, ref) for c in evidence)


def format_provision(ref: ProvisionRef) -> str:
    parts = [ref.document_id, ref.article]
    if ref.clause:
        parts.append(ref.clause)
    if ref.point:
        parts.append(ref.point)
    return " | ".join(parts)


def score_retrieval(
    evidence: Sequence[LegalChunk],
    gold: GoldItem,
    *,
    k: int | None = None,
) -> RetrievalScore:
    """Score an evidence pack against gold docs + provisions.

    ``precision_at_k`` treats every gold-matching chunk as a true positive and
    all other retrieved chunks as false positives (bag-level, not ranked AP).
    """
    pack = list(evidence[:k] if k is not None else evidence)
    docs_present = {c.document_id for c in pack}
    hit_docs = [d for d in gold.must_docs if d in docs_present]
    missing_docs = [d for d in gold.must_docs if d not in docs_present]
    doc_recall = (
        len(hit_docs) / len(gold.must_docs) if gold.must_docs else 1.0
    )

    hit_provisions: list[str] = []
    missing_provisions: list[str] = []
    for ref in gold.must_provisions:
        label = format_provision(ref)
        if provision_hit(pack, ref):
            hit_provisions.append(label)
        else:
            missing_provisions.append(label)
    provision_recall = (
        len(hit_provisions) / len(gold.must_provisions)
        if gold.must_provisions
        else 1.0
    )

    if not pack:
        precision = 0.0
    else:
        tp = 0
        for chunk in pack:
            matched = False
            if chunk.document_id in gold.must_docs and not gold.must_provisions:
                matched = True
            for ref in gold.must_provisions:
                if _chunk_matches_provision(chunk, ref):
                    matched = True
                    break
                # Broader TP: same doc+article as a gold provision
                if (
                    chunk.document_id == ref.document_id
                    and chunk.article == ref.article
                ):
                    matched = True
                    break
            if matched:
                tp += 1
        precision = tp / len(pack)

    return RetrievalScore(
        doc_recall=doc_recall,
        provision_recall=provision_recall,
        precision_at_k=precision,
        hit_docs=hit_docs,
        hit_provisions=hit_provisions,
        missing_docs=missing_docs,
        missing_provisions=missing_provisions,
        k=len(pack),
    )


def score_answer(
    answer: str,
    gold: GoldItem,
    *,
    abstained: bool = False,
) -> AnswerScore:
    text = (answer or "").casefold()
    hits = [p for p in gold.must_phrases if p.casefold() in text]
    missing = [p for p in gold.must_phrases if p.casefold() not in text]
    phrase_recall = (
        len(hits) / len(gold.must_phrases) if gold.must_phrases else 1.0
    )
    return AnswerScore(
        phrase_recall=phrase_recall,
        hit_phrases=hits,
        missing_phrases=missing,
        abstained=abstained,
    )


def mean(values: Iterable[float]) -> float:
    xs = list(values)
    return sum(xs) / len(xs) if xs else 0.0


def f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
