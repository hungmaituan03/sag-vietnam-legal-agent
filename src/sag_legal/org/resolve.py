from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OrgRef:
    org_id: str
    display_name: str
    aliases: list[str]
    org_type: str
    mst: str | None = None 
    needs_clarify: bool = False 
    clarify_prompt: str = ""

DEFAULT_CATALOG: dict[str, dict] = {
    "VCC": {
        "canonical_name": "Công ty Tài chính Cổ phần Tín Việt",
        "aliases": ["VietCredit", "Tín Việt", "VCC"],
        "entity_type": "financial_company",
        "tax_id": None,
        "status": "active",
    },

    "WAKA": {
        "canonical_name": "Công ty Cổ phần Sách điện tử Waka",
        "aliases": ["Waka", "Sách điện tử Waka", "WAKA"],
        "entity_type": "ebook_company",
        "tax_id": "0108796796",
        "status": "active",
    },
}

def _normalize(text):
    return text.lower().strip()

def _entry_to_ref(org_id, entry, *, needs_clarify, clarify_prompt="") -> OrgRef:
    return OrgRef(
        org_id=org_id,
        display_name=entry["canonical_name"],
        aliases=list(entry.get("aliases") or []),
        org_type=entry["entity_type"],
        mst=entry.get("tax_id"),
        needs_clarify=needs_clarify,
        clarify_prompt=clarify_prompt,
    )

def resolve_orgs(query: str, catalog: dict | None = None) -> list[OrgRef]:
    catalog = DEFAULT_CATALOG if catalog is None else catalog
    q = _normalize(query)
    hits: list[str] = []

    for org_id, entry in catalog.items():
        if entry.get("status", "active") != "active":
            continue
        names = [entry["canonical_name"], *entry.get("aliases", [])]
        for name in names:
            n = _normalize(name)
            if len(n) >= 2 and n in q: 
                hits.append(org_id)
                break
        
    if not hits: 
        return []
    if len(hits) == 1:
        oid = hits[0]
        return [_entry_to_ref(oid, catalog[oid], needs_clarify=False)]

    names = [catalog[oid]["canonical_name"] for oid in hits]
    prompt = "Bạn đang hỏi về tổ chức nào: " + "; ".join(names) + "?"
    return [
        _entry_to_ref(oid, catalog[oid], needs_clarify=True, clarify_prompt=prompt)
        for oid in hits
    ]