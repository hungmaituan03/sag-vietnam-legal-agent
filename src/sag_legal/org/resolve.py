from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OrgRef:
    org_id: str
    display_name: str
    aliases: list[str]
    org_type: str
    mst: str | None = None


# Study fixture: one fake TCTD only (Company A / Alpha Finance).
DEFAULT_CATALOG: dict[str, dict] = {
    "COA": {
        "canonical_name": "Công ty Tài chính TNHH Một thành viên Alpha",
        "aliases": [
            "Company A",
            "Công ty A",
            "Alpha Finance",
            "Alpha",
            "COA",
        ],
        "entity_type": "financial_company",
        "tax_id": "0199999999",
        "status": "active",
    },
}


def _normalize(text: str) -> str:
    return text.lower().strip()


def _entry_to_ref(org_id: str, entry: dict) -> OrgRef:
    return OrgRef(
        org_id=org_id,
        display_name=entry["canonical_name"],
        aliases=list(entry.get("aliases") or []),
        org_type=entry["entity_type"],
        mst=entry.get("tax_id"),
    )


def lookup_org(org_id: str, catalog: dict | None = None) -> OrgRef | None:
    """Return OrgRef for a catalog id, or None if unknown/inactive."""
    catalog = DEFAULT_CATALOG if catalog is None else catalog
    entry = catalog.get(org_id)
    if entry is None or entry.get("status", "active") != "active":
        return None
    return _entry_to_ref(org_id, entry)


def resolve_orgs(query: str, catalog: dict | None = None) -> list[OrgRef]:
    """Match org aliases in the query. At most one clear hit (COA-only catalog)."""
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
    # Single-org study scope: take the first match; no clarify path.
    oid = hits[0]
    return [_entry_to_ref(oid, catalog[oid])]
