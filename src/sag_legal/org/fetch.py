from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sag_legal.org.resolve import OrgRef


@dataclass
class OrgDocument:
    org_id: str
    doc_type: str
    text: str


def _default_org_root() -> Path:
    # org/fetch.py → parents[3] is repo root (src/sag_legal/org → repo)
    return Path(__file__).resolve().parents[3] / "data" / "org"


def fetch_org_docs(
    org: OrgRef,
    *,
    root: Path | None = None,
) -> list[OrgDocument]:
    base = root if root is not None else _default_org_root()
    org_dir = base / org.org_id
    if not org_dir.is_dir():
        return []
    docs = []
    for path in sorted(org_dir.glob("*.md")):
        docs.append(
            OrgDocument(
                org_id=org.org_id,
                doc_type=path.stem,
                text=path.read_text(encoding="utf-8"),
            )
        )
    return docs
