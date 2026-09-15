from sag_legal.org.fetch import fetch_org_docs
from sag_legal.org.resolve import OrgRef, resolve_orgs


def test_fetch_vcc_from_repo_fixtures():
    org = resolve_orgs("VietCredit")[0]
    docs = fetch_org_docs(org)
    assert {d.doc_type for d in docs} == {"dkkd", "dieu_le"}
    assert all(d.text.strip() for d in docs)

def test_fetch_skips_when_needs_clarify():
    org = OrgRef(
        org_id="VCC",
        display_name="x",
        aliases=[],
        org_type="financial_company",
        needs_clarify=True,
    )
    assert fetch_org_docs(org) == []

def test_fetch_missing_dir(tmp_path):
    org = OrgRef(
        org_id="NOPE",
        display_name="does not exist",
        aliases=[],
        org_type="financial_company",
        needs_clarify=False,
    )
    assert fetch_org_docs(org, root=tmp_path) == []