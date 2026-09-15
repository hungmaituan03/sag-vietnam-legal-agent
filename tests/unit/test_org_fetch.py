from sag_legal.org.fetch import fetch_org_docs
from sag_legal.org.resolve import OrgRef, resolve_orgs


def test_fetch_coa_from_repo_fixtures():
    org = resolve_orgs("Company A")[0]
    docs = fetch_org_docs(org)
    assert {d.doc_type for d in docs} == {"dkkd", "dieu_le", "quy_trinh"}
    assert all(d.text.strip() for d in docs)
    assert all(d.org_id == "COA" for d in docs)


def test_fetch_missing_dir(tmp_path):
    org = OrgRef(
        org_id="NOPE",
        display_name="does not exist",
        aliases=[],
        org_type="financial_company",
    )
    assert fetch_org_docs(org, root=tmp_path) == []
