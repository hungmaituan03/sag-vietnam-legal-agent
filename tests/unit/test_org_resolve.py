from sag_legal.org.resolve import resolve_orgs


def test_resolve_alias():
    result = resolve_orgs("VietCredit vốn điều lệ?")
    assert len(result) == 1
    assert result[0].org_id=="VCC"
    assert result[0].needs_clarify is False 

def test_resolve_no_org():
    assert resolve_orgs("Kỳ kế toán năm?") == []

def test_resolve_ambiguous_two_orgs():
    result = resolve_orgs("VietCredit và Waka")
    assert len(result) == 2
    assert {r.org_id for r in result} == {"VCC", "WAKA"}
    assert all(r.needs_clarify for r in result)
