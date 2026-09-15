from sag_legal.org.resolve import lookup_org, resolve_orgs


def test_resolve_alias():
    result = resolve_orgs("Company A vốn điều lệ?")
    assert len(result) == 1
    assert result[0].org_id == "COA"


def test_resolve_alpha_finance():
    result = resolve_orgs("Alpha Finance cho vay trên 500 triệu?")
    assert len(result) == 1
    assert result[0].org_id == "COA"


def test_resolve_no_org():
    assert resolve_orgs("Kỳ kế toán năm?") == []


def test_lookup_org_known():
    ref = lookup_org("COA")
    assert ref is not None
    assert ref.org_id == "COA"


def test_lookup_org_unknown():
    assert lookup_org("NOPE") is None
