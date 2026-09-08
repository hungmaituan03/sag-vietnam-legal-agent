from sag_legal import __version__
from sag_legal.health import healthcheck


def test_version_present():
    assert __version__


def test_healthcheck():
    payload = healthcheck()
    assert payload["status"] == "ok"
    assert payload["stage"] == "week01-scaffold"
