"""Package health helpers used by CI smoke tests."""

from __future__ import annotations

from sag_legal import __version__


def healthcheck() -> dict[str, str]:
    return {
        "status": "ok",
        "package": "sag-legal",
        "version": __version__,
        "stage": "week01-scaffold",
    }
