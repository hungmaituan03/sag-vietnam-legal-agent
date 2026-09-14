"""Public generation API."""

from sag_legal.generation.draft import DraftAnswer, InvalidDraftError, generate_draft

__all__ = ["DraftAnswer", "InvalidDraftError", "generate_draft"]
