# Data layout

## `raw/`
Authoritative source files (PDF/HTML/text) plus provenance. Prefer official government/legal URLs.

Do **not** commit proprietary company documents or unverified bulk dumps from third-party repos without mentor approval. Keep large binary corpora out of git; document how to obtain them instead.

## `processed/`
Structured documents and structure-aware chunks (Điều / Khoản / Điểm metadata, effective dates, status).

## Provenance fields to preserve
- `source_url`
- `document_number`, `document_type`, `issuing_authority`
- `issued_date`, `effective_date`, `expiration_date`, `status`
- `amends` / `replaces` / `replaced_by` when known
