# GitHub workflow — show process over time

This internship is graded on **visible process**, not only a final demo. GitHub is our timeline.

## Cadence

| When | What |
|------|------|
| Daily | Small commits on a feature branch; push before end of day when possible |
| Mid-week | Open or update a PR (even as draft) so mentors can comment early |
| Friday 16:00 | Close the weekly milestone issue with links to PR, report, demo notes |

## Branch model (simple)

```text
main              ← always green; protected once remote exists
└── week01/...    ← example: week01/scaffold, week01/bm25-baseline
└── week02/...
```

Rules:
1. Do **not** push broken `main`. Prefer PRs.
2. One focused concern per PR when possible (ingestion ≠ retrieval).
3. PR description must say **what / why / how to test / known limits**.

## What “process” looks like in the history

Mentors should be able to scroll commits and see:

1. Scaffold → ingest → chunk → BM25 → dense → hybrid → Voyage → SAG → Hindsight → Q&A …
2. Experiments with config filenames and metric notes linked from PRs
3. Failures documented (not deleted) — “tried X, failed because Y”
4. Reuse attribution (SAG / Paparusi) called out in PR body

## Weekly milestone issue

Every Monday (or first day of the week), open an issue from the **Weekly milestone** template. Check boxes as you go. On Friday, comment:

- Link to merged/open PR(s)
- Link to report / demo notes (repo path or `SAG-docs`)
- 3 bullet “what I learned / what broke / next week”

## CI

GitHub Actions runs on every push/PR:

- install package
- `ruff` lint
- `pytest`

A red CI is feedback, not shame — fix it in the same PR.

## Secrets

Never commit `.env` or API keys. Use GitHub **Actions secrets** only when CI truly needs them (most early weeks do not).
