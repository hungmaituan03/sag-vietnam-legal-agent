# Commit conventions

Use short, imperative subjects. Prefer many small commits over one giant dump.

## Format

```text
<type>: <short why/what>

Optional body: context, trade-offs, follow-ups.
```

## Types we use

| Type | Use for |
|------|---------|
| `chore` | Scaffold, tooling, CI, repo hygiene |
| `docs` | README, architecture, process notes |
| `feat` | New capability (parser, BM25, Hindsight, …) |
| `fix` | Bug fix |
| `test` | Tests only |
| `refactor` | No behavior change |
| `experiment` | Benchmark / ablation wiring (link config + metrics in body) |

## Examples

```text
chore: scaffold package layout and CI

feat: add BM25 retriever over structured chunks

experiment: compare BM25 vs dense on eval-v0
Hypothesis: dense helps paraphrases; BM25 wins exact Điều numbers.
Config: configs/exp_w3_hybrid.yaml
```

## What not to do

- Vague messages: `update`, `wip`, `fix stuff`
- Committing secrets, large raw corpora, or generated vector DBs
- Rewriting public history on `main` (`push --force`) unless mentor asks
