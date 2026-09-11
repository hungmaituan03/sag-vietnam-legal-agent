# Khung 1 — RAG vs SAG retrieval eval (v0)

Demo / Friday report. Numbers from a live run of
`notebooks/eval_khung1_rag_vs_sag.ipynb` against `data/raw/uts_vlc_processed.json`
(Khung 1 pack). Primary metric is **ProvRecall**; Precision@K is secondary.

**One-line verdict:** On this 5-query gold, SAG did **not** beat Voyage-only RAG
on provision coverage (Δ ProvRecall = 0) and **lowered** mean Precision@K because
the pack grew from K=5 → K=15 without recovering the remaining miss.

## 1. What we compared

```text
Query → Hybrid (BM25 + dense) → Voyage rerank  →  [RAG evidence]
                              ↘ SAG expand     →  [SAG evidence]
```

| Arm | Evidence | Typical K |
| --- | --- | --- |
| **RAG** | Voyage shortlist only (`use_sag=False`) | 5 |
| **SAG** | Same seeds + structural/semantic expand (`max_extra=10`) | 15 |

Corpus after ingest: **8644** chunks · SAG index **8469** events / **5508** entities.  
Gold: `evaluation/datasets/khung1_gold_v0.jsonl` (n=5).  
Scoring: `evaluation/metrics.py` (DocRecall, ProvRecall, bag-level Precision@K).

## 2. Leaderboard (means)

| Arm | DocRecall | ProvRecall | Precision@K | F1 (P↔Prov) |
| --- | ---: | ---: | ---: | ---: |
| RAG | 1.00 | **0.80** | **0.36** | 0.48 |
| SAG | 1.00 | **0.80** | 0.25 | 0.37 |

```text
Δ ProvRecall (SAG − RAG) = +0.000
```

How to read this in the demo:

- **Tied ProvRecall** → SAG did not find extra gold provisions the seeds already missed.
- **Precision↓** → expected when K grows; only a win if ProvRecall rises enough to matter.
- **F1↓** → mostly the precision drop, not a separate story.

## 3. Per-query (what to click through)

| Id | Criterion | ProvRecall RAG→SAG | Precision RAG→SAG | Note for demo |
| --- | --- | --- | --- | --- |
| `q01_dieu40_orphan` | orphan_repair | 1.0 → 1.0 | 0.20 → 0.20 | Seeds already hit Điều 40 / Khoản 2 |
| `q02_aml_tctd` | cross_law | 1.0 → 1.0 | 0.40 → **0.47** | Rare case: SAG pack a bit denser on gold |
| `q03_ky_ke_toan` | lexical_anchor | 1.0 → 1.0 | 0.60 → 0.40 | Control: SAG did not break recall, did dilute P |
| `q04_nhan_hieu` | definition_anchor | **0.0 → 0.0** | 0.00 → 0.00 | **Both miss** SHTT Điều 4 Khoản 16 |
| `q05_pcrt_nguyen_tac` | parent_repair | 1.0 → 1.0 | 0.60 → 0.20 | Recall ok; expansion adds many non-gold chunks |

**Only systematic miss:** `law-2005-luat-so-huu-tri-tue | Điều 4 | Khoản 16`  
(doc id is present → DocRecall=1; wrong article/clause in the pack → ProvRecall=0).  
SAG expansion did not repair that miss on this run.

## 4. Demo script (~5 min)

1. Open this file (verdict + leaderboard).
2. Open the notebook → summary cell (same table live).
3. Show **q04** miss list (both arms) — corpus/retrieval routing issue, not “SAG magic”.
4. Show **q03** or **q05** — same ProvRecall, lower Precision → padding.
5. Point at PR code: gold JSONL → `score_retrieval` → notebook arms.

Optional screenshot set:

- Leaderboard table (section 2)
- Per-query row for q04 + `missing_provisions`
- One RAG vs SAG evidence print for q05 (size difference)

## 5. Limits (say these out loud)

- Gold is **tiny** (5 hand queries) — smoke signal, not a benchmark.
- Precision is **bag-level @K**, not ranked AP; larger K hurts SAG mechanically.
- Answer / phrase scoring was not the focus of this run (`phrase_recall` unset).
- First Khung 1 embed rebuild is slow; use `data/processed/khung1_embeddings.npz` cache.

## 6. Next slice

1. Harden gold on **q04** (check chunking of Điều 4 Khoản 16; seed diversity).
2. Add 2–3 queries where Voyage seeds are *known orphans* so ProvRecall can move.
3. Report orphan-repair rate (like `docs/experiments/sag_evidence.md`) alongside ProvRecall.
4. Optional: cap SAG extras or re-rank expanded pack before scoring Precision@K.

## Repro

```bash
source .venv/bin/activate
pip install -e ".[dev]"
# VOYAGE_API_KEY in .env
jupyter notebook notebooks/eval_khung1_rag_vs_sag.ipynb
```

Offline unit tests only (no Voyage):

```bash
pytest -q evaluation/tests
```
