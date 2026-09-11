# Evaluation

## Datasets
- `datasets/khung1_gold_v0.jsonl` — small hand gold for Khung 1 RAG vs SAG

## Scoring library
- `metrics.py` — DocRecall, ProvRecall, Precision@K, phrase recall, F1

## Notebook
- `../notebooks/eval_khung1_rag_vs_sag.ipynb` — introduction, criteria, scoring, live run

```bash
pip install -e ".[dev]"
cd <repo-root>
SKIP_VOYAGE=1 jupyter notebook notebooks/eval_khung1_rag_vs_sag.ipynb
```
