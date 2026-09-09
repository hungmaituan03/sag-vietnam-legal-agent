from dataclasses import dataclass
from sag_legal.models import LegalChunk
import re
from rank_bm25 import BM25Okapi

@dataclass
class ScoreChunk:
    chunk: LegalChunk
    score: float

def search_bm25(query: str, chunks: list[LegalChunk], k: int = 5,) -> list[ScoreChunk]:
    if not chunks: return []
    k = max(0,k)
    if k == 0: return []
    tokenized_corpus = [tokenize(chunk.text) for chunk in chunks]
    tokenized_query = tokenize(query)
    if not tokenized_query: return []
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)
    paired = [ 
        ScoreChunk(chunk=chunk, score=float(score))
        for chunk, score in zip(chunks, scores)
    ]
    paired.sort(key=lambda x: x.score, reverse=True)
    return paired[:k]


def tokenize(text:str) -> list[str]:
    cleaned_text = re.sub(r"[^\s\w]", " ", text.lower(), flags=re.UNICODE)
    tokens = cleaned_text.split()
    return [t for t in tokens if t]

