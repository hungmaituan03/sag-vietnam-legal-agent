from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from sag_legal.models import LegalChunk


@dataclass
class ScoreChunk:
    chunk: LegalChunk
    score: float

def cosine(a,b)->float: 
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

def embed_texts(texts: list[str], model=None) -> list[list[float]]:
    if not texts: 
        return []
    if callable(model):
        return model(texts)
    else:
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        vectors = model.encode(texts, normalize_embeddings=True)
        return [list(v) for v in vectors]

def search_dense(query, chunks, k=5, model=None):
    if not chunks: 
        return []
    k=max(0,k)
    if k == 0:
        return []
    chunk_vecs = embed_texts([c.text for c in chunks], model = model)
    query_vecs = embed_texts([query], model=model)[0]
    paired = [ 
        ScoreChunk(chunk=c, score=cosine(query_vecs,v))
        for c, v in zip(chunks, chunk_vecs, strict=True)
    ]
    paired.sort(key=lambda x: x.score, reverse=True)
    return paired[:k]