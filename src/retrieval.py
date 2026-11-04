"""Sparse and multilingual dense retrieval helpers."""
import re
import numpy as np
from rank_bm25 import BM25Okapi

def tokenize(text):
    return re.findall(r"\w+", str(text).lower(), flags=re.UNICODE)

def bm25_rankings(queries, documents):
    model=BM25Okapi([tokenize(x) for x in documents])
    scores=np.asarray([model.get_scores(tokenize(q)) for q in queries],dtype=np.float32)
    return np.argsort(-scores,axis=1)

def load_embedding_model(model_id="intfloat/multilingual-e5-small"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_id)

def dense_rankings(model, queries, documents, e5_prefix=True, batch_size=32):
    q=[("query: "+str(x)) if e5_prefix else str(x) for x in queries]
    d=[("passage: "+str(x)) if e5_prefix else str(x) for x in documents]
    qv=model.encode(q,batch_size=batch_size,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=True)
    dv=model.encode(d,batch_size=batch_size,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=True)
    return np.argsort(-(qv @ dv.T),axis=1)
