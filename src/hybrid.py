"""Reciprocal Rank Fusion for combining two retrieval rankings."""
import numpy as np

def rrf_fuse(rankings_a, rankings_b, k=60):
    a,b=np.asarray(rankings_a),np.asarray(rankings_b)
    if a.shape != b.shape: raise ValueError("Both ranking matrices must have the same shape")
    n_queries,n_docs=a.shape
    fused=np.zeros((n_queries,n_docs),dtype=np.float64)
    for i in range(n_queries):
        for rank,doc_id in enumerate(a[i],start=1): fused[i,doc_id]+=1.0/(k+rank)
        for rank,doc_id in enumerate(b[i],start=1): fused[i,doc_id]+=1.0/(k+rank)
    return np.argsort(-fused,axis=1)
