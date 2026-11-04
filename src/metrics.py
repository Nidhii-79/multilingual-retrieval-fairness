"""Ranking metrics and bootstrap intervals."""
import numpy as np

def ranking_from_scores(scores):
    return np.argsort(-np.asarray(scores), axis=1)

def recall_at_k(rankings, gold, k):
    rankings, gold = np.asarray(rankings), np.asarray(gold)
    if len(rankings) != len(gold): raise ValueError("rankings and gold must have equal length")
    if len(gold) == 0: raise ValueError("Cannot evaluate an empty query set")
    return float(np.mean([g in row[:k] for row, g in zip(rankings, gold)]))

def mean_reciprocal_rank(rankings, gold):
    rankings, gold = np.asarray(rankings), np.asarray(gold)
    if len(rankings) != len(gold): raise ValueError("rankings and gold must have equal length")
    if len(gold) == 0: raise ValueError("Cannot evaluate an empty query set")
    rr=[]
    for row, target in zip(rankings, gold):
        hit=np.flatnonzero(row == target)
        rr.append(1.0/(int(hit[0])+1) if len(hit) else 0.0)
    return float(np.mean(rr))

def bootstrap_recall_at_k(rankings, gold, k=5, n_boot=1000, seed=42):
    rankings, gold = np.asarray(rankings), np.asarray(gold)
    rng=np.random.default_rng(seed); n=len(gold)
    if n == 0: raise ValueError("Cannot bootstrap an empty query set")
    vals=[]
    for _ in range(n_boot):
        ix=rng.integers(0,n,n)
        vals.append(recall_at_k(rankings[ix], gold[ix], k))
    return tuple(float(x) for x in np.percentile(vals,[2.5,97.5]))
