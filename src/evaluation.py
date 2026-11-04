"""Evaluate rankings and assemble comparable result rows."""
import time
from .metrics import recall_at_k, mean_reciprocal_rank

def evaluate_rankings(rankings, gold, system, query_lang, doc_lang, condition, elapsed_seconds=0.0):
    n=len(gold)
    return {"system":system,"query_lang":query_lang,"doc_lang":doc_lang,"condition":condition,
            "n_queries":n,"recall@1":recall_at_k(rankings,gold,1),
            "recall@5":recall_at_k(rankings,gold,5),"recall@10":recall_at_k(rankings,gold,10),
            "mrr":mean_reciprocal_rank(rankings,gold),
            "latency_ms_per_query":elapsed_seconds*1000/max(1,n)}
