import numpy as np
from src.metrics import recall_at_k, mean_reciprocal_rank

def test_recall_and_mrr():
    rankings=np.array([[2,0,1],[0,1,2]])
    gold=np.array([0,2])
    assert recall_at_k(rankings,gold,2)==0.5
    assert abs(mean_reciprocal_rank(rankings,gold)-(0.5+1/3)/2)<1e-12
