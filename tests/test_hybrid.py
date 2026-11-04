import numpy as np
from src.hybrid import rrf_fuse

def test_rrf_places_shared_top_document_first():
    a=np.array([[0,1,2]])
    b=np.array([[0,2,1]])
    fused=rrf_fuse(a,b)
    assert fused[0,0] == 0
