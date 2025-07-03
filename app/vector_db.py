import faiss
import numpy as np
import os

class VectorDB:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
    def add(self, vecs: np.ndarray):
        self.index.add(vecs)
    def search(self, vec: np.ndarray, k=1):
        return self.index.search(vec, k)