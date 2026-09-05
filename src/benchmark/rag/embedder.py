"""Lightweight local text embedding engine for Apple Silicon RAG pipeline."""

import math
import collections
from typing import List
import numpy as np


class LocalTextEmbedder:
    """Fast, deterministic, offline text embedding generator."""

    def __init__(self, vector_dim: int = 256):
        self.vector_dim = vector_dim

    def _hash_token(self, token: str) -> int:
        """Deterministic string hash into vector dimension index."""
        h = 0
        for char in token:
            h = (h * 31 + ord(char)) & 0xFFFFFFFF
        return h % self.vector_dim

    def embed_text(self, text: str) -> np.ndarray:
        """Embeds single text string into a L2-normalized dense vector."""
        tokens = text.lower().split()
        if not tokens:
            return np.zeros(self.vector_dim, dtype=np.float32)

        vec = np.zeros(self.vector_dim, dtype=np.float32)
        counts = collections.Counter(tokens)

        for token, count in counts.items():
            idx = self._hash_token(token)
            tf = 1.0 + math.log(count)
            vec[idx] += tf

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Embeds a batch of text documents."""
        embeddings = [self.embed_text(t) for t in texts]
        return np.vstack(embeddings).astype(np.float32)
