"""Vector store index with FAISS and NumPy fallback."""

from typing import Dict, List, Tuple, Any
import numpy as np


class SimpleVectorStore:
    """Local vector store index for RAG chunk retrieval."""

    def __init__(self, vector_dim: int = 256):
        self.vector_dim = vector_dim
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: List[np.ndarray] = []
        self.faiss_index = None
        self.use_faiss = False

        try:
            import faiss

            self.faiss_index = faiss.IndexFlatIP(vector_dim)
            self.use_faiss = True
        except ImportError:
            self.use_faiss = False

    def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        """Adds documents and associated normalized embedding vectors."""
        self.documents.extend(documents)
        if self.use_faiss and self.faiss_index is not None:
            self.faiss_index.add(embeddings.astype(np.float32))
        else:
            for vec in embeddings:
                self.embeddings.append(vec.astype(np.float32))

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """Searches top-K most similar document chunks."""
        if not self.documents:
            return []

        query_vector = query_vector.astype(np.float32).reshape(1, -1)

        if self.use_faiss and self.faiss_index is not None:
            scores, indices = self.faiss_index.search(query_vector, min(top_k, len(self.documents)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents) and idx >= 0:
                    results.append((self.documents[idx], float(score)))
            return results
        else:
            # NumPy Cosine / Inner Product similarity fallback
            matrix = np.vstack(self.embeddings)
            scores = np.dot(matrix, query_vector.squeeze())
            top_indices = np.argsort(scores)[::-1][:top_k]
            results = []
            for idx in top_indices:
                results.append((self.documents[idx], float(scores[idx])))
            return results
