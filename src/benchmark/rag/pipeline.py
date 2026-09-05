"""End-to-end local RAG pipeline with fixed retrieval engine."""

from typing import Dict, List, Tuple, Any
from benchmark.rag.embedder import LocalTextEmbedder
from benchmark.rag.vector_store import SimpleVectorStore
from benchmark.inference.base import BaseInferenceBackend, InferenceOutput


class RAGPipeline:
    """Fixed local retrieval pipeline connecting embedder, vector store, and generation LLM."""

    def __init__(self, vector_dim: int = 256):
        self.embedder = LocalTextEmbedder(vector_dim=vector_dim)
        self.vector_store = SimpleVectorStore(vector_dim=vector_dim)

    def index_documents(self, documents: List[Dict[str, Any]]):
        """Indexes text documents with metadata (doc_id, text, title)."""
        texts = [doc["text"] for doc in documents]
        embeddings = self.embedder.embed_documents(texts)
        self.vector_store.add_documents(documents, embeddings)

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """Retrieves top-K relevant document chunks for query."""
        q_vec = self.embedder.embed_text(query)
        return self.vector_store.search(q_vec, top_k=top_k)

    def run_rag_generation(
        self,
        query: str,
        backend: BaseInferenceBackend,
        prompt_template: str,
        top_k: int = 3,
        max_tokens: int = 256,
    ) -> Tuple[InferenceOutput, List[Dict[str, Any]]]:
        """Runs retrieval, formats prompt with retrieved chunks, and generates answer with local LLM."""
        retrieved = self.retrieve(query, top_k=top_k)
        retrieved_docs = [doc for doc, score in retrieved]

        formatted_chunks = "\n---\n".join([f"Document [{doc['doc_id']}]: {doc['text']}" for doc in retrieved_docs])
        prompt = prompt_template.format(retrieved_chunks=formatted_chunks, question=query)

        output = backend.generate(prompt=prompt, max_tokens=max_tokens)
        return output, retrieved_docs
