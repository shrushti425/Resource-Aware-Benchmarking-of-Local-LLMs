"""Unit tests for local RAG embedder, vector store, and search."""

import pytest
from benchmark.rag.embedder import LocalTextEmbedder
from benchmark.rag.vector_store import SimpleVectorStore
from benchmark.rag.pipeline import RAGPipeline


def test_local_embedder():
    embedder = LocalTextEmbedder(vector_dim=128)
    vec = embedder.embed_text("Apple M2 unified memory architecture")
    assert vec.shape == (128,)


def test_vector_store():
    store = SimpleVectorStore(vector_dim=128)
    embedder = LocalTextEmbedder(vector_dim=128)

    docs = [
        {"doc_id": "d1", "text": "Apple M2 processor"},
        {"doc_id": "d2", "text": "Intel Core processor"},
    ]
    vecs = embedder.embed_documents([d["text"] for d in docs])
    store.add_documents(docs, vecs)

    q_vec = embedder.embed_text("Apple M2")
    res = store.search(q_vec, top_k=1)
    assert len(res) == 1
    assert res[0][0]["doc_id"] == "d1"
