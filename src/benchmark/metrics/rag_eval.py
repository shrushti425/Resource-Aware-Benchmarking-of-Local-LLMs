"""RAG Metrics: Recall@K, Mean Reciprocal Rank (MRR), and Generation F1/Exact Match."""

from typing import List, Dict
from benchmark.metrics.text import exact_match_score, f1_score


def calculate_retrieval_metrics(retrieved_doc_ids: List[str], ground_truth_doc_ids: List[str], ks: List[int] = [1, 3, 5]) -> Dict[str, float]:
    """Calculates Recall@K for K in [1, 3, 5] and Mean Reciprocal Rank (MRR)."""
    gt_set = set(ground_truth_doc_ids)
    if not gt_set:
        return {f"recall_at_{k}": 0.0 for k in ks} | {"mrr": 0.0}

    results = {}

    # Recall@K
    for k in ks:
        top_k = set(retrieved_doc_ids[:k])
        hits = len(gt_set.intersection(top_k))
        results[f"recall_at_{k}"] = round(hits / len(gt_set), 4)

    # MRR (Reciprocal Rank of first relevant document)
    rr = 0.0
    for idx, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in gt_set:
            rr = 1.0 / idx
            break

    results["mrr"] = round(rr, 4)
    return results


def evaluate_rag_generation(prediction: str, reference: str) -> Dict[str, float]:
    """Evaluates RAG generated answer against ground truth answer."""
    em = exact_match_score(prediction, reference)
    f1 = f1_score(prediction, reference)
    return {
        "exact_match": em,
        "f1": f1,
    }
