"""Unit tests for deterministic metrics (EM, F1, ROUGE, JSON, Recall@K, MRR)."""

import pytest
from benchmark.metrics.text import exact_match_score, f1_score, classification_metrics, calculate_rouge_scores
from benchmark.metrics.json_eval import evaluate_json_output
from benchmark.metrics.rag_eval import calculate_retrieval_metrics, evaluate_rag_generation


def test_exact_match():
    assert exact_match_score("Apple M2", "apple m2") == 1.0
    assert exact_match_score("The Apple M2", "Apple M2") == 1.0
    assert exact_match_score("Intel Core i9", "Apple M2") == 0.0


def test_f1_score():
    assert f1_score("unified memory architecture", "unified memory") > 0.6
    assert f1_score("completely wrong answer", "apple m2") == 0.0


def test_classification_metrics():
    preds = ["Technology", "Finance", "Sports"]
    refs = ["Technology", "Finance", "Politics"]
    res = classification_metrics(preds, refs)
    assert res["accuracy"] == pytest.approx(0.6667, abs=0.01)


def test_rouge_scores():
    r = calculate_rouge_scores("local llm inference on apple silicon", "local llm benchmark on apple silicon")
    assert r["rouge1"] > 0.5
    assert r["rougeL"] > 0.5


def test_json_evaluation():
    valid_json = '{"name": "M2", "status": "active"}'
    eval_res = evaluate_json_output(valid_json, expected_schema_keys=["name", "status"])
    assert eval_res["json_validity"] == 1.0
    assert eval_res["schema_compliance"] == 1.0

    invalid_json = "not json at all"
    eval_inv = evaluate_json_output(invalid_json, expected_schema_keys=["name"])
    assert eval_inv["json_validity"] == 0.0


def test_rag_retrieval_metrics():
    retrieved = ["doc_01", "doc_02", "doc_03"]
    gt = ["doc_01"]
    res = calculate_retrieval_metrics(retrieved, gt, ks=[1, 3])
    assert res["recall_at_1"] == 1.0
    assert res["mrr"] == 1.0
