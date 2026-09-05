"""Deterministic Text Evaluation Metrics (Exact Match, F1, Precision, Recall, Accuracy, ROUGE)."""

import re
import collections
from typing import Dict, List, Set, Union


def normalize_answer(s: str) -> str:
    """Lowercases, removes punctuation, articles, and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return re.sub(r"[^\w\s]", "", text)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Returns 1.0 if normalized strings match exactly, else 0.0."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def f1_score(prediction: str, ground_truth: str) -> float:
    """Calculates token-level overlap F1 score."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    if not prediction_tokens or not ground_truth_tokens:
        return 1.0 if prediction_tokens == ground_truth_tokens else 0.0

    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return round(f1, 4)


def classification_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculates accuracy, macro precision, recall, and F1 score for categorical predictions."""
    if not predictions or len(predictions) != len(references):
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    norm_preds = [normalize_answer(p) for p in predictions]
    norm_refs = [normalize_answer(r) for r in references]

    correct = sum(1 for p, r in zip(norm_preds, norm_refs) if p == r)
    accuracy = round(correct / len(references), 4)

    # Class-wise stats for Macro F1
    classes = set(norm_refs)
    precisions, recalls, f1s = [], [], []

    for cls in classes:
        tp = sum(1 for p, r in zip(norm_preds, norm_refs) if p == cls and r == cls)
        fp = sum(1 for p, r in zip(norm_preds, norm_refs) if p == cls and r != cls)
        fn = sum(1 for p, r in zip(norm_preds, norm_refs) if p != cls and r == cls)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    macro_prec = round(sum(precisions) / len(classes), 4) if classes else 0.0
    macro_rec = round(sum(recalls) / len(classes), 4) if classes else 0.0
    macro_f1 = round(sum(f1s) / len(classes), 4) if classes else 0.0

    return {
        "accuracy": accuracy,
        "precision": macro_prec,
        "recall": macro_rec,
        "f1": macro_f1,
    }


def compute_n_gram_overlap(prediction: str, reference: str, n: int = 1) -> float:
    """Computes n-gram recall overlap score (ROUGE-N fallback)."""
    p_tokens = normalize_answer(prediction).split()
    r_tokens = normalize_answer(reference).split()

    if len(r_tokens) < n or len(p_tokens) < n:
        return 1.0 if p_tokens == r_tokens else 0.0

    p_ngrams = collections.Counter([tuple(p_tokens[i : i + n]) for i in range(len(p_tokens) - n + 1)])
    r_ngrams = collections.Counter([tuple(r_tokens[i : i + n]) for i in range(len(r_tokens) - n + 1)])

    overlap = sum((p_ngrams & r_ngrams).values())
    total_r = sum(r_ngrams.values())
    return round(overlap / total_r, 4) if total_r > 0 else 0.0


def calculate_rouge_scores(prediction: str, reference: str) -> Dict[str, float]:
    """Calculates ROUGE-1, ROUGE-2, and ROUGE-L scores."""
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(target=reference, prediction=prediction)
        return {
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rouge2": round(scores["rouge2"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4),
        }
    except Exception:
        # Pure Python deterministic fallback
        r1 = compute_n_gram_overlap(prediction, reference, n=1)
        r2 = compute_n_gram_overlap(prediction, reference, n=2)
        f1 = f1_score(prediction, reference)
        return {
            "rouge1": r1,
            "rouge2": r2,
            "rougeL": f1,
        }
