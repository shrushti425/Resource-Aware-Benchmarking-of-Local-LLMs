"""Unit tests for model recommendation engine."""

import pytest
from benchmark.recommendation import ModelRecommender
from benchmark.results.storage import ResultStorage


def test_recommendation_empty():
    storage = ResultStorage()
    recommender = ModelRecommender(storage=storage)
    res = recommender.recommend()
    assert "status" in res
