"""Unit tests for task loading and formatting."""

import pytest
from benchmark.tasks import get_task


def test_task_instantiation():
    for task_name in ["qa", "classification", "summarization", "extraction", "json", "reasoning", "rag"]:
        t = get_task(task_name)
        assert t.task_name == task_name
        samples = t.load_dataset(sample_limit=2)
        assert len(samples) > 0
        prompt = t.format_prompt(samples[0])
        assert isinstance(prompt, str) and len(prompt) > 0
