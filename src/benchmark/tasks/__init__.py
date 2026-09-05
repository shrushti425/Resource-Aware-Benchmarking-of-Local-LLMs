"""Task Hierarchy Package Export."""

from benchmark.tasks.base import BaseTask
from benchmark.tasks.qa import QATask
from benchmark.tasks.classification import ClassificationTask
from benchmark.tasks.summarization import SummarizationTask
from benchmark.tasks.extraction import InformationExtractionTask
from benchmark.tasks.json_gen import StructuredGenerationTask
from benchmark.tasks.reasoning import ReasoningTask
from benchmark.tasks.rag_task import RAGTask

TASK_REGISTRY = {
    "qa": QATask,
    "classification": ClassificationTask,
    "summarization": SummarizationTask,
    "extraction": InformationExtractionTask,
    "json": StructuredGenerationTask,
    "reasoning": ReasoningTask,
    "rag": RAGTask,
}


def get_task(task_name: str, dataset_path=None, prompt_path=None) -> BaseTask:
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task name '{task_name}'. Available: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[task_name](dataset_path=dataset_path, prompt_path=prompt_path)


__all__ = [
    "BaseTask",
    "QATask",
    "ClassificationTask",
    "SummarizationTask",
    "InformationExtractionTask",
    "StructuredGenerationTask",
    "ReasoningTask",
    "RAGTask",
    "TASK_REGISTRY",
    "get_task",
]
