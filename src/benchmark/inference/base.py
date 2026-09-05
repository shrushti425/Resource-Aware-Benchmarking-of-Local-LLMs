"""Base Inference Backend Interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class InferenceOutput:
    text: str
    prompt_tokens: int
    output_tokens: int
    ttft_sec: float
    generation_time_sec: float
    tokens_per_sec: float
    load_time_sec: float
    metadata: Dict[str, Any]


class BaseInferenceBackend(ABC):
    """Abstract base interface for model inference backends."""

    def __init__(self, model_path: str, context_length: int = 2048, n_gpu_layers: int = -1, cpu_threads: int = 4):
        self.model_path = model_path
        self.context_length = context_length
        self.n_gpu_layers = n_gpu_layers
        self.cpu_threads = cpu_threads
        self.is_loaded = False
        self.load_time_sec = 0.0

    @abstractmethod
    def load_model(self) -> float:
        """Loads the model into memory. Returns model load time in seconds."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.9,
        seed: int = 42,
        stop_sequences: Optional[List[str]] = None,
    ) -> InferenceOutput:
        """Executes text generation and returns timing and output metrics."""
        pass

    @abstractmethod
    def unload_model(self):
        """Frees model from memory."""
        pass
