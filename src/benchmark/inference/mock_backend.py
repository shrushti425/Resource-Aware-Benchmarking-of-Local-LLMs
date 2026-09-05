"""Mock Inference Backend for synthetic evaluation and zero-model testing."""

import time
import random
from typing import Any, Dict, List, Optional
from benchmark.inference.base import BaseInferenceBackend, InferenceOutput


class MockInferenceBackend(BaseInferenceBackend):
    """Synthetic inference backend for fast dry-run benchmarks and automated tests."""

    def __init__(self, model_path: str = "mock-model", context_length: int = 2048, n_gpu_layers: int = -1, cpu_threads: int = 4):
        super().__init__(model_path, context_length, n_gpu_layers, cpu_threads)
        self.mock_tps = 35.0  # Simulated 35 tokens/sec on M2 Metal

    def load_model(self) -> float:
        self.load_time_sec = 0.15
        self.is_loaded = True
        return self.load_time_sec

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.9,
        seed: int = 42,
        stop_sequences: Optional[List[str]] = None,
    ) -> InferenceOutput:
        if not self.is_loaded:
            self.load_model()

        # Deterministic seed for mock reproducibility
        rng = random.Random(seed + len(prompt))

        # Generate realistic output based on prompt task type
        prompt_lower = prompt.lower()
        if "categories:" in prompt_lower or "category:" in prompt_lower:
            text = "Technology"
        elif "json" in prompt_lower or "schema" in prompt_lower:
            text = '{"name": "Apple M2", "architecture": "ARM64", "unified_memory_gb": 8, "status": "active"}'
        elif "summary:" in prompt_lower or "summarize" in prompt_lower:
            text = "The article discusses the performance and efficiency of local language models operating under unified memory constraints on Apple Silicon."
        elif "extracted" in prompt_lower or "extract" in prompt_lower:
            text = "Apple, MacBook Air, M2, Metal API"
        elif "final answer:" in prompt_lower or "problem:" in prompt_lower:
            text = "Step 1: Compute 12 * 8 = 96. Step 2: Add 4 = 100.\nFinal Answer: 100"
        elif "retrieved documents:" in prompt_lower:
            text = "Unified memory allows the M2 CPU and GPU to share high-bandwidth RAM directly without copying over PCIe."
        else:
            text = "The Apple Silicon M2 unified memory architecture provides high-bandwidth shared memory for efficient local LLM inference."

        prompt_tokens = max(10, len(prompt.split()) * 2)
        output_tokens = max(5, len(text.split()) * 2)

        # Simulate timing
        ttft_sec = rng.uniform(0.04, 0.12)
        gen_time_sec = ttft_sec + (output_tokens / self.mock_tps)
        tps = round(output_tokens / gen_time_sec, 2)

        return InferenceOutput(
            text=text,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            ttft_sec=round(ttft_sec, 4),
            generation_time_sec=round(gen_time_sec, 4),
            tokens_per_sec=tps,
            load_time_sec=self.load_time_sec,
            metadata={
                "n_gpu_layers": self.n_gpu_layers,
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed,
                "mock": True,
            },
        )

    def unload_model(self):
        self.is_loaded = False
