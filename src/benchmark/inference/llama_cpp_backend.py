"""llama.cpp Inference Backend via llama-cpp-python."""

import time
from typing import Any, Dict, List, Optional
from benchmark.inference.base import BaseInferenceBackend, InferenceOutput


class LlamaCppBackend(BaseInferenceBackend):
    """Inference engine leveraging llama-cpp-python with Metal acceleration on Apple Silicon."""

    def __init__(self, model_path: str, context_length: int = 2048, n_gpu_layers: int = -1, cpu_threads: int = 4):
        super().__init__(model_path, context_length, n_gpu_layers, cpu_threads)
        self.llm = None

    def load_model(self) -> float:
        """Loads GGUF model into memory using llama.cpp backend."""
        if self.is_loaded and self.llm is not None:
            return self.load_time_sec

        start_time = time.perf_counter()
        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python is not installed. Please install it using: "
                "CMAKE_ARGS='-DGGML_METAL=on' pip install llama-cpp-python"
            )

        print(f"[*] Loading model GGUF from '{self.model_path}' (n_gpu_layers={self.n_gpu_layers})...")
        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.context_length,
                n_gpu_layers=self.n_gpu_layers,
                n_threads=self.cpu_threads,
                verbose=False,
            )
        except Exception as e:
            if self.n_gpu_layers != 0:
                print(f"[!] Metal acceleration failed ({e}). Falling back to CPU execution (n_gpu_layers=0)...")
                self.n_gpu_layers = 0
                self.llm = Llama(
                    model_path=self.model_path,
                    n_ctx=self.context_length,
                    n_gpu_layers=0,
                    n_threads=self.cpu_threads,
                    verbose=False,
                )
            else:
                raise e

        self.load_time_sec = round(time.perf_counter() - start_time, 4)
        self.is_loaded = True
        print(f"[✓] Model successfully loaded in {self.load_time_sec:.2f}s.")
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
        """Generates completion using stream token timer to measure TTFT accurately."""
        if not self.is_loaded or self.llm is None:
            self.load_model()

        start_time = time.perf_counter()
        first_token_time = None
        output_text_chunks = []

        stream = self.llm(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            stop=stop_sequences or [],
            stream=True,
        )

        for chunk in stream:
            now = time.perf_counter()
            if first_token_time is None:
                first_token_time = now

            delta = chunk["choices"][0]["text"]
            output_text_chunks.append(delta)

        end_time = time.perf_counter()
        full_text = "".join(output_text_chunks).strip()

        if first_token_time is None:
            ttft_sec = end_time - start_time
        else:
            ttft_sec = first_token_time - start_time

        gen_time_sec = end_time - start_time

        # Token counting via llama.cpp tokenizer
        prompt_tokens = len(self.llm.tokenize(prompt.encode("utf-8")))
        output_tokens = len(self.llm.tokenize(full_text.encode("utf-8")))

        tps = round(output_tokens / gen_time_sec, 2) if gen_time_sec > 0 else 0.0

        return InferenceOutput(
            text=full_text,
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
            },
        )

    def unload_model(self):
        """Frees model from memory."""
        if self.llm is not None:
            del self.llm
            self.llm = None
        self.is_loaded = False
