"""Benchmark Evaluator & Core Experiment Engine."""

import datetime
from typing import Any, Dict, List, Optional
from benchmark.config import BenchmarkConfig
from benchmark.hardware import HardwareMonitor
from benchmark.models.registry import ModelRegistry
from benchmark.inference.base import BaseInferenceBackend
from benchmark.inference.llama_cpp_backend import LlamaCppBackend
from benchmark.inference.mock_backend import MockInferenceBackend
from benchmark.tasks import TASK_REGISTRY, get_task
from benchmark.monitoring.timer import BenchmarkTimer
from benchmark.monitoring.resource import ResourceMonitor
from benchmark.results.storage import ResultStorage
from benchmark.results.resume import ResumeManager


class BenchmarkEvaluator:
    """Core experiment runner carrying out warm-up, inference, resource tracking, and result storage."""

    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        storage: Optional[ResultStorage] = None,
        use_mock: bool = False,
    ):
        self.config = config or BenchmarkConfig()
        self.storage = storage or ResultStorage()
        self.use_mock = use_mock
        self.registry = ModelRegistry(config=self.config)
        self.hw_info = HardwareMonitor.detect_hardware()

    def run_benchmark(
        self,
        profile_name: str = "quick",
        model_id: Optional[str] = None,
        task_name: Optional[str] = None,
        resume: bool = False,
    ) -> List[Dict[str, Any]]:
        """Executes full benchmark suite across selected models and tasks."""
        profile = self.config.profiles.get(profile_name, self.config.profiles["quick"])
        run_id = self.storage.generate_run_id()
        resume_mgr = ResumeManager(self.storage) if resume else None

        # Determine target models
        target_models = []
        if model_id:
            m = self.registry.get_model(model_id)
            if not m:
                raise ValueError(f"Model ID '{model_id}' not found in registry.")
            target_models = [m]
        else:
            target_models = self.registry.list_models(include_disabled=False)

        # Target tasks
        target_tasks = [task_name] if task_name else list(TASK_REGISTRY.keys())

        print(f"==================================================")
        print(f"Starting Local LLM Benchmark Run: {run_id}")
        print(f"Hardware: {self.hw_info['chip']} ({self.hw_info['total_unified_memory_gb']} GB RAM)")
        print(f"Profile: {profile_name.upper()} ({profile.samples_per_task} samples/task)")
        print(f"Backend Mode: {'MOCK (Synthetic)' if self.use_mock else 'llama.cpp (Metal/GGUF)'}")
        print(f"Models: {[m.id for m in target_models]}")
        print(f"Tasks: {target_tasks}")
        print(f"==================================================")

        all_results = []

        for model_cfg in target_models:
            print(f"\n---> Evaluating Model: [{model_cfg.id}] ({model_cfg.name})")

            # Memory feasibility guard
            is_feasible, reason = HardwareMonitor.check_memory_feasibility(model_cfg.file_size_gb)
            if not is_feasible and not self.use_mock:
                print(f"[!] INFEASIBLE: Skipping model {model_cfg.id}. Reason: {reason}")
                # Log infeasible record
                infeasible_record = {
                    "run_id": run_id,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "model": model_cfg.id,
                    "family": model_cfg.family,
                    "parameters": model_cfg.params_b,
                    "quantization": model_cfg.quantization,
                    "status": "infeasible_for_current_hardware",
                    "infeasibility_reason": reason,
                    "hardware": self.hw_info,
                }
                self.storage.append_result(infeasible_record)
                all_results.append(infeasible_record)
                continue

            # Instantiate inference backend
            backend: BaseInferenceBackend
            if self.use_mock:
                backend = MockInferenceBackend(
                    model_path=model_cfg.id,
                    context_length=model_cfg.recommended_context,
                )
            else:
                model_file = self.registry.get_model_file_path(model_cfg.id)
                if not model_file.exists():
                    print(f"[!] ERROR: Model file missing at {model_file}. Run 'python -m benchmark models download --model {model_cfg.id}' first.")
                    continue

                backend = LlamaCppBackend(
                    model_path=str(model_file),
                    context_length=model_cfg.recommended_context,
                    n_gpu_layers=self.config.defaults.n_gpu_layers,
                    cpu_threads=self.config.defaults.cpu_threads,
                )

            # Load model
            try:
                load_time = backend.load_model()
            except Exception as e:
                print(f"[!] FAILED to load model {model_cfg.id}: {e}")
                continue

            # Perform Warm-Up Inference
            print(f"[*] Executing {profile.warmup_runs} warm-up run(s)...")
            for _ in range(profile.warmup_runs):
                backend.generate("Warm-up prompt.", max_tokens=10)

            resource_mon = ResourceMonitor()

            # Execute tasks
            for t_name in target_tasks:
                print(f"  [Task: {t_name}] Loading dataset...")
                task_inst = get_task(t_name)
                samples = task_inst.load_dataset(sample_limit=profile.samples_per_task)

                for idx, sample in enumerate(samples):
                    sample_id = sample.get("id", f"sample_{idx}")

                    if resume_mgr and resume_mgr.is_completed(model_cfg.id, t_name, sample_id):
                        print(f"    [Skip] {t_name}/{sample_id} already evaluated.")
                        continue

                    # Prepare prompt
                    prompt_text = task_inst.format_prompt(sample)

                    # Handle RAG Task retrieval specifically
                    retrieved_ids = []
                    if t_name == "rag":
                        retrieved = task_inst.rag_pipeline.retrieve(sample["question"], top_k=3)
                        retrieved_docs = [d for d, s in retrieved]
                        retrieved_ids = [d["doc_id"] for d in retrieved_docs]
                        sample["retrieved_doc_ids"] = retrieved_ids
                        formatted_chunks = "\n---\n".join([f"Document [{d['doc_id']}]: {d['text']}" for d in retrieved_docs])
                        prompt_text = task_inst.prompt_template.format(retrieved_chunks=formatted_chunks, question=sample["question"])

                    # Execute timed inference
                    with BenchmarkTimer() as timer:
                        inf_out = backend.generate(
                            prompt=prompt_text,
                            max_tokens=self.config.defaults.max_output_tokens,
                            temperature=self.config.defaults.temperature,
                            top_p=self.config.defaults.top_p,
                            seed=self.config.defaults.seed,
                        )

                    # Resource snapshot
                    res_snap = resource_mon.get_resource_snapshot()

                    # Evaluate quality metric for single sample
                    quality_metrics = task_inst.evaluate_sample(inf_out.text, sample)

                    # Construct JSON record matching schema requirements
                    record = {
                        "run_id": run_id,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "model": model_cfg.id,
                        "family": model_cfg.family,
                        "parameters": model_cfg.params_b,
                        "quantization": model_cfg.quantization,
                        "task": t_name,
                        "profile": profile_name,
                        "sample_id": sample_id,
                        "status": "success",
                        "prompt_version": "1.0",
                        "temperature": self.config.defaults.temperature,
                        "seed": self.config.defaults.seed,
                        "context_length": model_cfg.recommended_context,
                        "hardware": self.hw_info,
                        "latency": timer.elapsed_sec,
                        "ttft": inf_out.ttft_sec,
                        "tokens_per_second": inf_out.tokens_per_sec,
                        "prompt_tokens": inf_out.prompt_tokens,
                        "output_tokens": inf_out.output_tokens,
                        "process_memory_gb": res_snap["process_memory_gb"],
                        "peak_memory": res_snap["peak_process_memory_gb"],
                        "quality_metrics": quality_metrics,
                        "prediction_text": inf_out.text,
                    }

                    if retrieved_ids:
                        record["retrieved_doc_ids"] = retrieved_ids

                    self.storage.append_result(record)
                    all_results.append(record)

            # Unload model after completing all tasks
            backend.unload_model()

        # Update processed CSV
        self.storage.export_processed_csv()
        print(f"\n[✓] Benchmark execution complete. Total samples evaluated: {len(all_results)}")
        return all_results
