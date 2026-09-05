"""Configuration loader for the local LLM benchmarking framework."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class ModelConfig:
    id: str
    family: str
    name: str
    params_b: float
    repo_id: str
    filename: str
    quantization: str
    file_size_gb: float
    recommended_context: int = 2048
    enabled: bool = True
    feasibility_rating: str = "high"


@dataclass
class BenchmarkProfile:
    description: str
    samples_per_task: int
    warmup_runs: int
    iterations: int


@dataclass
class BenchmarkDefaults:
    temperature: float = 0.0
    top_p: float = 0.9
    seed: int = 42
    max_output_tokens: int = 256
    context_length: int = 2048
    n_gpu_layers: int = -1
    cpu_threads: int = 4
    batch_size: int = 512


@dataclass
class HardwareSafetyConfig:
    device_name: str = "MacBook Air M2"
    chip_family: str = "Apple Silicon"
    total_memory_gb: float = 8.0
    max_model_file_size_gb: float = 3.8
    min_available_memory_gb: float = 1.2
    warn_swap_usage_gb: float = 1.5
    infeasible_memory_threshold_gb: float = 5.5
    prefer_metal: bool = True
    fallback_to_cpu_on_error: bool = True


class BenchmarkConfig:
    def __init__(self, config_dir: Optional[Path] = None):
        if config_dir is None:
            config_dir = Path(__file__).resolve().parent.parent.parent / "configs"
        self.config_dir = Path(config_dir)

        self.models: List[ModelConfig] = []
        self.profiles: Dict[str, BenchmarkProfile] = {}
        self.defaults = BenchmarkDefaults()
        self.hardware = HardwareSafetyConfig()
        self.datasets: Dict[str, Dict[str, Any]] = {}

        self._load_all()

    def _load_all(self):
        models_file = self.config_dir / "models.yaml"
        if models_file.exists():
            with open(models_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                for m in data.get("models", []):
                    self.models.append(ModelConfig(**m))

        bench_file = self.config_dir / "benchmark.yaml"
        if bench_file.exists():
            with open(bench_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                for name, prof in data.get("profiles", {}).items():
                    self.profiles[name] = BenchmarkProfile(**prof)
                if "defaults" in data:
                    self.defaults = BenchmarkDefaults(**data["defaults"])

        hw_file = self.config_dir / "hardware.yaml"
        if hw_file.exists():
            with open(hw_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                target = data.get("target_hardware", {})
                thresh = data.get("safety_thresholds", {})
                metal = data.get("metal_acceleration", {})
                self.hardware = HardwareSafetyConfig(
                    device_name=target.get("device_name", "MacBook Air M2"),
                    chip_family=target.get("chip_family", "Apple Silicon"),
                    total_memory_gb=target.get("total_memory_gb", 8.0),
                    max_model_file_size_gb=thresh.get("max_model_file_size_gb", 3.8),
                    min_available_memory_gb=thresh.get("min_available_memory_gb", 1.2),
                    warn_swap_usage_gb=thresh.get("warn_swap_usage_gb", 1.5),
                    infeasible_memory_threshold_gb=thresh.get("infeasible_memory_threshold_gb", 5.5),
                    prefer_metal=metal.get("prefer_metal", True),
                    fallback_to_cpu_on_error=metal.get("fallback_to_cpu_on_error", True),
                )

        data_file = self.config_dir / "datasets.yaml"
        if data_file.exists():
            with open(data_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                self.datasets = data.get("datasets", {})

    def get_model(self, model_id: str) -> Optional[ModelConfig]:
        for m in self.models:
            if m.id == model_id:
                return m
        return None
