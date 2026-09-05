"""Model Registry Manager."""

from pathlib import Path
from typing import Dict, List, Optional
from benchmark.config import BenchmarkConfig, ModelConfig
from benchmark.hardware import HardwareMonitor


class ModelRegistry:
    """Manages local LLM model configurations and verification."""

    def __init__(self, config: Optional[BenchmarkConfig] = None, models_dir: Optional[Path] = None):
        self.config = config or BenchmarkConfig()
        if models_dir is None:
            models_dir = Path(__file__).resolve().parent.parent.parent.parent / "models"
        self.models_dir = Path(models_dir)

    def list_models(self, include_disabled: bool = False) -> List[ModelConfig]:
        """Returns list of registered models."""
        if include_disabled:
            return self.config.models
        return [m for m in self.config.models if m.enabled]

    def get_model(self, model_id: str) -> Optional[ModelConfig]:
        """Gets model config by unique model ID."""
        return self.config.get_model(model_id)

    def get_model_file_path(self, model_id: str) -> Path:
        """Returns target file path for a model."""
        m = self.get_model(model_id)
        if not m:
            raise ValueError(f"Unknown model ID: {model_id}")
        return self.models_dir / m.filename

    def verify_models(self) -> Dict[str, Dict]:
        """Verifies local existence and hardware feasibility of registered models."""
        result = {}
        for m in self.config.models:
            local_path = self.models_dir / m.filename
            exists = local_path.exists()
            actual_size_gb = round(local_path.stat().st_size / (1024**3), 2) if exists else 0.0

            is_feasible, reason = HardwareMonitor.check_memory_feasibility(m.file_size_gb)

            result[m.id] = {
                "id": m.id,
                "name": m.name,
                "family": m.family,
                "quantization": m.quantization,
                "expected_size_gb": m.file_size_gb,
                "local_path": str(local_path),
                "exists": exists,
                "actual_size_gb": actual_size_gb,
                "feasible": is_feasible,
                "feasibility_reason": reason,
                "enabled": m.enabled,
            }
        return result
