"""Result Storage Engine for JSONL append-only raw logging and CSV export."""

import json
import uuid
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd


class ResultStorage:
    """Manages raw experiment JSONL storage and processed CSV dataset aggregation."""

    def __init__(self, results_dir: Optional[Path] = None):
        if results_dir is None:
            results_dir = Path(__file__).resolve().parent.parent.parent.parent / "results"
        self.results_dir = Path(results_dir)
        self.raw_dir = self.results_dir / "raw"
        self.processed_dir = self.results_dir / "processed"

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def generate_run_id() -> str:
        """Generates unique timestamped run ID."""
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = str(uuid.uuid4())[:6]
        return f"run_{now}_{uid}"

    def append_result(self, record: Dict[str, Any], filename: str = "benchmark_results.jsonl"):
        """Appends single experiment result record to JSONL raw file."""
        target = self.raw_dir / filename
        with open(target, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load_all_results(self, filename: str = "benchmark_results.jsonl") -> List[Dict[str, Any]]:
        """Loads all raw JSONL experiment records."""
        target = self.raw_dir / filename
        if not target.exists():
            return []

        records = []
        with open(target, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
        return records

    def export_processed_csv(self, jsonl_filename: str = "benchmark_results.jsonl", csv_filename: str = "summary.csv") -> Path:
        """Converts JSONL experiment records into flattened pandas DataFrame CSV."""
        records = self.load_all_results(jsonl_filename)
        if not records:
            empty_df = pd.DataFrame()
            out_path = self.processed_dir / csv_filename
            empty_df.to_csv(out_path, index=False)
            return out_path

        flattened = []
        for r in records:
            row = {
                "run_id": r.get("run_id"),
                "timestamp": r.get("timestamp"),
                "model_id": r.get("model"),
                "model_family": r.get("family"),
                "parameters_b": r.get("parameters"),
                "quantization": r.get("quantization"),
                "task": r.get("task"),
                "profile": r.get("profile"),
                "sample_id": r.get("sample_id"),
                "status": r.get("status", "success"),
                "latency_sec": r.get("latency"),
                "ttft_sec": r.get("ttft"),
                "tokens_per_second": r.get("tokens_per_second"),
                "prompt_tokens": r.get("prompt_tokens"),
                "output_tokens": r.get("output_tokens"),
                "process_memory_gb": r.get("process_memory_gb"),
                "peak_memory_gb": r.get("peak_memory"),
                "available_memory_gb": r.get("hardware", {}).get("available_memory_gb"),
                "swap_used_gb": r.get("hardware", {}).get("swap_used_gb"),
            }
            # Flatten quality metrics
            q_metrics = r.get("quality_metrics", {})
            for k, v in q_metrics.items():
                row[f"metric_{k}"] = v

            flattened.append(row)

        df = pd.DataFrame(flattened)
        out_path = self.processed_dir / csv_filename
        df.to_csv(out_path, index=False)
        return out_path
