"""Statistical aggregation pipeline for raw benchmark results."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from benchmark.results.storage import ResultStorage


class BenchmarkAggregator:
    """Aggregates raw experiment runs into statistical summaries (mean, median, std, min, max)."""

    def __init__(self, storage: Optional[ResultStorage] = None):
        self.storage = storage or ResultStorage()

    def aggregate_summary(self, jsonl_filename: str = "benchmark_results.jsonl") -> pd.DataFrame:
        """Loads raw JSONL records and computes aggregated metrics grouped by model and task."""
        records = self.storage.load_all_results(jsonl_filename)
        if not records:
            return pd.DataFrame()

        rows = []
        for r in records:
            if r.get("status") != "success":
                continue

            model = r.get("model")
            task = r.get("task")
            quant = r.get("quantization")
            params = r.get("parameters")
            family = r.get("family")
            latency = r.get("latency", 0.0)
            ttft = r.get("ttft", 0.0)
            tps = r.get("tokens_per_second", 0.0)
            peak_mem = r.get("peak_memory", 0.0)

            # Primary quality score extracted from task quality metrics
            q_metrics = r.get("quality_metrics", {})
            q_score = 0.0
            if "accuracy" in q_metrics:
                q_score = q_metrics["accuracy"]
            elif "f1" in q_metrics:
                q_score = q_metrics["f1"]
            elif "exact_match" in q_metrics:
                q_score = q_metrics["exact_match"]
            elif "json_validity" in q_metrics:
                q_score = q_metrics["json_validity"]
            elif "rougeL" in q_metrics:
                q_score = q_metrics["rougeL"]

            rows.append({
                "model": model,
                "family": family,
                "parameters_b": params,
                "quantization": quant,
                "task": task,
                "quality_score": q_score,
                "latency_sec": latency,
                "ttft_sec": ttft,
                "tokens_per_second": tps,
                "peak_memory_gb": peak_mem,
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame()

        grouped = df.groupby(["model", "family", "parameters_b", "quantization"]).agg(
            mean_quality=("quality_score", "mean"),
            median_quality=("quality_score", "median"),
            std_quality=("quality_score", "std"),
            mean_latency=("latency_sec", "mean"),
            mean_ttft=("ttft_sec", "mean"),
            mean_tps=("tokens_per_second", "mean"),
            max_tps=("tokens_per_second", "max"),
            mean_peak_memory=("peak_memory_gb", "mean"),
            max_peak_memory=("peak_memory_gb", "max"),
            sample_count=("quality_score", "count"),
        ).reset_index()

        # Fill NaNs in std
        grouped["std_quality"] = grouped["std_quality"].fillna(0.0)
        return grouped
