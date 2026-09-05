"""Transparent Hardware-Aware Model Recommendation Engine."""

from typing import Dict, List, Any, Optional
import pandas as pd
from benchmark.analysis.aggregator import BenchmarkAggregator
from benchmark.results.storage import ResultStorage
from benchmark.hardware import HardwareMonitor


class ModelRecommender:
    """Recommends optimal local LLM configurations subject to RAM, latency, and quality constraints."""

    def __init__(self, storage: Optional[ResultStorage] = None):
        self.storage = storage or ResultStorage()
        self.aggregator = BenchmarkAggregator(storage=self.storage)

    def recommend(
        self,
        max_memory_gb: float = 4.0,
        max_latency_sec: float = 10.0,
        min_quality_score: float = 0.60,
        jsonl_filename: str = "benchmark_results.jsonl",
    ) -> Dict[str, Any]:
        """Filters infeasible and non-compliant models, returning transparently ranked recommendations."""
        df = self.aggregator.aggregate_summary(jsonl_filename)

        total_ram = HardwareMonitor.get_unified_memory_gb()

        if df.empty:
            return {
                "status": "no_data",
                "message": "No benchmark results found. Run the benchmark first.",
                "recommendations": [],
                "rejected": [],
            }

        recommendations = []
        rejected = []

        for _, row in df.iterrows():
            model_id = row["model"]
            mem_used = row["mean_peak_memory"]
            lat = row["mean_latency"]
            qual = row["mean_quality"]
            tps = row["mean_tps"]
            quant = row["quantization"]
            family = row["family"]

            reasons = []
            is_feasible = True

            # Constraint checks
            if mem_used > max_memory_gb:
                is_feasible = False
                reasons.append(f"Peak memory ({mem_used:.2f} GB) exceeds max threshold ({max_memory_gb:.2f} GB)")

            if lat > max_latency_sec:
                is_feasible = False
                reasons.append(f"Latency ({lat:.2f} s) exceeds max threshold ({max_latency_sec:.2f} s)")

            if qual < min_quality_score:
                is_feasible = False
                reasons.append(f"Quality ({qual:.2f}) is below minimum threshold ({min_quality_score:.2f})")

            item = {
                "model_id": model_id,
                "family": family,
                "quantization": quant,
                "mean_quality": qual,
                "mean_latency": lat,
                "tokens_per_second": tps,
                "peak_memory_gb": mem_used,
            }

            if is_feasible:
                # Calculate transparent composite efficiency rank score
                # 50% quality + 25% speed + 25% memory headroom
                mem_headroom = max(0.0, (max_memory_gb - mem_used) / max_memory_gb)
                speed_norm = min(1.0, tps / 50.0)
                rank_score = round(0.50 * qual + 0.25 * speed_norm + 0.25 * mem_headroom, 4)

                item["rank_score"] = rank_score
                item["recommendation_reason"] = (
                    f"Passed all constraints on {total_ram:.1f}GB RAM system. "
                    f"Achieved {qual:.2f} quality at {tps:.1f} tokens/sec using {mem_used:.2f}GB RAM."
                )
                recommendations.append(item)
            else:
                item["rejection_reasons"] = reasons
                rejected.append(item)

        # Sort recommendations by rank score descending
        recommendations.sort(key=lambda x: x["rank_score"], reverse=True)

        return {
            "status": "success",
            "constraints": {
                "max_memory_gb": max_memory_gb,
                "max_latency_sec": max_latency_sec,
                "min_quality_score": min_quality_score,
                "system_total_ram_gb": total_ram,
            },
            "recommendations": recommendations,
            "rejected": rejected,
        }
