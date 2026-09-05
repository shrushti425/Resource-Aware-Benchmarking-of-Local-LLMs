"""Automated Academic Research Report Generator."""

import sys
import datetime
from pathlib import Path
from typing import Optional

# Add src to path
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from benchmark.results.storage import ResultStorage
from benchmark.analysis.aggregator import BenchmarkAggregator
from benchmark.hardware import HardwareMonitor


def generate_research_report(output_dir: Optional[Path] = None) -> Path:
    """Generates comprehensive academic research report based strictly on empirical benchmark results."""
    if output_dir is None:
        output_dir = root_dir / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "benchmark_report.md"

    storage = ResultStorage(results_dir=root_dir / "results")
    records = storage.load_all_results()
    hw_info = HardwareMonitor.detect_hardware()

    has_data = len(records) > 0
    aggregator = BenchmarkAggregator(storage=storage)
    df_summary = aggregator.aggregate_summary() if has_data else None

    # Construct Markdown academic report
    lines = []
    lines.append("# Resource-Aware Benchmarking of Local LLMs on Apple Silicon for Offline AI and RAG Applications")
    lines.append(f"\n**Date:** {datetime.datetime.now().strftime('%B %d, %Y')}")
    lines.append("**Target Hardware Environment:** Apple Silicon MacBook Air M2 (8 GB Unified Memory, Metal Acceleration)")
    lines.append("\n---\n")

    lines.append("## 1. Abstract")
    lines.append(
        "Deploying Large Language Models (LLMs) locally on resource-constrained consumer hardware requires balancing answer accuracy, "
        "inference latency, and unified memory footprint. This paper presents a reproducible resource-aware experimental framework "
        "evaluating quantized open LLM families (Llama, Qwen, Gemma) across 7 benchmark tasks (QA, Classification, Summarization, Extraction, "
        "Structured JSON, Reasoning, RAG) on an Apple Silicon M2 laptop with 8 GB unified memory using `llama.cpp` Metal backend."
    )
    lines.append("\n---\n")

    lines.append("## 2. Introduction & Research Questions")
    lines.append("### Primary Research Question:")
    lines.append("> How do local LLM family, model size, and quantization affect answer quality, inference speed, latency, and memory consumption on a resource-constrained Apple Silicon laptop?")
    lines.append("\n### Secondary Research Questions:")
    lines.append("> Given an 8 GB unified memory limit, which local LLM configuration provides the best quality-efficiency trade-off?")
    lines.append("> How does the choice of local generation model affect answer quality and performance when the retrieval system remains fixed?")
    lines.append("\n---\n")

    lines.append("## 3. Methodology & System Hardware")
    lines.append("### Detected Hardware Diagnostics:")
    lines.append(f"- **Processor:** {hw_info['chip']}")
    lines.append(f"- **Operating System:** {hw_info['os']}")
    lines.append(f"- **Total Unified Memory:** {hw_info['total_unified_memory_gb']} GB")
    lines.append(f"- **Metal Acceleration:** {'Enabled' if hw_info['metal_available'] else 'CPU Fallback'}")

    lines.append("\n### Benchmark Tasks:")
    lines.append("1. **Question Answering (QA):** Exact Match & F1")
    lines.append("2. **Text Classification:** Accuracy & Macro F1")
    lines.append("3. **Summarization:** ROUGE-1, ROUGE-2, ROUGE-L")
    lines.append("4. **Information Extraction:** Entity F1")
    lines.append("5. **Structured JSON Generation:** JSON Validity & Schema Compliance")
    lines.append("6. **Reasoning:** Exact Match Accuracy")
    lines.append("7. **Retrieval-Augmented Generation (RAG):** Recall@K, MRR, Generation F1")
    lines.append("\n---\n")

    lines.append("## 4. Empirical Benchmark Results")

    if not has_data:
        lines.append("\n> [!WARNING]")
        lines.append("> **RESULTS NOT YET AVAILABLE**")
        lines.append(">\n> No empirical benchmark runs have been logged in `results/raw/benchmark_results.jsonl`. Execute `python -m benchmark run` to populate empirical measurements.")
    else:
        lines.append("\n### Summary Table:")
        lines.append("| Model | Family | Params (B) | Quantization | Quality | Latency (s) | Throughput (t/s) | Peak Memory (GB) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in df_summary.iterrows():
            lines.append(
                f"| {r['model']} | {r['family']} | {r['parameters_b']} | {r['quantization']} | {r['mean_quality']:.3f} | {r['mean_latency']:.2f} | {r['mean_tps']:.1f} | {r['mean_peak_memory']:.2f} |"
            )

    lines.append("\n---\n")
    lines.append("## 5. Limitations & Scientific Integrity")
    lines.append("- **Memory Constraint:** 8 GB unified memory restricts model parameter sizes to <=4B in Q4_K_M quantization.")
    lines.append("- **Automated Metrics:** String overlap and JSON compliance approximate semantic correctness.")
    lines.append("- **Hardware Monitoring:** macOS process RSS and system swap usage provide practical empirical bounds.")

    lines.append("\n---\n")
    lines.append("## 6. Conclusion")
    lines.append("This study establishes a practical, reproducible resource-aware selection methodology for offline local LLMs on Apple Silicon hardware.")

    report_content = "\n".join(lines)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"[✓] Academic research report generated at: {report_file}")
    return report_file


if __name__ == "__main__":
    generate_research_report()
