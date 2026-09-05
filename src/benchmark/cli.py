"""Unified Command Line Interface for Local LLM Benchmarking System."""

import sys
from pathlib import Path
import click

# Ensure src in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from benchmark.config import BenchmarkConfig
from benchmark.hardware import HardwareMonitor
from benchmark.models.registry import ModelRegistry
from benchmark.models.downloader import ModelDownloader
from benchmark.evaluation.evaluator import BenchmarkEvaluator
from benchmark.analysis.aggregator import BenchmarkAggregator
from benchmark.analysis.plotting import BenchmarkPlotter
from benchmark.recommendation import ModelRecommender
from scripts.generate_report import generate_research_report
from scripts.download_datasets import generate_local_datasets


@click.group()
def main():
    """Resource-Aware Local LLM Benchmarking Tool for Apple Silicon."""
    pass


@main.command()
def hardware():
    """Display Apple Silicon hardware diagnostics and memory availability."""
    click.echo("=== Apple Silicon Hardware Diagnostics ===")
    hw = HardwareMonitor.detect_hardware()
    for k, v in hw.items():
        click.echo(f"  {k:30s}: {v}")


@main.group()
def models():
    """Manage local LLM models and registries."""
    pass


@models.command(name="list")
def models_list():
    """List registered LLM models."""
    reg = ModelRegistry()
    click.echo("=== Registered Local LLMs ===")
    click.echo(f"{'ID':20s} {'NAME':25s} {'FAMILY':10s} {'PARAMS':8s} {'QUANT':8s} {'SIZE(GB)':10s} {'FEASIBILITY'}")
    click.echo("-" * 95)
    for m in reg.list_models(include_disabled=True):
        status = "ENABLED" if m.enabled else "DISABLED"
        click.echo(f"{m.id:20s} {m.name:25s} {m.family:10s} {m.params_b:<8.2f} {m.quantization:8s} {m.file_size_gb:<10.2f} {m.feasibility_rating}")


@models.command(name="verify")
def models_verify():
    """Verify local existence and memory feasibility of GGUF model files."""
    reg = ModelRegistry()
    ver = reg.verify_models()
    click.echo("=== Model Files Verification ===")
    for m_id, info in ver.items():
        exists_str = "EXISTS" if info["exists"] else "MISSING"
        feas_str = "SAFE" if info["feasible"] else f"WARNING ({info['feasibility_reason']})"
        click.echo(f"[{exists_str:7s}] {m_id:20s} ({info['quantization']}) - {feas_str}")


@models.command(name="download")
@click.option("--model", "model_id", required=True, help="Model ID to download from Hugging Face.")
def models_download(model_id: str):
    """Download specific GGUF model file explicitly."""
    dl = ModelDownloader()
    dl.download_model(model_id)


@main.group()
def datasets():
    """Manage benchmark datasets."""
    pass


@datasets.command(name="download")
def datasets_download():
    """Pre-fetch and initialize benchmark dataset files."""
    generate_local_datasets()


@main.command()
@click.option("--profile", default="quick", help="Benchmark profile (quick, standard, full).")
@click.option("--model", "model_id", default=None, help="Target specific model ID.")
@click.option("--task", "task_name", default=None, help="Target specific task name.")
@click.option("--mock/--no-mock", default=False, help="Use synthetic mock backend for testing.")
@click.option("--resume/--no-resume", default=False, help="Resume benchmark skipping completed runs.")
def run(profile: str, model_id: str, task_name: str, mock: bool, resume: bool):
    """Run benchmark experiments on local LLMs."""
    evaluator = BenchmarkEvaluator(use_mock=mock)
    evaluator.run_benchmark(profile_name=profile, model_id=model_id, task_name=task_name, resume=resume)


@main.command()
def evaluate():
    """Evaluate and aggregate raw benchmark experiment data."""
    agg = BenchmarkAggregator()
    df = agg.aggregate_summary()
    if df.empty:
        click.echo("[!] No benchmark results found. Run 'python -m benchmark run' first.")
    else:
        click.echo("=== Aggregated Benchmark Metrics ===")
        click.echo(df.to_string())


@main.command()
def analyze():
    """Compute statistics and generate all 9 research figures in plots/."""
    plotter = BenchmarkPlotter()
    count = plotter.generate_all_plots()
    click.echo(f"[✓] Analysis completed. {count} figures generated in plots/.")


@main.command()
@click.option("--max-memory", default=4.0, type=float, help="Maximum allowed peak memory (GB).")
@click.option("--max-latency", default=10.0, type=float, help="Maximum allowed end-to-end latency (seconds).")
@click.option("--min-quality", default=0.60, type=float, help="Minimum quality score threshold.")
def recommend(max_memory: float, max_latency: float, min_quality: float):
    """Recommend optimal model configuration under hardware constraints."""
    recommender = ModelRecommender()
    res = recommender.recommend(max_memory_gb=max_memory, max_latency_sec=max_latency, min_quality_score=min_quality)
    if res["status"] == "no_data":
        click.echo(f"[!] {res['message']}")
        return

    click.echo(f"=== Model Recommendations (Max RAM: {max_memory}GB, Max Latency: {max_latency}s, Min Quality: {min_quality}) ===")
    for idx, rec in enumerate(res["recommendations"], start=1):
        click.echo(f"Rank {idx}: {rec['model_id']} (Score: {rec['rank_score']:.3f})")
        click.echo(f"  Reason: {rec['recommendation_reason']}\n")

    if res["rejected"]:
        click.echo("--- Infeasible / Rejected Configurations ---")
        for r in res["rejected"]:
            click.echo(f"Rejected {r['model_id']}: {', '.join(r['rejection_reasons'])}")


@main.command()
def report():
    """Generate academic benchmark research report in reports/benchmark_report.md."""
    generate_research_report()


if __name__ == "__main__":
    main()
