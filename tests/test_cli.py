"""Unit tests for Click CLI commands."""

from click.testing import CliRunner
from benchmark.cli import main


def test_cli_hardware():
    runner = CliRunner()
    result = runner.invoke(main, ["hardware"])
    assert result.exit_code == 0
    assert "Apple Silicon Hardware Diagnostics" in result.output


def test_cli_models_list():
    runner = CliRunner()
    result = runner.invoke(main, ["models", "list"])
    assert result.exit_code == 0
    assert "Llama-3.2-1B-Instruct" in result.output


def test_cli_models_verify():
    runner = CliRunner()
    result = runner.invoke(main, ["models", "verify"])
    assert result.exit_code == 0
    assert "Model Files Verification" in result.output


def test_cli_run_mock():
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--profile", "quick", "--mock"])
    assert result.exit_code == 0
    assert "Benchmark execution complete" in result.output
