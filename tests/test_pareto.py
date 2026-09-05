"""Unit tests for Pareto Frontier analyzer."""

import pytest
import pandas as pd
from benchmark.analysis.pareto import ParetoAnalyzer


def test_pareto_frontier():
    df = pd.DataFrame([
        {"model": "m1", "mean_peak_memory": 1.0, "mean_quality": 0.50},  # Pareto (low mem, ok quality)
        {"model": "m2", "mean_peak_memory": 2.0, "mean_quality": 0.80},  # Pareto (higher qual)
        {"model": "m3", "mean_peak_memory": 2.5, "mean_quality": 0.70},  # Dominated by m2 (more mem, less qual)
    ])

    pa = ParetoAnalyzer()
    df_p = pa.compute_pareto_frontier(df, x_col="mean_peak_memory", y_col="mean_quality", x_maximize=False, y_maximize=True)
    assert len(df_p) == 2
    assert "m3" not in df_p["model"].values
    assert "m2" in df_p["model"].values
