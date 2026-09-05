"""Pareto Frontier Analysis Engine for Quality vs Efficiency trade-offs."""

from typing import List, Dict, Any
import pandas as pd
import numpy as np


class ParetoAnalyzer:
    """Multi-objective Pareto frontier identification engine."""

    @staticmethod
    def is_pareto_efficient(costs: np.ndarray, maximize_mask: List[bool]) -> np.ndarray:
        """Finds non-dominated Pareto points.

        Args:
            costs: (n_points, n_objectives) numpy array.
            maximize_mask: list of booleans indicating True if objective is maximized, False if minimized.
        """
        # Convert to minimization form by negating maximized objectives
        c = costs.copy()
        for idx, max_flag in enumerate(maximize_mask):
            if max_flag:
                c[:, idx] = -c[:, idx]

        is_efficient = np.ones(c.shape[0], dtype=bool)
        for i, c_i in enumerate(c):
            if is_efficient[i]:
                # Keep points that are not strictly dominated by c_i
                is_efficient[is_efficient] = np.any(c[is_efficient] < c_i, axis=1) | np.all(c[is_efficient] == c_i, axis=1)
                is_efficient[i] = True
        return is_efficient

    def compute_pareto_frontier(
        self,
        df_summary: pd.DataFrame,
        x_col: str = "mean_peak_memory",
        y_col: str = "mean_quality",
        x_maximize: bool = False,
        y_maximize: bool = True,
    ) -> pd.DataFrame:
        """Identifies 2D Pareto frontier from aggregated benchmark dataframe."""
        if df_summary.empty or x_col not in df_summary.columns or y_col not in df_summary.columns:
            return pd.DataFrame()

        clean_df = df_summary.dropna(subset=[x_col, y_col]).copy()
        if clean_df.empty:
            return pd.DataFrame()

        costs = clean_df[[x_col, y_col]].to_numpy()
        mask = [x_maximize, y_maximize]

        is_pareto = self.is_pareto_efficient(costs, mask)
        clean_df["is_pareto"] = is_pareto
        return clean_df[clean_df["is_pareto"]].sort_values(by=x_col)
