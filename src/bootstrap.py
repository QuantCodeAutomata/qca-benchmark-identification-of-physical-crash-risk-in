"""
Circular block bootstrap for inference on identified-set statistics.

Implements horizon-stratified circular block bootstrap on calendar weeks
with block length 8 weeks and B=2000 replications, as specified in exp_2.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def circular_block_bootstrap_weeks(
    results_df: pd.DataFrame,
    stat_fn: Callable[[pd.DataFrame], Dict],
    block_length: int = 8,
    n_boot: int = 2000,
    horizons: Optional[List[int]] = None,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Circular block bootstrap on calendar weeks, stratified by horizon.

    For each replication:
    1. For each horizon, resample calendar weeks in circular blocks of
       length block_length weeks.
    2. Reconstruct the set of cross-sections for the resampled weeks.
    3. Compute statistics using stat_fn.

    Args:
        results_df: DataFrame with columns 'date', 'horizon', and result columns.
            Must have a 'week' column or one will be created from 'date'.
        stat_fn: Function mapping a DataFrame to a Dict of scalar statistics.
        block_length: Length of each bootstrap block in weeks.
        n_boot: Number of bootstrap replications.
        horizons: Horizons to stratify on (default: all unique horizons).
        seed: Random seed.

    Returns:
        Dict mapping statistic name -> np.ndarray of shape (n_boot,) bootstrap values.
    """
    rng = np.random.default_rng(seed)

    if horizons is None:
        horizons = sorted(results_df["horizon"].unique())

    # Add week identifier
    df = results_df.copy()
    if "week" not in df.columns:
        df["week"] = df["date"].dt.isocalendar().week.astype(int)
        df["year"] = df["date"].dt.year
        df["week_id"] = df["year"].astype(str) + "_" + df["week"].astype(str)

    # Collect weeks per horizon
    horizon_weeks: Dict[int, List[str]] = {}
    for h in horizons:
        h_df = df[df["horizon"] == h]
        weeks = sorted(h_df["week_id"].unique())
        horizon_weeks[h] = weeks

    # Reference statistics
    ref_stats = stat_fn(df)
    stat_names = list(ref_stats.keys())
    boot_stats: Dict[str, List[float]] = {k: [] for k in stat_names}

    for b in range(n_boot):
        boot_dfs = []
        for h in horizons:
            weeks = horizon_weeks[h]
            n_weeks = len(weeks)
            if n_weeks == 0:
                continue
            # Number of blocks needed to cover n_weeks
            n_blocks = int(np.ceil(n_weeks / block_length))
            # Circular: start indices wrap around
            starts = rng.integers(0, n_weeks, size=n_blocks)
            boot_week_ids = []
            for start in starts:
                for offset in range(block_length):
                    boot_week_ids.append(weeks[(start + offset) % n_weeks])
            # Trim to n_weeks
            boot_week_ids = boot_week_ids[:n_weeks]

            h_df = df[df["horizon"] == h]
            boot_h_df = pd.concat(
                [h_df[h_df["week_id"] == w] for w in boot_week_ids],
                ignore_index=True,
            )
            boot_dfs.append(boot_h_df)

        if boot_dfs:
            boot_df = pd.concat(boot_dfs, ignore_index=True)
            try:
                b_stats = stat_fn(boot_df)
                for k in stat_names:
                    boot_stats[k].append(b_stats.get(k, np.nan))
            except Exception as e:
                logger.debug("Bootstrap rep %d failed: %s", b, e)
                for k in stat_names:
                    boot_stats[k].append(np.nan)

    return {k: np.array(v) for k, v in boot_stats.items()}


def percentile_ci(
    boot_values: np.ndarray, alpha: float = 0.05
) -> Tuple[float, float]:
    """Compute (alpha/2, 1-alpha/2) percentile bootstrap CI."""
    valid = boot_values[~np.isnan(boot_values)]
    if len(valid) == 0:
        return np.nan, np.nan
    lo = float(np.percentile(valid, 100 * alpha / 2))
    hi = float(np.percentile(valid, 100 * (1 - alpha / 2)))
    return lo, hi
