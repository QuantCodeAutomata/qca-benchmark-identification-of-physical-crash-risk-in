"""
Experiment 1: Benchmark identification of physical crash-risk intervals.

Reproduces the paper's main benchmark experiment:
- SPX option cross-sections (synthetic data; replace with OptionMetrics for real results)
- CRRA gamma=2, crash threshold K=0.90, support bound B=5
- Full OTM quote set
- Horizons: 23, 30, 37 days
- Aggregates identified interval [q_lo, q_hi], [ell_lo, ell_hi], [s_lo, s_hi]
  by horizon (medians across valid cross-sections)

Note: Uses synthetic SPX data by default. Place OptionMetrics CSV at
data/optionmetrics_spx.csv to use real data.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", message="Solution may be inaccurate")

import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_spx_options
from src.pipeline import run_pipeline, aggregate_by_horizon

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Benchmark parameters (paper Table 1 / main benchmark)
GAMMA = 2.0
K = 0.90
B = 5.0
HORIZONS = [23, 30, 37]
QUOTE_SET = "otm"

# For fast prototyping: limit cross-sections. Set to None for full run.
MAX_CROSS_SECTIONS = 5  # Small for CI/testing; set None for full experiment


def run_benchmark(max_cs: int = MAX_CROSS_SECTIONS) -> pd.DataFrame:
    """Run the benchmark experiment and return results DataFrame."""
    logger.info("=== Experiment 1: Benchmark ===")
    logger.info("gamma=%.1f, K=%.2f, B=%.1f, quote_set=%s", GAMMA, K, B, QUOTE_SET)

    # Load data (synthetic fallback)
    df = load_spx_options(
        data_dir="data",
        use_synthetic=False,
        start_date="2013-01-01",
        end_date="2023-08-31",
        horizons=HORIZONS,
        n_puts=8,
        n_calls=5,
    )
    logger.info("Loaded %d quote rows across %d cross-sections.", len(df),
                df.groupby(["date", "horizon"]).ngroups)

    # Run pipeline
    results = run_pipeline(
        df=df,
        gamma=GAMMA,
        K=K,
        B=B,
        horizons=HORIZONS,
        quote_sets=[QUOTE_SET],
        max_cross_sections=max_cs,
        compute_frontier=False,
        verify_regularity=True,
        verbose=True,
    )

    logger.info("Solved %d cross-section-quote-set combinations.", len(results))
    valid = results["valid"].sum()
    logger.info("Valid solves: %d / %d", valid, len(results))

    return results


def make_summary_table(results: pd.DataFrame) -> pd.DataFrame:
    """Compute horizon-level median summary statistics."""
    summary = aggregate_by_horizon(results, quote_set=QUOTE_SET)
    return summary


def plot_time_series(results: pd.DataFrame) -> None:
    """Plot weekly q-interval time-series bands by horizon."""
    fig, axes = plt.subplots(len(HORIZONS), 1, figsize=(14, 4 * len(HORIZONS)), sharex=False)
    if len(HORIZONS) == 1:
        axes = [axes]

    valid_df = results[results["valid"] & (results["quote_set"] == QUOTE_SET)].copy()
    valid_df = valid_df.sort_values("date")

    for ax, h in zip(axes, HORIZONS):
        hdf = valid_df[valid_df["horizon"] == h]
        if hdf.empty:
            ax.set_title(f"{h}-day horizon (no valid data)")
            continue
        ax.fill_between(hdf["date"], hdf["q_lo_pct"], hdf["q_hi_pct"],
                        alpha=0.4, color="steelblue", label="q interval")
        ax.plot(hdf["date"], hdf["q_lo_pct"], color="steelblue", lw=0.8)
        ax.plot(hdf["date"], hdf["q_hi_pct"], color="steelblue", lw=0.8)
        ax.set_title(f"Benchmark q interval — {h}-day horizon (γ={GAMMA}, K={K}, B={B})")
        ax.set_ylabel("Crash probability q (%)")
        ax.set_xlabel("Date")
        ax.legend()

    plt.tight_layout()
    out_path = RESULTS_DIR / "exp1_benchmark_q_timeseries.png"
    plt.savefig(out_path, dpi=100)
    plt.close()
    logger.info("Saved: %s", out_path)


def main(max_cs: int = MAX_CROSS_SECTIONS) -> None:
    results = run_benchmark(max_cs)

    # Save raw results
    out_csv = RESULTS_DIR / "exp1_benchmark_results.csv"
    results.to_csv(out_csv, index=False)
    logger.info("Saved raw results: %s", out_csv)

    # Summary table
    summary = make_summary_table(results)
    logger.info("\n=== Horizon Median Summary (OTM, gamma=2, K=0.90, B=5) ===\n%s", summary.to_string())
    summary.to_csv(RESULTS_DIR / "exp1_benchmark_summary.csv")

    # Plot
    plot_time_series(results)

    # Regularity counts
    reg_df = results[results["quote_set"] == QUOTE_SET].copy()
    for h in HORIZONS:
        hdf = reg_df[reg_df["horizon"] == h]
        n_total = len(hdf)
        n_valid = hdf["valid"].sum()
        n_reg = hdf["regularity_verified"].sum()
        logger.info("Horizon %d: total=%d valid=%d regularity_verified=%d",
                    h, n_total, n_valid, n_reg)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_cs", type=int, default=MAX_CROSS_SECTIONS)
    args = ap.parse_args()
    main(args.max_cs)
