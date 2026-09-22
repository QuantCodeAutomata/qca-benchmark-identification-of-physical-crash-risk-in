"""
Experiment 2: Quote-density identification experiment.

Compares identified-set widths across four nested quote sets:
pair, local (8 puts), wing (all OTM puts), and all OTM (wing + calls).

Reports layer-by-layer proportional width reductions in q-interval.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", message="Solution may be inaccurate")

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_spx_options, get_cross_section, list_cross_sections
from src.quote_processor import normalize_quotes, build_nested_quote_sets, get_quote_set_summary
from src.socp_solver import CrashRiskSOCP
from src.pipeline import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

GAMMA = 2.0
K = 0.90
B = 5.0
HORIZONS = [23, 30, 37]
QUOTE_SETS = ["pair", "local", "wing", "otm"]
MAX_CROSS_SECTIONS = 5  # Increase for full run


def run_quote_density(max_cs: int = MAX_CROSS_SECTIONS) -> pd.DataFrame:
    """Run identified-set programs for all four nested quote sets."""
    logger.info("=== Experiment 2: Quote Density ===")

    df = load_spx_options(
        data_dir="data",
        use_synthetic=False,
        start_date="2013-01-01",
        end_date="2023-08-31",
        horizons=HORIZONS,
        n_puts=8,
        n_calls=5,
    )

    results = run_pipeline(
        df=df,
        gamma=GAMMA,
        K=K,
        B=B,
        horizons=HORIZONS,
        quote_sets=QUOTE_SETS,
        max_cross_sections=max_cs,
        compute_frontier=False,
        verify_regularity=False,
        verbose=True,
    )

    return results


def compute_width_reductions(results: pd.DataFrame) -> pd.DataFrame:
    """
    Compute layer-by-layer q-width reductions.

    For each cross-section, compute proportional reduction:
    delta_{A->B} = 1 - w_B / w_A

    Returns:
        DataFrame with reductions per (date, horizon).
    """
    pivot = results[results["valid"]].pivot_table(
        index=["date", "horizon"],
        columns="quote_set",
        values="q_width_pct",
        aggfunc="first",
    )
    # Only include rows where all four sets are available
    pivot = pivot.dropna(subset=QUOTE_SETS)

    reduction_cols = {}
    for qs in QUOTE_SETS:
        reduction_cols[f"w_{qs}"] = pivot[qs]

    # Layer reductions
    reduction_cols["delta_pair_to_local"] = 1.0 - pivot["local"] / pivot["pair"].replace(0, np.nan)
    reduction_cols["delta_local_to_wing"] = 1.0 - pivot["wing"] / pivot["local"].replace(0, np.nan)
    reduction_cols["delta_wing_to_otm"] = 1.0 - pivot["otm"] / pivot["wing"].replace(0, np.nan)
    reduction_cols["delta_pair_to_otm"] = 1.0 - pivot["otm"] / pivot["pair"].replace(0, np.nan)

    return pd.DataFrame(reduction_cols).reset_index()


def horizon_summary(reductions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate reductions by horizon (medians)."""
    cols = [c for c in reductions.columns if c not in ("date", "horizon")]
    return reductions.groupby("horizon")[cols].median().round(4)


def plot_width_comparison(results: pd.DataFrame) -> None:
    """Plot boxplot of q-widths by quote set for each horizon."""
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(5 * len(HORIZONS), 5))
    if len(HORIZONS) == 1:
        axes = [axes]

    valid_df = results[results["valid"]].copy()

    for ax, h in zip(axes, HORIZONS):
        hdf = valid_df[valid_df["horizon"] == h]
        data_by_set = [
            hdf[hdf["quote_set"] == qs]["q_width_pct"].dropna().values
            for qs in QUOTE_SETS
        ]
        ax.boxplot(data_by_set, tick_labels=QUOTE_SETS)
        ax.set_title(f"q-width by quote set — {h}-day horizon")
        ax.set_ylabel("q-interval width (%)")
        ax.set_xlabel("Quote set")

    plt.tight_layout()
    out_path = RESULTS_DIR / "exp2_quote_density_widths.png"
    plt.savefig(out_path, dpi=100)
    plt.close()
    logger.info("Saved: %s", out_path)


def main(max_cs: int = MAX_CROSS_SECTIONS) -> None:
    results = run_quote_density(max_cs)

    out_csv = RESULTS_DIR / "exp2_quote_density_results.csv"
    results.to_csv(out_csv, index=False)

    reductions = compute_width_reductions(results)
    summary = horizon_summary(reductions)
    logger.info("\n=== Exp2 Width Reduction Summary ===\n%s", summary.to_string())
    summary.to_csv(RESULTS_DIR / "exp2_width_reductions.csv")

    # Median widths by quote set and horizon
    valid_df = results[results["valid"]].copy()
    width_summary = valid_df.groupby(["horizon", "quote_set"])["q_width_pct"].median().unstack()
    logger.info("\n=== Median q-widths by horizon and quote set ===\n%s", width_summary.to_string())
    width_summary.to_csv(RESULTS_DIR / "exp2_median_widths.csv")

    plot_width_comparison(results)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_cs", type=int, default=MAX_CROSS_SECTIONS)
    args = ap.parse_args()
    main(args.max_cs)
