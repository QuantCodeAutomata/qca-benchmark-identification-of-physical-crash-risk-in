"""
Experiment 4: Kernel and threshold robustness.

Part A: Compare gamma=1 vs gamma=2 (log-investor vs CRRA).
Part B: Compare crash threshold K=0.85 vs K=0.90.

Both use full OTM quote set, B=5, SPX Wednesday cross-sections.
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
from src.quote_processor import normalize_quotes, build_nested_quote_sets
from src.socp_solver import CrashRiskSOCP
from src.utils import stratified_head

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

HORIZONS = [23, 30, 37]
B = 5.0
QUOTE_SET = "otm"
MAX_CROSS_SECTIONS = 5


def run_kernel_comparison(max_cs: int = MAX_CROSS_SECTIONS) -> pd.DataFrame:
    """
    Part A: Compare gamma=1 vs gamma=2 on common solved sample.
    """
    logger.info("=== Experiment 4A: Kernel Comparison (gamma=1 vs gamma=2) ===")

    df = load_spx_options(
        data_dir="data", use_synthetic=False,
        start_date="2013-01-01", end_date="2023-08-31",
        horizons=HORIZONS, n_puts=8, n_calls=5,
    )

    cs_list = list_cross_sections(df)
    cs_list = cs_list[cs_list["horizon"].isin(HORIZONS)]
    if max_cs:
        cs_list = stratified_head(cs_list, max_cs)

    all_results = []
    for _, cs_row in cs_list.iterrows():
        date = cs_row["date"]
        horizon = int(cs_row["horizon"])
        cs_df = get_cross_section(df, date, horizon)
        cs_norm = normalize_quotes(cs_df)
        nested = build_nested_quote_sets(cs_norm, K=0.90)
        quotes_otm = nested[QUOTE_SET]

        if len(quotes_otm) == 0:
            continue

        for gamma in [1.0, 2.0]:
            solver = CrashRiskSOCP(gamma=gamma, K=0.90, B=B)
            result = solver.solve_cross_section(
                quotes=quotes_otm, date=date, horizon=horizon,
                quote_set=QUOTE_SET, verify_regularity=False,
            )
            d = result.summary_dict()
            d["gamma"] = gamma
            all_results.append(d)

    return pd.DataFrame(all_results)


def run_threshold_comparison(max_cs: int = MAX_CROSS_SECTIONS) -> pd.DataFrame:
    """
    Part B: Compare K=0.85 vs K=0.90 with gamma=2, B=5.
    """
    logger.info("=== Experiment 4B: Threshold Comparison (K=0.85 vs K=0.90) ===")

    df = load_spx_options(
        data_dir="data", use_synthetic=False,
        start_date="2013-01-01", end_date="2023-08-31",
        horizons=HORIZONS, n_puts=8, n_calls=5,
    )

    cs_list = list_cross_sections(df)
    cs_list = cs_list[cs_list["horizon"].isin(HORIZONS)]
    if max_cs:
        cs_list = stratified_head(cs_list, max_cs)

    all_results = []
    for _, cs_row in cs_list.iterrows():
        date = cs_row["date"]
        horizon = int(cs_row["horizon"])
        cs_df = get_cross_section(df, date, horizon)
        cs_norm = normalize_quotes(cs_df)

        for K_val in [0.85, 0.90]:
            nested = build_nested_quote_sets(cs_norm, K=K_val)
            quotes_otm = nested[QUOTE_SET]

            if len(quotes_otm) == 0:
                continue

            solver = CrashRiskSOCP(gamma=2.0, K=K_val, B=B)
            result = solver.solve_cross_section(
                quotes=quotes_otm, date=date, horizon=horizon,
                quote_set=QUOTE_SET, verify_regularity=False,
            )
            d = result.summary_dict()
            d["K"] = K_val
            all_results.append(d)

    return pd.DataFrame(all_results)


def summarize_kernel_comparison(results: pd.DataFrame) -> pd.DataFrame:
    """Compute median widths by gamma and horizon."""
    valid = results[results["valid"]].copy()
    return valid.groupby(["horizon", "gamma"])[
        ["q_width_pct", "ell_width_pct", "s_width_pct"]
    ].median().round(4)


def summarize_threshold_comparison(results: pd.DataFrame) -> pd.DataFrame:
    """Compute median endpoints and widths by K and horizon."""
    valid = results[results["valid"]].copy()
    return valid.groupby(["horizon", "K"])[
        ["q_lo_pct", "q_hi_pct", "q_width_pct", "s_width_pct"]
    ].median().round(4)


def plot_kernel_comparison(results: pd.DataFrame) -> None:
    """Plot q-width comparison gamma=1 vs gamma=2."""
    valid = results[results["valid"]].copy()
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(5 * len(HORIZONS), 5))
    if len(HORIZONS) == 1:
        axes = [axes]

    for ax, h in zip(axes, HORIZONS):
        hdf = valid[valid["horizon"] == h]
        for gamma, color in [(1.0, "orange"), (2.0, "steelblue")]:
            gdf = hdf[hdf["gamma"] == gamma]["q_width_pct"].dropna()
            ax.hist(gdf, bins=20, alpha=0.5, color=color, label=f"γ={gamma}")
        ax.set_title(f"q-width histogram — {h}-day")
        ax.set_xlabel("q width (%)")
        ax.legend()

    plt.tight_layout()
    out_path = RESULTS_DIR / "exp4a_kernel_comparison.png"
    plt.savefig(out_path, dpi=100)
    plt.close()
    logger.info("Saved: %s", out_path)


def main(max_cs: int = MAX_CROSS_SECTIONS) -> None:
    # Part A: Kernel
    kernel_results = run_kernel_comparison(max_cs)
    kernel_results.to_csv(RESULTS_DIR / "exp4a_kernel_comparison.csv", index=False)
    summary_kernel = summarize_kernel_comparison(kernel_results)
    logger.info("\n=== Kernel Comparison (median widths) ===\n%s", summary_kernel.to_string())
    summary_kernel.to_csv(RESULTS_DIR / "exp4a_kernel_summary.csv")
    plot_kernel_comparison(kernel_results)

    # Part B: Threshold
    thresh_results = run_threshold_comparison(max_cs)
    thresh_results.to_csv(RESULTS_DIR / "exp4b_threshold_comparison.csv", index=False)
    summary_thresh = summarize_threshold_comparison(thresh_results)
    logger.info("\n=== Threshold Comparison (median endpoints) ===\n%s", summary_thresh.to_string())
    summary_thresh.to_csv(RESULTS_DIR / "exp4b_threshold_summary.csv")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_cs", type=int, default=MAX_CROSS_SECTIONS)
    args = ap.parse_args()
    main(args.max_cs)
