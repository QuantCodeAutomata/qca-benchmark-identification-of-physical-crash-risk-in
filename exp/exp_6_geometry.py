"""
Experiment 6: Joint-versus-marginal geometry and directional support-gap analysis.

Compares the joint identified set Theta_{q,ell} with the marginal rectangle
formed by separate q and ell intervals. Measures normalized support gaps
and directional contractions across nested quote layers.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", message="Solution may be inaccurate")

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_spx_options, get_cross_section, list_cross_sections
from src.quote_processor import normalize_quotes, build_nested_quote_sets
from src.socp_solver import CrashRiskSOCP
from src.frontier import (
    compute_support_function,
    compute_diagonal_support_gaps,
    put_pair_scale,
)
from src.utils import stratified_head

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

GAMMA = 2.0
K = 0.90
B = 5.0
HORIZONS = [23, 30, 37]
QUOTE_SETS = ["pair", "local", "wing", "otm"]
N_FRONTIER_DIRS = 32
MAX_CROSS_SECTIONS = 3  # Small for CI; paper uses 976 common cross-sections


def run_geometry_experiment(max_cs: int = MAX_CROSS_SECTIONS) -> pd.DataFrame:
    """
    Compute joint frontiers and support gaps for selected cross-sections.
    """
    logger.info("=== Experiment 6: Geometry Analysis ===")

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
    frontier_data = []  # Store for selected frontier plots

    for i, (_, cs_row) in enumerate(cs_list.iterrows()):
        date = cs_row["date"]
        horizon = int(cs_row["horizon"])
        cs_df = get_cross_section(df, date, horizon)
        cs_norm = normalize_quotes(cs_df)
        nested = build_nested_quote_sets(cs_norm, K=K)

        cs_data = {"date": date, "horizon": horizon}
        qs_results = {}

        for qs_name in QUOTE_SETS:
            quotes = nested.get(qs_name, pd.DataFrame())
            if len(quotes) == 0:
                continue

            solver = CrashRiskSOCP(gamma=GAMMA, K=K, B=B)
            result = solver.solve_cross_section(
                quotes=quotes, date=date, horizon=horizon,
                quote_set=qs_name, compute_frontier=True,
                n_frontier_dirs=N_FRONTIER_DIRS, verify_regularity=False,
            )
            qs_results[qs_name] = result
            d = result.summary_dict()
            d["quote_set"] = qs_name
            all_results.append(d)

        # Compute support gaps for cross-sections that have all four sets
        if all(qs in qs_results for qs in QUOTE_SETS):
            pair_res = qs_results["pair"]
            scale = put_pair_scale(
                pair_res.q_lo or 0.0, pair_res.q_hi or 0.0,
                pair_res.ell_lo or 0.0, pair_res.ell_hi or 0.0,
            )

            for qs_name, result in qs_results.items():
                if result.valid:
                    marginal_rect = (
                        result.q_lo or 0.0, result.q_hi or 0.0,
                        result.ell_lo or 0.0, result.ell_hi or 0.0,
                    )
                    gaps = compute_diagonal_support_gaps(
                        result.frontier_points, marginal_rect, scale
                    )
                    for direction, gap in gaps.items():
                        all_results[-1][f"support_gap_{direction}"] = gap

        # Save frontier for plotting (only first cross-section of each horizon)
        if i < len(HORIZONS):
            frontier_data.append({"date": date, "horizon": horizon, "qs_results": qs_results})

    results_df = pd.DataFrame(all_results)

    # Plot frontiers for selected cross-sections
    _plot_frontiers(frontier_data)

    return results_df


def _plot_frontiers(frontier_data: list) -> None:
    """Plot nested q-ell frontiers for selected cross-sections."""
    n_plots = len(frontier_data)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]

    colors = {"pair": "red", "local": "orange", "wing": "steelblue", "otm": "darkblue"}

    for ax, fd in zip(axes, frontier_data):
        date = fd["date"]
        horizon = fd["horizon"]
        qs_results = fd["qs_results"]

        for qs_name in QUOTE_SETS:
            if qs_name not in qs_results:
                continue
            result = qs_results[qs_name]
            pts = result.frontier_points
            if pts is None or np.all(np.isnan(pts)):
                continue
            valid_pts = pts[~np.any(np.isnan(pts), axis=1)]
            if len(valid_pts) < 2:
                continue
            # Scale to percentage
            ax.scatter(
                valid_pts[:, 0] * 100,
                valid_pts[:, 1] * 100,
                s=8, color=colors.get(qs_name, "gray"),
                label=qs_name, alpha=0.7,
            )

        ax.set_xlabel("Crash prob q (%)")
        ax.set_ylabel("Shortfall ℓ (%)")
        ax.set_title(f"Joint frontier\n{date.date()} {horizon}-day")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = RESULTS_DIR / "exp6_joint_frontiers.png"
    plt.savefig(out_path, dpi=100)
    plt.close()
    logger.info("Saved: %s", out_path)


def summarize_support_gaps(results: pd.DataFrame) -> pd.DataFrame:
    """Compute median support gaps across cross-sections."""
    gap_cols = [c for c in results.columns if c.startswith("support_gap_")]
    if not gap_cols:
        return pd.DataFrame()
    return results.groupby("quote_set")[gap_cols].median().round(4)


def main(max_cs: int = MAX_CROSS_SECTIONS) -> None:
    results = run_geometry_experiment(max_cs)
    results.to_csv(RESULTS_DIR / "exp6_geometry_results.csv", index=False)

    gap_summary = summarize_support_gaps(results)
    if not gap_summary.empty:
        logger.info("\n=== Median Normalized Support Gaps ===\n%s", gap_summary.to_string())
        gap_summary.to_csv(RESULTS_DIR / "exp6_support_gaps.csv")

    logger.info("Experiment 6 complete.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_cs", type=int, default=MAX_CROSS_SECTIONS)
    args = ap.parse_args()
    main(args.max_cs)
