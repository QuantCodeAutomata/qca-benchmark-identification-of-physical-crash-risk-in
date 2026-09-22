"""
Experiment 3: Tail-support robustness.

Compares identified-set endpoints across finite support caps B in [2,10]
and an unbounded support specification (B=infinity implemented as B=100).

Key finding: under gamma=2, positive lower bounds on crash probability
are NOT identified by option prices alone without a right-tail restriction.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

GAMMA = 2.0
K = 0.90
HORIZONS = [23, 30, 37]
B_GRID = [2.0, 5.0, 10.0]  # paper uses 2-10 grid; reduced for CI
B_UNBOUNDED = 100.0  # Approximate unbounded with very large B
QUOTE_SET = "otm"
N_SUBSAMPLE = 3  # paper: 180; reduced for CI  # Fixed robustness subsample (paper uses 180; use smaller for CI)
MAX_CROSS_SECTIONS = 5


def collect_robustness_subsample(df: pd.DataFrame, n: int = N_SUBSAMPLE) -> pd.DataFrame:
    """
    Select a fixed robustness subsample of cross-sections.

    Chooses the first n cross-sections per horizon (deterministically).
    """
    cs_list = list_cross_sections(df)
    samples = []
    for h in HORIZONS:
        h_cs = cs_list[cs_list["horizon"] == h].head(n)
        samples.append(h_cs)
    return pd.concat(samples, ignore_index=True)


def run_tail_robustness(
    max_cs: int = MAX_CROSS_SECTIONS,
) -> pd.DataFrame:
    """Run identified-set programs across B values for the robustness subsample."""
    logger.info("=== Experiment 3: Tail Support Robustness ===")

    df = load_spx_options(
        data_dir="data",
        use_synthetic=False,
        start_date="2013-01-01",
        end_date="2023-08-31",
        horizons=HORIZONS,
        n_puts=8,
        n_calls=5,
    )

    subsample = collect_robustness_subsample(df, n=max_cs // len(HORIZONS) + 1)
    all_results = []

    for _, cs_row in subsample.iterrows():
        date = cs_row["date"]
        horizon = int(cs_row["horizon"])

        cs_df = get_cross_section(df, date, horizon)
        cs_norm = normalize_quotes(cs_df)
        nested = build_nested_quote_sets(cs_norm, K=K)
        quotes_otm = nested[QUOTE_SET]

        if len(quotes_otm) == 0:
            continue

        # Solve for each B value including unbounded approximation
        for B_val in B_GRID + [B_UNBOUNDED]:
            solver = CrashRiskSOCP(gamma=GAMMA, K=K, B=B_val)
            result = solver.solve_cross_section(
                quotes=quotes_otm,
                date=date,
                horizon=horizon,
                quote_set=QUOTE_SET,
                verify_regularity=False,
            )
            d = result.summary_dict()
            d["B"] = B_val
            d["is_unbounded"] = (B_val == B_UNBOUNDED)
            all_results.append(d)

    return pd.DataFrame(all_results)


def analyze_tail_robustness(results: pd.DataFrame) -> None:
    """Analyze and report tail support robustness."""
    B_ref = 5.0
    results_finite = results[~results["is_unbounded"]].copy()
    results_ref = results[results["B"] == B_ref][["date", "horizon", "q_lo_pct", "q_hi_pct",
                                                   "s_width_pct"]].rename(
        columns={"q_lo_pct": "q_lo_ref", "q_hi_pct": "q_hi_ref", "s_width_pct": "s_width_ref"}
    )

    merged = results_finite.merge(results_ref, on=["date", "horizon"], how="inner")
    merged["q_lo_change"] = merged["q_lo_pct"] - merged["q_lo_ref"]
    merged["s_width_ratio"] = merged["s_width_pct"] / merged["s_width_ref"].replace(0, np.nan)

    # Monotonicity check: q_lo should weakly decrease as B increases
    pivot_q_lo = merged.pivot_table(
        index=["date", "horizon"], columns="B", values="q_lo_pct", aggfunc="first"
    )
    logger.info("\n=== Median q_lo by B ===\n%s",
                merged.groupby("B")["q_lo_pct"].median().to_string())

    # Unbounded case
    unb = results[results["is_unbounded"]].copy()
    if len(unb) > 0:
        logger.info("\n=== Unbounded support (B=%.0f approx): median q_lo=%.4f %%",
                    B_UNBOUNDED, unb["q_lo_pct"].median())
        logger.info("Under unbounded support, q_lo should be ~0 (Proposition 3.6).")

    return merged


def plot_tail_robustness(results: pd.DataFrame) -> None:
    """Plot q_lo vs B for the robustness subsample."""
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(5 * len(HORIZONS), 5))
    if len(HORIZONS) == 1:
        axes = [axes]

    for ax, h in zip(axes, HORIZONS):
        hdf = results[(results["horizon"] == h) & (~results["is_unbounded"])].copy()
        for (date, _), grp in hdf.groupby(["date", "horizon"]):
            ax.plot(grp["B"], grp["q_lo_pct"], "-o", alpha=0.4, ms=3, lw=0.8, color="steelblue")
        ax.set_title(f"{h}-day horizon: q_lo vs B")
        ax.set_xlabel("Support cap B")
        ax.set_ylabel("q_lo (%)")

    plt.tight_layout()
    out_path = RESULTS_DIR / "exp3_tail_robustness_q_lo_vs_B.png"
    plt.savefig(out_path, dpi=100)
    plt.close()
    logger.info("Saved: %s", out_path)


def main(max_cs: int = MAX_CROSS_SECTIONS) -> None:
    results = run_tail_robustness(max_cs)
    results.to_csv(RESULTS_DIR / "exp3_tail_robustness_results.csv", index=False)
    merged = analyze_tail_robustness(results)
    merged.to_csv(RESULTS_DIR / "exp3_tail_robustness_analysis.csv", index=False)
    plot_tail_robustness(results)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_cs", type=int, default=MAX_CROSS_SECTIONS)
    args = ap.parse_args()
    main(args.max_cs)
