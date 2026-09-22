"""
Experiment 7: Regularity verification and numerical exactness audit.

Verifies Assumption 3.3 (regularity) for each cross-section:
- Tests existence of strictly interior witness measure.
- Records maximum achievable interior slack.
- Audits solver optimality status for each conic program.
- Produces regularity table by horizon.
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

GAMMA = 2.0
K = 0.90
B = 5.0
HORIZONS = [23, 30, 37]
QUOTE_SET = "otm"
MAX_CROSS_SECTIONS = 5


def run_regularity_audit(max_cs: int = MAX_CROSS_SECTIONS) -> pd.DataFrame:
    """
    Run regularity verification and solver audit for each cross-section.
    """
    logger.info("=== Experiment 7: Regularity Verification Audit ===")

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
    solver = CrashRiskSOCP(gamma=GAMMA, K=K, B=B)

    for _, cs_row in cs_list.iterrows():
        date = cs_row["date"]
        horizon = int(cs_row["horizon"])
        cs_df = get_cross_section(df, date, horizon)
        cs_norm = normalize_quotes(cs_df)
        nested = build_nested_quote_sets(cs_norm, K=K)
        quotes_otm = nested[QUOTE_SET]

        if len(quotes_otm) == 0:
            all_results.append({
                "date": date, "horizon": horizon, "n_quotes": 0,
                "valid": False, "regularity_verified": False,
                "regularity_slack": None, "status_q_lo": "no_quotes",
                "status_q_hi": "no_quotes",
            })
            continue

        result = solver.solve_cross_section(
            quotes=quotes_otm, date=date, horizon=horizon,
            quote_set=QUOTE_SET, verify_regularity=True,
        )
        all_results.append(result.summary_dict())

    return pd.DataFrame(all_results)


def produce_regularity_table(results: pd.DataFrame) -> pd.DataFrame:
    """Produce regularity verification table by horizon."""
    table_rows = []
    for h in HORIZONS:
        hdf = results[results["horizon"] == h]
        n_total = len(hdf)
        n_valid = hdf["valid"].sum()
        n_reg = hdf["regularity_verified"].sum()
        slacks = hdf[hdf["regularity_slack"].notna()]["regularity_slack"]
        med_slack = slacks.median() if len(slacks) > 0 else np.nan
        table_rows.append({
            "horizon": h,
            "n_total": n_total,
            "n_valid": int(n_valid),
            "n_regularity_verified": int(n_reg),
            "median_slack": med_slack,
            "pct_regularity_verified": 100.0 * n_reg / n_total if n_total > 0 else 0.0,
        })
    return pd.DataFrame(table_rows).set_index("horizon")


def identify_failed_dates(results: pd.DataFrame) -> pd.DataFrame:
    """List cross-sections where validity or regularity fails."""
    failed = results[~results["valid"] | ~results["regularity_verified"]].copy()
    return failed[["date", "horizon", "valid", "regularity_verified",
                   "regularity_slack", "status_q_lo", "status_q_hi"]]


def plot_regularity_slack(results: pd.DataFrame) -> None:
    """Plot histogram of regularity slack by horizon."""
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(5 * len(HORIZONS), 4))
    if len(HORIZONS) == 1:
        axes = [axes]

    for ax, h in zip(axes, HORIZONS):
        hdf = results[(results["horizon"] == h) & results["regularity_slack"].notna()]
        slacks = hdf["regularity_slack"].dropna()
        if len(slacks) == 0:
            ax.set_title(f"{h}-day (no data)")
            continue
        ax.hist(slacks, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(slacks.median(), color="red", lw=1.5, label=f"median={slacks.median():.2e}")
        ax.set_title(f"Regularity slack — {h}-day horizon")
        ax.set_xlabel("Max interior slack ε")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = RESULTS_DIR / "exp7_regularity_slack.png"
    plt.savefig(out_path, dpi=100)
    plt.close()
    logger.info("Saved: %s", out_path)


def main(max_cs: int = MAX_CROSS_SECTIONS) -> None:
    results = run_regularity_audit(max_cs)
    results.to_csv(RESULTS_DIR / "exp7_regularity_results.csv", index=False)

    reg_table = produce_regularity_table(results)
    logger.info("\n=== Regularity Verification Table ===\n%s", reg_table.to_string())
    reg_table.to_csv(RESULTS_DIR / "exp7_regularity_table.csv")

    failed_dates = identify_failed_dates(results)
    if len(failed_dates) > 0:
        logger.info("\n=== Failed / Unverified Cross-Sections ===\n%s",
                    failed_dates.to_string())
        failed_dates.to_csv(RESULTS_DIR / "exp7_failed_dates.csv", index=False)

    plot_regularity_slack(results)

    logger.info(
        "Expected: ~405/406, 374/376, 332/333 regularity-verified for 23/30/37-day "
        "(paper benchmark on full OptionMetrics sample)."
    )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_cs", type=int, default=MAX_CROSS_SECTIONS)
    args = ap.parse_args()
    main(args.max_cs)
