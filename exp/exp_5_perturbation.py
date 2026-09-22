"""
Experiment 5: Quote perturbation and quote-deletion robustness.

Tests sensitivity of identified intervals to:
1. Doubling bid/ask spreads (24-cross-section subset).
2. Retaining every other strike (24-cross-section subset).
3. Removing the put wing beyond local eight (180-cross-section subsample).
4. Removing all OTM calls (180-cross-section subsample).
5. Removing the six nearby puts while keeping every other quote.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", message="Solution may be inaccurate")

import logging
import sys
from pathlib import Path
from typing import Optional

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

# Subset sizes (paper: 24 and 180; use smaller for CI)
N_SUBSET_SMALL = 3   # paper: 24
N_SUBSET_LARGE = 6  # paper: 180


# ---------------------------------------------------------------------------
# Quote transformation functions
# ---------------------------------------------------------------------------

def double_spreads(quotes: pd.DataFrame) -> pd.DataFrame:
    """Double each bid/ask spread around its midpoint."""
    df = quotes.copy()
    midpoint = 0.5 * (df["c_lo"] + df["c_hi"])
    half_spread = 0.5 * (df["c_hi"] - df["c_lo"])
    df["c_lo"] = (midpoint - 2.0 * half_spread).clip(lower=0.0)
    df["c_hi"] = midpoint + 2.0 * half_spread
    # Ensure no crossings
    mask = df["c_lo"] > df["c_hi"]
    df.loc[mask, "c_lo"] = df.loc[mask, "c_hi"]
    return df


def thin_every_other_strike(quotes: pd.DataFrame) -> pd.DataFrame:
    """Retain every other strike by sorted strike level (deterministic parity=0)."""
    df = quotes.sort_values("k_norm").reset_index(drop=True)
    return df.iloc[::2].reset_index(drop=True)


def drop_far_put_wing(quotes: pd.DataFrame, K: float = 0.90, n_local: int = 8) -> pd.DataFrame:
    """Remove put-wing quotes beyond the local eight (keep local set + calls)."""
    puts = quotes[quotes["option_type"] == "P"].copy()
    calls = quotes[quotes["option_type"] == "C"].copy()
    otm_puts = puts[puts["k_norm"] < K].copy()
    itm_puts = puts[puts["k_norm"] >= K].copy()
    otm_puts_sorted = otm_puts.sort_values("k_norm", ascending=False)  # nearest first
    local_puts = otm_puts_sorted.head(n_local)
    return pd.concat([itm_puts, local_puts, calls], ignore_index=True)


def drop_otm_calls(quotes: pd.DataFrame, K: float = 0.90) -> pd.DataFrame:
    """Remove all OTM calls (k > 1, effectively strike > forward)."""
    return quotes[~((quotes["option_type"] == "C") & (quotes["k_norm"] > 1.0))].copy()


def drop_six_nearby_puts_keep_every_other(
    quotes: pd.DataFrame, K: float = 0.90
) -> pd.DataFrame:
    """
    Remove the six nearest puts (beyond the pair) and retain every other quote.

    Steps:
    1. Find pair = (one put just below K, one put just above K).
    2. Find six nearest additional puts by |k-K|.
    3. Remove those six from the full quote set.
    4. Retain every other remaining strike.
    """
    type_col = "option_type" if "option_type" in quotes.columns else "type"
    puts = quotes[quotes[type_col] == "P"].copy()
    calls = quotes[quotes[type_col] == "C"].copy()
    otm_puts = puts[puts["k_norm"] < 1.0].copy()
    otm_puts = otm_puts.sort_values("k_norm", ascending=False).reset_index(drop=True)

    # Pair: one just below K, one just above K (or nearest in OTM put region)
    below_K = otm_puts[otm_puts["k_norm"] < K].copy()
    above_K_puts = otm_puts[otm_puts["k_norm"] >= K].copy()

    pair_idx = set()
    if len(below_K) > 0:
        pair_idx.add(below_K.index[0])
    if len(above_K_puts) > 0:
        pair_idx.add(above_K_puts.index[-1])

    # Six nearest additional puts by |k-K|
    remaining = otm_puts.drop(index=list(pair_idx))
    remaining = remaining.copy()
    remaining["dist_K"] = (remaining["k_norm"] - K).abs()
    remaining = remaining.sort_values("dist_K")
    six_nearby = remaining.head(6)
    to_drop = pair_idx | set(six_nearby.index)

    # Keep all puts not in to_drop
    kept_puts = otm_puts.drop(index=list(to_drop))
    combined = pd.concat([kept_puts, calls], ignore_index=True)
    # Thin every other
    return thin_every_other_strike(combined)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def solve_quotes(
    quotes: pd.DataFrame,
    date: pd.Timestamp,
    horizon: int,
    quote_set_name: str,
) -> dict:
    """Solve one transformed quote set and return summary dict."""
    solver = CrashRiskSOCP(gamma=GAMMA, K=K, B=B)
    result = solver.solve_cross_section(
        quotes=quotes, date=date, horizon=horizon,
        quote_set=quote_set_name, verify_regularity=False,
    )
    return result.summary_dict()


def run_perturbation_experiment(
    n_small: int = N_SUBSET_SMALL,
    n_large: int = N_SUBSET_LARGE,
) -> pd.DataFrame:
    """Run all perturbation/deletion experiments and collect results."""
    logger.info("=== Experiment 5: Quote Perturbation ===")

    df = load_spx_options(
        data_dir="data", use_synthetic=False,
        start_date="2013-01-01", end_date="2023-08-31",
        horizons=HORIZONS, n_puts=8, n_calls=5,
    )

    cs_list = list_cross_sections(df)
    cs_list = cs_list[cs_list["horizon"].isin(HORIZONS)].copy()

    all_results = []

    # --- Small subset: spread-doubling and strike-thinning ---
    small_cs = stratified_head(cs_list, n_small)
    logger.info("Running spread/thinning perturbations on %d cross-sections.", len(small_cs))

    for _, cs_row in small_cs.iterrows():
        date = cs_row["date"]
        horizon = int(cs_row["horizon"])
        cs_df = get_cross_section(df, date, horizon)
        cs_norm = normalize_quotes(cs_df)
        nested = build_nested_quote_sets(cs_norm, K=K)
        quotes_otm = nested["otm"]
        if len(quotes_otm) == 0:
            continue

        # Baseline
        base = solve_quotes(quotes_otm, date, horizon, "baseline")
        base["perturbation"] = "baseline"
        base["subset"] = "small"
        all_results.append(base)

        # Spread doubling
        q_doubled = double_spreads(quotes_otm)
        perturbed = solve_quotes(q_doubled, date, horizon, "doubled_spread")
        perturbed["perturbation"] = "doubled_spread"
        perturbed["subset"] = "small"
        all_results.append(perturbed)

        # Strike thinning
        q_thinned = thin_every_other_strike(quotes_otm)
        thinned = solve_quotes(q_thinned, date, horizon, "every_other_strike")
        thinned["perturbation"] = "every_other_strike"
        thinned["subset"] = "small"
        all_results.append(thinned)

    # --- Large subset: quote-layer deletion ---
    large_cs = stratified_head(cs_list, n_large)
    logger.info("Running quote-deletion experiments on %d cross-sections.", len(large_cs))

    for _, cs_row in large_cs.iterrows():
        date = cs_row["date"]
        horizon = int(cs_row["horizon"])
        cs_df = get_cross_section(df, date, horizon)
        cs_norm = normalize_quotes(cs_df)
        nested = build_nested_quote_sets(cs_norm, K=K)
        quotes_otm = nested["otm"]
        if len(quotes_otm) == 0:
            continue

        # Baseline (OTM)
        base = solve_quotes(quotes_otm, date, horizon, "baseline")
        base["perturbation"] = "baseline"
        base["subset"] = "large"
        all_results.append(base)

        # A: Remove far put wing (use local 8)
        q_local = nested["local"]
        res_a = solve_quotes(q_local, date, horizon, "drop_far_wing")
        res_a["perturbation"] = "drop_far_wing"
        res_a["subset"] = "large"
        all_results.append(res_a)

        # B: Remove OTM calls (use wing only)
        q_wing = nested["wing"]
        res_b = solve_quotes(q_wing, date, horizon, "drop_otm_calls")
        res_b["perturbation"] = "drop_otm_calls"
        res_b["subset"] = "large"
        all_results.append(res_b)

        # C: Remove six nearby puts, keep every other quote
        q_c = drop_six_nearby_puts_keep_every_other(quotes_otm, K=K)
        res_c = solve_quotes(q_c, date, horizon, "drop_6nearby_thin")
        res_c["perturbation"] = "drop_6nearby_thin"
        res_c["subset"] = "large"
        all_results.append(res_c)

    return pd.DataFrame(all_results)


def analyze_perturbations(results: pd.DataFrame) -> None:
    """Compute median absolute changes relative to baseline."""
    for subset_label, subset_df in results.groupby("subset"):
        baseline = subset_df[subset_df["perturbation"] == "baseline"][
            ["date", "horizon", "q_lo_pct", "q_hi_pct", "s_lo_pct", "s_hi_pct"]
        ].rename(columns={"q_lo_pct": "q_lo_base", "q_hi_pct": "q_hi_base",
                           "s_lo_pct": "s_lo_base", "s_hi_pct": "s_hi_base"})

        for perturb in subset_df["perturbation"].unique():
            if perturb == "baseline":
                continue
            pert_df = subset_df[subset_df["perturbation"] == perturb].merge(
                baseline, on=["date", "horizon"], how="inner"
            )
            if pert_df.empty:
                continue
            q_change = (pert_df["q_lo_pct"] - pert_df["q_lo_base"]).abs()
            s_change = (pert_df["s_lo_pct"] - pert_df["s_lo_base"]).abs()
            logger.info(
                "[%s] %s: median |Δq_lo|=%.4f%%, median |Δs_lo|=%.4f%%",
                subset_label, perturb,
                q_change.median(), s_change.median(),
            )


def main(max_cs: int = N_SUBSET_LARGE) -> None:
    n_large = max_cs
    n_small = max(1, max_cs // 2)
    results = run_perturbation_experiment(n_small=n_small, n_large=n_large)
    results.to_csv(RESULTS_DIR / "exp5_perturbation_results.csv", index=False)
    analyze_perturbations(results)
    logger.info("Experiment 5 complete. Results saved.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_cs", type=int, default=N_SUBSET_LARGE)
    args = ap.parse_args()
    main(args.max_cs)
