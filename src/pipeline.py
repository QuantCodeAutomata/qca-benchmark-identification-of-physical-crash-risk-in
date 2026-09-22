"""
Analysis pipeline: runs all cross-sections through quote normalization,
nested set construction, and SOCP solving.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

from src.data_loader import get_cross_section, list_cross_sections
from src.quote_processor import normalize_quotes, build_nested_quote_sets
from src.socp_solver import CrashRiskSOCP, CrossSectionResult

logger = logging.getLogger(__name__)


def run_pipeline(
    df: pd.DataFrame,
    gamma: float = 2.0,
    K: float = 0.90,
    B: float = 5.0,
    horizons: Optional[List[int]] = None,
    quote_sets: Optional[List[str]] = None,
    max_cross_sections: Optional[int] = None,
    compute_frontier: bool = False,
    verify_regularity: bool = True,
    solver: str = "CLARABEL",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the full identified-set pipeline for all cross-sections.

    For each (date, horizon) cross-section:
    1. Normalize quotes to call-payoff intervals.
    2. Build nested quote sets.
    3. Solve identified-set endpoint programs.

    Args:
        df: Full option dataset (from load_spx_options).
        gamma: CRRA risk-aversion parameter.
        K: Crash threshold.
        B: Right support bound.
        horizons: Horizons to process (default: all available).
        quote_sets: Quote sets to solve (default: ['otm']).
        max_cross_sections: Limit for testing (None = all).
        compute_frontier: Whether to compute joint frontiers.
        verify_regularity: Whether to verify Assumption 3.3.
        solver: CVXPY solver name.
        verbose: Whether to log progress.

    Returns:
        DataFrame of CrossSectionResult summary dicts, one row per
        (date, horizon, quote_set).
    """
    if horizons is None:
        horizons = sorted(df["horizon"].unique())
    if quote_sets is None:
        quote_sets = ["otm"]

    solver_obj = CrashRiskSOCP(
        gamma=gamma, K=K, B=B, solver=solver
    )

    cross_sections = list_cross_sections(df)
    cross_sections = cross_sections[cross_sections["horizon"].isin(horizons)]

    if max_cross_sections is not None:
        # Stratify by horizon so each horizon gets at least 1 cross-section
        n_per_horizon = max(1, max_cross_sections // len(horizons))
        cross_sections = (
            cross_sections.groupby("horizon", group_keys=False)
            .apply(lambda g: g.head(n_per_horizon))
            .reset_index(drop=True)
        )

    all_results = []
    n_total = len(cross_sections) * len(quote_sets)
    n_done = 0

    for _, cs_row in cross_sections.iterrows():
        date = cs_row["date"]
        horizon = int(cs_row["horizon"])

        cs_df = get_cross_section(df, date, horizon)
        cs_norm = normalize_quotes(cs_df)
        nested_sets = build_nested_quote_sets(cs_norm, K=K)

        for qs_name in quote_sets:
            qs_df = nested_sets.get(qs_name, pd.DataFrame())
            result = solver_obj.solve_cross_section(
                quotes=qs_df,
                date=date,
                horizon=horizon,
                quote_set=qs_name,
                compute_frontier=compute_frontier,
                verify_regularity=verify_regularity,
            )
            all_results.append(result.summary_dict())
            n_done += 1

            if verbose and n_done % 50 == 0:
                logger.info("Progress: %d / %d cross-sections solved.", n_done, n_total)

    if not all_results:
        return pd.DataFrame()

    return pd.DataFrame(all_results)


def aggregate_by_horizon(
    results: pd.DataFrame, quote_set: str = "otm"
) -> pd.DataFrame:
    """
    Compute horizon-level median summaries for valid cross-sections.

    Args:
        results: Output of run_pipeline.
        quote_set: Quote set to aggregate.

    Returns:
        DataFrame indexed by horizon with median endpoints and widths.
    """
    df = results[(results["quote_set"] == quote_set) & (results["valid"])].copy()
    if df.empty:
        return pd.DataFrame()

    agg_cols = [
        "q_lo_pct", "q_hi_pct", "q_width_pct",
        "ell_lo_pct", "ell_hi_pct", "ell_width_pct",
        "s_lo_pct", "s_hi_pct", "s_width_pct",
        "n_quotes",
    ]
    return df.groupby("horizon")[agg_cols].median().round(4)
