"""
Utility functions for crash-risk identification.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import List, Optional, Tuple


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safe division returning default when denominator is near zero."""
    if abs(b) < 1e-15:
        return default
    return a / b


def psd_2x2_constraints(
    u: cp.Expression,
    v: cp.Expression,
    w: cp.Expression,
) -> List[cp.Constraint]:
    """
    Return cvxpy PSD constraints for the 2x2 matrix [[u,v],[v,w]] >> 0.

    Uses the SOC equivalence: u>=0, w>=0, ||(2v, u-w)||_2 <= u+w.

    Args:
        u, v, w: cvxpy scalar expressions forming the symmetric matrix.

    Returns:
        List of cvxpy constraints.
    """
    return [
        u >= 0,
        w >= 0,
        cp.norm(cp.hstack([2 * v, u - w]), 2) <= u + w,
    ]


def bmat_2x2_psd(
    u: cp.Expression,
    v: cp.Expression,
    w: cp.Expression,
) -> cp.Constraint:
    """Return a single 2x2 PSD constraint [[u,v],[v,w]] >> 0 via cp.bmat."""
    M = cp.bmat(
        [
            [cp.reshape(u, (1, 1)), cp.reshape(v, (1, 1))],
            [cp.reshape(v, (1, 1)), cp.reshape(w, (1, 1))],
        ]
    )
    return M >> 0


def get_wednesday_dates(start: str, end: str) -> pd.DatetimeIndex:
    """Return all Wednesdays in [start, end]."""
    return pd.date_range(start=start, end=end, freq="W-WED")


def is_optimal(status: str) -> bool:
    """Check if solver status indicates a successful solve."""
    return status in ("optimal", "optimal_inaccurate")


def stratified_head(df: pd.DataFrame, max_total: int, horizon_col: str = "horizon") -> pd.DataFrame:
    """Take at most max_total rows stratified equally across horizon values.

    Parameters
    ----------
    df:
        DataFrame with a horizon_col column.
    max_total:
        Maximum total rows to return.
    horizon_col:
        Column used for stratification.

    Returns
    -------
    Stratified subsample with at most max_total rows.
    """
    horizons = df[horizon_col].unique()
    n_per = max(1, max_total // len(horizons))
    return (
        df.groupby(horizon_col, group_keys=False)
        .apply(lambda g: g.head(n_per))
        .reset_index(drop=True)
    )


def percentile_ci(
    data: np.ndarray, alpha: float = 0.05
) -> Tuple[float, float]:
    """Compute (alpha/2, 1-alpha/2) percentile confidence interval."""
    lo = np.percentile(data, 100 * alpha / 2)
    hi = np.percentile(data, 100 * (1 - alpha / 2))
    return lo, hi
