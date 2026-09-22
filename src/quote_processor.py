"""
Quote normalization and nested quote set construction for crash-risk identification.

Converts raw option quotes to normalized call-payoff bid/ask intervals and
builds the four nested quote sets (pair, local, wing, otm) for each cross-section.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# OTM convention: puts with k_norm < 1.0, calls with k_norm > 1.0 (relative to forward)
OTM_PUT_MONEYNESS_THRESHOLD = 1.0
OTM_CALL_MONEYNESS_THRESHOLD = 1.0


def normalize_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize strikes and convert all quotes to call-payoff bid/ask intervals.

    Normalization steps:
    1. Normalized strike: k_i = kappa_i / F
    2. For calls: c_lo = bid / (D*F), c_hi = ask / (D*F)  [already normalized]
    3. For puts: convert via put-call parity:
       c_put = p / (D*F) + 1 - k_i
       so c_lo = bid/(D*F) + 1 - k_i, c_hi = ask/(D*F) + 1 - k_i

    OTM convention (forward moneyness):
    - OTM put: k_norm < 1.0 (strike below forward)
    - OTM call: k_norm > 1.0 (strike above forward)

    Args:
        df: Raw cross-section DataFrame with columns:
            type (C/P), strike, bid, ask, D, F, horizon, date, maturity_date.

    Returns:
        DataFrame with added columns: k_norm, c_lo, c_hi, is_otm.
    """
    df = df.copy()
    DF = df["D"] * df["F"]  # D * F = discount factor * forward

    # Normalized strike
    df["k_norm"] = df["strike"] / df["F"]

    # Normalized prices (divide by D*F so that E_Q[(X-k)+] is expressed in forward units)
    df["c_lo"] = np.where(
        df["type"] == "C",
        df["bid"] / DF,
        df["bid"] / DF + 1.0 - df["k_norm"],
    )
    df["c_hi"] = np.where(
        df["type"] == "C",
        df["ask"] / DF,
        df["ask"] / DF + 1.0 - df["k_norm"],
    )

    # OTM flag: puts below forward, calls above forward
    df["is_otm"] = np.where(
        df["type"] == "P",
        df["k_norm"] < OTM_PUT_MONEYNESS_THRESHOLD,
        df["k_norm"] > OTM_CALL_MONEYNESS_THRESHOLD,
    )

    # Clip lower bound to 0 (call payoffs are non-negative)
    df["c_lo"] = np.maximum(0.0, df["c_lo"])

    # Drop quotes where c_lo > c_hi (should be rare after clipping)
    n_before = len(df)
    df = df[df["c_lo"] <= df["c_hi"]].copy()
    if len(df) < n_before:
        logger.debug("normalize_quotes: dropped %d crossed normalized quotes.", n_before - len(df))

    return df


def build_nested_quote_sets(
    df: pd.DataFrame,
    K: float = 0.90,
) -> Dict[str, pd.DataFrame]:
    """
    Build four nested OTM quote sets for a single normalized cross-section.

    The cross-section df must already be normalized (has k_norm, c_lo, c_hi).

    Definitions (all using OTM put/call convention relative to forward):
    - pair: the one retained put strike just below K and one just above K
            (two puts nearest the crash threshold K on each side)
    - local: pair + the 6 nearest additional retained put strikes by |k_i - K|
             (8 puts total nearest the threshold)
    - wing: all retained OTM puts (k_norm < 1.0, type=P)
    - otm: wing + all retained OTM calls (k_norm > 1.0, type=C)

    Tie-breaking for nearest-to-K selection:
    1. Sort by |k_i - K| ascending
    2. Then by k_i ascending (lower strike first)
    3. Then by original row index ascending

    Args:
        df: Normalized cross-section with k_norm, c_lo, c_hi, type.
        K: Crash threshold (default 0.90).

    Returns:
        Dict with keys 'pair', 'local', 'wing', 'otm', each a DataFrame
        of normalized quotes for that quote set.
    """
    # Retain only OTM puts and OTM calls
    otm_puts = df[(df["type"] == "P") & (df["k_norm"] < OTM_PUT_MONEYNESS_THRESHOLD)].copy()
    otm_calls = df[(df["type"] == "C") & (df["k_norm"] > OTM_CALL_MONEYNESS_THRESHOLD)].copy()

    # Sort puts by distance to K (primary), then by k_norm (secondary), then by index
    otm_puts = otm_puts.reset_index(drop=False)  # preserve original index
    otm_puts["dist_to_K"] = np.abs(otm_puts["k_norm"] - K)
    otm_puts = otm_puts.sort_values(
        ["dist_to_K", "k_norm", "index"], ascending=[True, True, True]
    ).reset_index(drop=True)

    # --- Pair: closest put below K and closest put above K ---
    below_K = otm_puts[otm_puts["k_norm"] < K].sort_values("dist_to_K")
    above_K = otm_puts[otm_puts["k_norm"] > K].sort_values("dist_to_K")
    at_K = otm_puts[otm_puts["k_norm"] == K]

    pair_parts = []
    if len(below_K) > 0:
        pair_parts.append(below_K.iloc[[0]])
    if len(above_K) > 0:
        pair_parts.append(above_K.iloc[[0]])
    if len(at_K) > 0:
        pair_parts.append(at_K.iloc[[0]])

    pair_df = pd.concat(pair_parts).drop_duplicates(subset=["k_norm"]) if pair_parts else pd.DataFrame()

    # --- Local eight: pair + 6 nearest additional puts ---
    if len(pair_df) > 0:
        pair_knorms = set(pair_df["k_norm"].values)
        additional = otm_puts[~otm_puts["k_norm"].isin(pair_knorms)]
        additional_6 = additional.head(6)
        local_df = pd.concat([pair_df, additional_6]).drop_duplicates(subset=["k_norm"])
    else:
        local_df = otm_puts.head(8).copy()

    # --- Wing: all OTM puts ---
    wing_df = otm_puts.copy()

    # --- OTM: wing + all OTM calls ---
    otm_df = pd.concat([wing_df, otm_calls]).drop_duplicates(subset=["k_norm"])

    # Restore original column ordering and drop helper columns
    def _clean(d: pd.DataFrame) -> pd.DataFrame:
        if len(d) == 0:
            return d
        cols = [c for c in d.columns if c not in ("dist_to_K", "index")]
        return d[cols].sort_values("k_norm").reset_index(drop=True)

    return {
        "pair": _clean(pair_df),
        "local": _clean(local_df),
        "wing": _clean(wing_df),
        "otm": _clean(otm_df),
    }


def get_quote_set_summary(sets: Dict[str, pd.DataFrame]) -> Dict[str, int]:
    """Return quote counts for each nested set."""
    return {name: len(df) for name, df in sets.items()}
