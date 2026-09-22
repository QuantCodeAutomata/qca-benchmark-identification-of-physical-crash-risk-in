"""
Synthetic SPX option data generator for testing.

Generates realistic Black-Scholes based option cross-sections mimicking
OptionMetrics end-of-day SPX quotes for the crash-risk identification experiments.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import List, Optional

from src.utils import get_wednesday_dates


def _bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call price."""
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def _bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes put price."""
    call = _bs_call(S, K, T, r, sigma)
    return call - S + K * np.exp(-r * T)


def _implied_vol_skew(moneyness: float, base_vol: float) -> float:
    """Approximate implied vol with a smile/skew (puts more expensive for OTM)."""
    # moneyness = K/F, so <1 is OTM put / ITM call
    log_m = np.log(moneyness)
    # Negative skew: OTM puts have higher vol
    skew = -0.15 * log_m - 0.30 * log_m**2
    return np.clip(base_vol + skew, 0.05, 2.0)


def generate_synthetic_cross_section(
    date: pd.Timestamp,
    horizon: int,
    seed: Optional[int] = None,
    n_puts: int = 20,
    n_calls: int = 10,
    spot: float = 4000.0,
    vol: float = 0.20,
    rate: float = 0.03,
) -> pd.DataFrame:
    """
    Generate a synthetic SPX option cross-section using Black-Scholes prices.

    Args:
        date: Cross-section date (Wednesday).
        horizon: Calendar days to maturity.
        seed: Random seed for reproducibility.
        n_puts: Number of OTM put strikes.
        n_calls: Number of OTM call strikes.
        spot: Spot price of the index.
        vol: Base ATM implied volatility.
        rate: Risk-free rate.

    Returns:
        DataFrame with columns: date, type, strike, maturity_date,
        bid, ask, volume, D, F.
    """
    rng = np.random.default_rng(seed)

    T = horizon / 365.0
    D = np.exp(-rate * T)
    F = spot * np.exp(rate * T)
    maturity_date = date + pd.Timedelta(days=horizon)

    # Put strikes: from ~60% to ~100% of forward
    put_moneyness = np.sort(
        np.concatenate([
            np.linspace(0.60, 0.95, n_puts - 2),
            [0.975, 0.99],
        ])
    )
    put_strikes = put_moneyness * F

    # Call strikes: from ~100% to ~140% of forward
    call_moneyness = np.sort(
        np.concatenate([
            [1.01, 1.025],
            np.linspace(1.05, 1.40, n_calls - 2),
        ])
    )
    call_strikes = call_moneyness * F

    rows = []
    for K_raw in put_strikes:
        m = K_raw / F
        sigma = _implied_vol_skew(m, vol)
        mid = _bs_put(spot, K_raw, T, rate, sigma)
        if mid <= 0:
            mid = 1e-4
        # Spread: wider for deep OTM
        spread_pct = 0.03 + 0.10 * max(0.0, 0.80 - m)
        half = max(mid * spread_pct, 0.05)
        bid = max(0.0, mid - half)
        ask = mid + half
        volume = int(rng.integers(10, 500))
        rows.append({
            "date": date,
            "type": "P",
            "strike": round(K_raw, 2),
            "maturity_date": maturity_date,
            "bid": round(bid, 4),
            "ask": round(ask, 4),
            "volume": volume,
            "D": D,
            "F": F,
        })

    for K_raw in call_strikes:
        m = K_raw / F
        sigma = _implied_vol_skew(m, vol)
        mid = _bs_call(spot, K_raw, T, rate, sigma)
        if mid <= 0:
            mid = 1e-4
        spread_pct = 0.03 + 0.05 * max(0.0, m - 1.10)
        half = max(mid * spread_pct, 0.05)
        bid = max(0.0, mid - half)
        ask = mid + half
        volume = int(rng.integers(10, 500))
        rows.append({
            "date": date,
            "type": "C",
            "strike": round(K_raw, 2),
            "maturity_date": maturity_date,
            "bid": round(bid, 4),
            "ask": round(ask, 4),
            "volume": volume,
            "D": D,
            "F": F,
        })

    return pd.DataFrame(rows)


def generate_synthetic_dataset(
    start_date: str = "2013-01-01",
    end_date: str = "2023-08-31",
    horizons: List[int] = None,
    seed: int = 42,
    n_puts: int = 20,
    n_calls: int = 10,
) -> pd.DataFrame:
    """
    Generate a synthetic dataset of SPX option cross-sections for all
    Wednesdays in [start_date, end_date] and each specified horizon.

    Args:
        start_date: Start of sample period.
        end_date: End of sample period.
        horizons: List of calendar-day horizons (default: [23, 30, 37]).
        seed: Base random seed.
        n_puts: Number of put strikes per cross-section.
        n_calls: Number of call strikes per cross-section.

    Returns:
        DataFrame with all cross-sections concatenated.
    """
    if horizons is None:
        horizons = [23, 30, 37]

    wednesdays = get_wednesday_dates(start_date, end_date)
    all_dfs = []

    # Use a slowly varying spot/vol process for realism
    rng = np.random.default_rng(seed)
    n_weeks = len(wednesdays)
    # Simulate log-spot and vol paths
    log_spots = np.cumsum(np.concatenate([[np.log(4000.0)],
        0.001 * rng.standard_normal(n_weeks)]))[:n_weeks]
    vols = np.clip(
        0.18 + np.cumsum(np.concatenate([[0.0],
            0.002 * rng.standard_normal(n_weeks)]))[:n_weeks],
        0.10, 0.60,
    )

    for idx, date in enumerate(wednesdays):
        spot = np.exp(log_spots[idx])
        vol = vols[idx]
        for h_idx, horizon in enumerate(horizons):
            cs_seed = seed + idx * 100 + h_idx
            df = generate_synthetic_cross_section(
                date=date,
                horizon=horizon,
                seed=cs_seed,
                n_puts=n_puts,
                n_calls=n_calls,
                spot=spot,
                vol=vol,
            )
            df["horizon"] = horizon
            all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)
