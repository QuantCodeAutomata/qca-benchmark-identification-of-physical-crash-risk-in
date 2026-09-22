"""
Data loader for SPX option cross-sections.

Attempts to load OptionMetrics data from data/optionmetrics_spx.csv;
falls back to synthetic data if not found.
"""
from __future__ import annotations

import os
import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from src.utils import get_wednesday_dates
from src.synthetic_data import generate_synthetic_dataset

logger = logging.getLogger(__name__)

OPTIONMETRICS_CSV = "optionmetrics_spx.csv"
REQUIRED_COLS = ["date", "type", "strike", "maturity_date", "bid", "ask", "volume", "D", "F"]


def load_spx_options(
    data_dir: str = "data",
    use_synthetic: bool = False,
    start_date: str = "2013-01-01",
    end_date: str = "2023-08-31",
    horizons: Optional[List[int]] = None,
    synthetic_seed: int = 42,
    n_puts: int = 20,
    n_calls: int = 10,
) -> pd.DataFrame:
    """
    Load SPX option data. Attempts real OptionMetrics CSV; falls back to synthetic.

    Args:
        data_dir: Directory containing optionmetrics_spx.csv.
        use_synthetic: Force use of synthetic data.
        start_date: Start of sample period.
        end_date: End of sample period.
        horizons: Calendar-day horizons to retain (default [23, 30, 37]).
        synthetic_seed: Seed for synthetic data generation.
        n_puts: Number of put strikes for synthetic data.
        n_calls: Number of call strikes for synthetic data.

    Returns:
        DataFrame with columns: date, type, strike, maturity_date,
        bid, ask, volume, D, F, horizon.
    """
    if horizons is None:
        horizons = [23, 30, 37]

    csv_path = os.path.join(data_dir, OPTIONMETRICS_CSV)

    if not use_synthetic and os.path.exists(csv_path):
        logger.info("Loading OptionMetrics data from %s", csv_path)
        df = _load_optionmetrics_csv(csv_path, start_date, end_date)
        df = get_exact_horizon_cross_sections(df, horizons)
    else:
        if not use_synthetic:
            logger.warning(
                "%s not found; using synthetic data. "
                "Place OptionMetrics SPX CSV at %s for real results.",
                csv_path, csv_path,
            )
        logger.info("Generating synthetic SPX option data.")
        df = generate_synthetic_dataset(
            start_date=start_date,
            end_date=end_date,
            horizons=horizons,
            seed=synthetic_seed,
            n_puts=n_puts,
            n_calls=n_calls,
        )

    df = _parse_dates(df)
    df = _keep_wednesdays_only(df, start_date, end_date)
    df = filter_quotes(df)
    df = _deduplicate_cross_sections(df, horizons)
    return df.reset_index(drop=True)


def _load_optionmetrics_csv(
    path: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """Load and lightly validate an OptionMetrics CSV file."""
    df = pd.read_csv(path, low_memory=False)
    for col in REQUIRED_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {path}")
    df = df[REQUIRED_COLS].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["maturity_date"] = pd.to_datetime(df["maturity_date"])
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    # Compute horizon as calendar days from cross-section date to maturity
    df["horizon"] = (df["maturity_date"] - df["date"]).dt.days
    return df


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure date columns are datetime."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["maturity_date"] = pd.to_datetime(df["maturity_date"])
    return df


def _keep_wednesdays_only(
    df: pd.DataFrame, start_date: str, end_date: str
) -> pd.DataFrame:
    """Retain only cross-sections whose date falls on a Wednesday."""
    wednesdays = set(get_wednesday_dates(start_date, end_date).date)
    mask = df["date"].dt.date.isin(wednesdays)
    n_dropped = (~mask).sum()
    if n_dropped > 0:
        logger.debug("Dropped %d rows not on a Wednesday.", n_dropped)
    return df[mask].copy()


def _deduplicate_cross_sections(
    df: pd.DataFrame, horizons: List[int]
) -> pd.DataFrame:
    """Keep at most one cross-section per (date, horizon) combination."""
    # For synthetic data this is already one-per-date-horizon.
    # For OptionMetrics, take the first maturity matching the exact horizon.
    return df.drop_duplicates(subset=["date", "horizon", "type", "strike"])


def filter_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply standard quote filters:
    - strike > 0
    - bid >= 0
    - ask > 0
    - bid <= ask
    - volume > 0

    Args:
        df: Raw option quote DataFrame.

    Returns:
        Filtered DataFrame.
    """
    df = df.copy()
    mask = (
        (df["strike"] > 0)
        & (df["bid"] >= 0)
        & (df["ask"] > 0)
        & (df["bid"] <= df["ask"])
        & (df["volume"] > 0)
    )
    n_dropped = (~mask).sum()
    if n_dropped > 0:
        logger.debug("filter_quotes: dropped %d quotes.", n_dropped)
    return df[mask].copy()


def get_exact_horizon_cross_sections(
    df: pd.DataFrame, horizons: List[int]
) -> pd.DataFrame:
    """
    Filter to cross-sections with exact calendar day horizons.

    Args:
        df: DataFrame with 'horizon' column.
        horizons: List of acceptable horizon values in days.

    Returns:
        Filtered DataFrame.
    """
    if "horizon" not in df.columns:
        if "maturity_date" in df.columns and "date" in df.columns:
            df = df.copy()
            df["horizon"] = (
                pd.to_datetime(df["maturity_date"])
                - pd.to_datetime(df["date"])
            ).dt.days
        else:
            raise ValueError("Cannot compute horizon: missing maturity_date or date columns.")
    return df[df["horizon"].isin(horizons)].copy()


def get_cross_section(
    df: pd.DataFrame, date: pd.Timestamp, horizon: int
) -> pd.DataFrame:
    """
    Extract a single cross-section from the full dataset.

    Args:
        df: Full option dataset.
        date: Cross-section date.
        horizon: Calendar days to maturity.

    Returns:
        DataFrame for the single cross-section.
    """
    mask = (df["date"] == date) & (df["horizon"] == horizon)
    return df[mask].copy()


def list_cross_sections(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary of available (date, horizon) cross-sections."""
    grp = df.groupby(["date", "horizon"])["strike"].count().reset_index()
    grp.columns = ["date", "horizon", "n_quotes"]
    return grp.sort_values(["horizon", "date"]).reset_index(drop=True)
