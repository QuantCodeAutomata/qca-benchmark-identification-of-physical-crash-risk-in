"""Tests for src/data_loader.py and src/synthetic_data.py."""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import (
    filter_quotes,
    get_exact_horizon_cross_sections,
    get_cross_section,
    list_cross_sections,
    load_spx_options,
)
from src.synthetic_data import (
    generate_synthetic_cross_section,
    generate_synthetic_dataset,
)
from src.utils import get_wednesday_dates  # Wednesday helper in utils


# ---------------------------------------------------------------------------
# filter_quotes
# ---------------------------------------------------------------------------

class TestFilterQuotes:
    def _make_df(self, records):
        return pd.DataFrame(records)

    def test_drops_zero_strike(self):
        df = self._make_df([
            {"strike": 0.0, "bid": 1.0, "ask": 2.0, "volume": 10},
            {"strike": 3500.0, "bid": 1.0, "ask": 2.0, "volume": 10},
        ])
        out = filter_quotes(df)
        assert len(out) == 1

    def test_drops_negative_bid(self):
        df = self._make_df([
            {"strike": 3500.0, "bid": -1.0, "ask": 2.0, "volume": 10},
            {"strike": 3500.0, "bid": 0.0, "ask": 2.0, "volume": 10},
        ])
        out = filter_quotes(df)
        assert len(out) == 1

    def test_drops_zero_ask(self):
        df = self._make_df([
            {"strike": 3500.0, "bid": 0.0, "ask": 0.0, "volume": 10},
            {"strike": 3500.0, "bid": 0.0, "ask": 0.5, "volume": 10},
        ])
        out = filter_quotes(df)
        assert len(out) == 1

    def test_drops_crossed_quotes(self):
        df = self._make_df([
            {"strike": 3500.0, "bid": 5.0, "ask": 3.0, "volume": 10},  # crossed
            {"strike": 3500.0, "bid": 1.0, "ask": 3.0, "volume": 10},  # ok
        ])
        out = filter_quotes(df)
        assert len(out) == 1

    def test_drops_zero_volume(self):
        df = self._make_df([
            {"strike": 3500.0, "bid": 1.0, "ask": 2.0, "volume": 0},
            {"strike": 3500.0, "bid": 1.0, "ask": 2.0, "volume": 1},
        ])
        out = filter_quotes(df)
        assert len(out) == 1

    def test_zero_bid_is_allowed(self):
        """bid=0 is valid (deeply OTM options can have zero bid)."""
        df = self._make_df([
            {"strike": 3500.0, "bid": 0.0, "ask": 0.5, "volume": 5},
        ])
        out = filter_quotes(df)
        assert len(out) == 1

    def test_all_valid_kept(self):
        df = self._make_df([
            {"strike": s, "bid": 1.0, "ask": 2.0, "volume": 10}
            for s in [3400, 3500, 3600]
        ])
        out = filter_quotes(df)
        assert len(out) == 3

    def test_empty_returns_empty(self):
        df = pd.DataFrame(columns=["strike", "bid", "ask", "volume"])
        out = filter_quotes(df)
        assert len(out) == 0


# ---------------------------------------------------------------------------
# get_exact_horizon_cross_sections
# ---------------------------------------------------------------------------

class TestGetExactHorizonCrossSections:
    def test_filters_to_specified_horizons(self):
        df = pd.DataFrame({
            "horizon": [23, 30, 37, 45, 60],
            "other": [1, 2, 3, 4, 5],
        })
        out = get_exact_horizon_cross_sections(df, [23, 30, 37])
        assert set(out["horizon"].unique()) == {23, 30, 37}

    def test_computes_horizon_from_dates(self):
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        maturities = dates + pd.to_timedelta([30, 23, 37], unit="D")
        df = pd.DataFrame({"date": dates, "maturity_date": maturities})
        out = get_exact_horizon_cross_sections(df, [30, 37])
        assert set(out["horizon"].unique()).issubset({30, 37})

    def test_returns_empty_if_no_match(self):
        df = pd.DataFrame({"horizon": [60, 90]})
        out = get_exact_horizon_cross_sections(df, [23, 30, 37])
        assert len(out) == 0


# ---------------------------------------------------------------------------
# list_cross_sections
# ---------------------------------------------------------------------------

class TestListCrossSections:
    def test_returns_grouped_summary(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-08"]),
            "horizon": [30, 30, 30],
            "strike": [3500, 3600, 3500],
        })
        summary = list_cross_sections(df)
        assert "n_quotes" in summary.columns
        assert len(summary) >= 1


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

class TestGenerateSyntheticCrossSection:
    def test_returns_dataframe(self):
        df = generate_synthetic_cross_section(pd.Timestamp("2020-03-18"), horizon=30, seed=0)
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self):
        df = generate_synthetic_cross_section(pd.Timestamp("2020-03-18"), horizon=30, seed=0)
        for col in ["date", "type", "strike", "maturity_date", "bid", "ask", "volume", "D", "F"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_put_call_types(self):
        df = generate_synthetic_cross_section(pd.Timestamp("2020-03-18"), horizon=30, seed=0)
        assert set(df["type"].unique()).issubset({"P", "C"})

    def test_bid_le_ask(self):
        df = generate_synthetic_cross_section(pd.Timestamp("2020-03-18"), horizon=30, seed=0)
        assert (df["bid"] <= df["ask"]).all()

    def test_positive_strikes(self):
        df = generate_synthetic_cross_section(pd.Timestamp("2020-03-18"), horizon=30, seed=0)
        assert (df["strike"] > 0).all()

    def test_forward_constraint(self):
        """D and F should be strictly positive."""
        df = generate_synthetic_cross_section(pd.Timestamp("2020-03-18"), horizon=30, seed=0)
        assert (df["D"] > 0).all()
        assert (df["F"] > 0).all()

    def test_maturity_offset(self):
        date = pd.Timestamp("2020-03-18")
        horizon = 30
        df = generate_synthetic_cross_section(date, horizon=horizon, seed=0)
        expected_maturity = date + pd.Timedelta(days=horizon)
        assert (df["maturity_date"] == expected_maturity).all()

    def test_deterministic_with_seed(self):
        date = pd.Timestamp("2020-03-18")
        df1 = generate_synthetic_cross_section(date, horizon=30, seed=123)
        df2 = generate_synthetic_cross_section(date, horizon=30, seed=123)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_different_data(self):
        date = pd.Timestamp("2020-03-18")
        df1 = generate_synthetic_cross_section(date, horizon=30, seed=1)
        df2 = generate_synthetic_cross_section(date, horizon=30, seed=2)
        assert not df1.equals(df2)


class TestGenerateSyntheticDataset:
    def test_returns_dataframe(self):
        df = generate_synthetic_dataset("2020-01-01", "2020-03-31", horizons=[30])
        assert isinstance(df, pd.DataFrame)

    def test_covers_wednesdays_only(self):
        df = generate_synthetic_dataset("2020-01-01", "2020-03-31", horizons=[30])
        wednesdays = get_wednesday_dates("2020-01-01", "2020-03-31")
        for d in df["date"].unique():
            assert pd.Timestamp(d).dayofweek == 2, f"{d} is not a Wednesday"

    def test_multiple_horizons(self):
        df = generate_synthetic_dataset("2020-01-01", "2020-02-28",
                                         horizons=[23, 30, 37], seed=42)
        assert set(df["horizon"].unique()) == {23, 30, 37}

    def test_nonempty(self):
        df = generate_synthetic_dataset("2020-01-01", "2020-03-31", horizons=[30])
        assert len(df) > 0


# ---------------------------------------------------------------------------
# load_spx_options (synthetic mode)
# ---------------------------------------------------------------------------

class TestLoadSPXOptionsSynthetic:
    def test_loads_without_error(self):
        df = load_spx_options(
            use_synthetic=True,
            start_date="2020-01-01",
            end_date="2020-06-30",
            horizons=[30],
            synthetic_seed=42,
            n_puts=10,
            n_calls=5,
        )
        assert len(df) > 0

    def test_only_wednesdays(self):
        df = load_spx_options(
            use_synthetic=True,
            start_date="2020-01-01",
            end_date="2020-06-30",
            horizons=[30],
        )
        for d in df["date"].unique():
            assert pd.Timestamp(d).dayofweek == 2

    def test_filters_applied(self):
        df = load_spx_options(
            use_synthetic=True,
            start_date="2020-01-01",
            end_date="2020-06-30",
            horizons=[30],
        )
        # All quotes should pass filters
        assert (df["strike"] > 0).all()
        assert (df["bid"] >= 0).all()
        assert (df["ask"] > 0).all()
        assert (df["bid"] <= df["ask"]).all()
        assert (df["volume"] > 0).all()
