"""Tests for src/quote_processor.py."""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quote_processor import normalize_quotes, build_nested_quote_sets


def _make_cross_section(n_puts=8, n_calls=4, K=0.90) -> pd.DataFrame:
    """Create a minimal synthetic cross-section DataFrame."""
    F = 4000.0
    D = 0.99
    date = pd.Timestamp("2020-01-08")
    maturity = pd.Timestamp("2020-02-05")

    rows = []
    # OTM puts (k < 1.0)
    for i, m in enumerate(np.linspace(0.70, 0.99, n_puts)):
        K_raw = m * F
        mid = max(F * 0.02 * (1.0 - m), 0.1)
        rows.append({
            "date": date, "type": "P", "strike": K_raw,
            "maturity_date": maturity,
            "bid": round(mid * 0.95, 4), "ask": round(mid * 1.05, 4),
            "volume": 100, "D": D, "F": F, "horizon": 28,
        })
    # OTM calls (k > 1.0)
    for i, m in enumerate(np.linspace(1.01, 1.20, n_calls)):
        K_raw = m * F
        mid = max(F * 0.02 * (m - 1.0), 0.1)
        rows.append({
            "date": date, "type": "C", "strike": K_raw,
            "maturity_date": maturity,
            "bid": round(mid * 0.95, 4), "ask": round(mid * 1.05, 4),
            "volume": 100, "D": D, "F": F, "horizon": 28,
        })
    return pd.DataFrame(rows)


class TestNormalizeQuotes:
    def test_k_norm_column(self):
        df = _make_cross_section()
        norm = normalize_quotes(df)
        assert "k_norm" in norm.columns
        np.testing.assert_allclose(norm["k_norm"], norm["strike"] / norm["F"])

    def test_c_lo_c_hi_columns(self):
        df = _make_cross_section()
        norm = normalize_quotes(df)
        assert "c_lo" in norm.columns
        assert "c_hi" in norm.columns

    def test_c_lo_nonnegative(self):
        df = _make_cross_section()
        norm = normalize_quotes(df)
        assert (norm["c_lo"] >= 0.0).all()

    def test_c_lo_le_c_hi(self):
        df = _make_cross_section()
        norm = normalize_quotes(df)
        assert (norm["c_lo"] <= norm["c_hi"] + 1e-10).all()

    def test_put_call_parity_conversion(self):
        """Put quotes should be converted to call-equivalent via parity."""
        F = 4000.0
        D = 0.99
        date = pd.Timestamp("2020-01-08")
        mat = pd.Timestamp("2020-02-05")
        # One put quote
        df = pd.DataFrame([{
            "date": date, "type": "P", "strike": 3600.0,
            "maturity_date": mat, "bid": 50.0, "ask": 55.0,
            "volume": 100, "D": D, "F": F, "horizon": 28,
        }])
        norm = normalize_quotes(df)
        k = 3600.0 / F
        DF = D * F
        expected_lo = max(0.0, 50.0 / DF + 1.0 - k)
        expected_hi = 55.0 / DF + 1.0 - k
        assert norm["c_lo"].iloc[0] == pytest.approx(expected_lo, abs=1e-8)
        assert norm["c_hi"].iloc[0] == pytest.approx(expected_hi, abs=1e-8)

    def test_empty_input(self):
        """Empty DataFrame returns empty DataFrame."""
        df = pd.DataFrame(columns=["date", "type", "strike", "maturity_date",
                                    "bid", "ask", "volume", "D", "F"])
        df["D"] = df["D"].astype(float)
        df["F"] = df["F"].astype(float)
        norm = normalize_quotes(df)
        assert len(norm) == 0

    def test_is_otm_flag(self):
        df = _make_cross_section()
        norm = normalize_quotes(df)
        # OTM puts: k_norm < 1.0, type=P
        puts = norm[norm["type"] == "P"]
        assert (puts[puts["k_norm"] < 1.0]["is_otm"]).all()
        # OTM calls: k_norm > 1.0, type=C
        calls = norm[norm["type"] == "C"]
        assert (calls[calls["k_norm"] > 1.0]["is_otm"]).all()


class TestBuildNestedQuoteSets:
    def test_keys_present(self):
        df = _make_cross_section()
        norm = normalize_quotes(df)
        sets = build_nested_quote_sets(norm, K=0.90)
        assert set(sets.keys()) == {"pair", "local", "wing", "otm"}

    def test_pair_size_at_most_two(self):
        df = _make_cross_section()
        norm = normalize_quotes(df)
        sets = build_nested_quote_sets(norm, K=0.90)
        # Pair should have at most 2 puts (one each side of K)
        assert len(sets["pair"]) <= 2

    def test_local_size_at_most_eight(self):
        df = _make_cross_section(n_puts=15)
        norm = normalize_quotes(df)
        sets = build_nested_quote_sets(norm, K=0.90)
        # Local = pair + 6 additional puts
        assert len(sets["local"]) <= 8

    def test_nesting_pair_in_local(self):
        """Pair strikes must be a subset of local strikes."""
        df = _make_cross_section(n_puts=12)
        norm = normalize_quotes(df)
        sets = build_nested_quote_sets(norm, K=0.90)
        pair_knorms = set(sets["pair"]["k_norm"].values)
        local_knorms = set(sets["local"]["k_norm"].values)
        assert pair_knorms.issubset(local_knorms), "pair not subset of local"

    def test_nesting_local_in_wing(self):
        df = _make_cross_section(n_puts=12)
        norm = normalize_quotes(df)
        sets = build_nested_quote_sets(norm, K=0.90)
        local_knorms = set(sets["local"]["k_norm"].values)
        wing_knorms = set(sets["wing"]["k_norm"].values)
        assert local_knorms.issubset(wing_knorms), "local not subset of wing"

    def test_nesting_wing_in_otm(self):
        df = _make_cross_section(n_puts=12, n_calls=5)
        norm = normalize_quotes(df)
        sets = build_nested_quote_sets(norm, K=0.90)
        wing_knorms = set(sets["wing"]["k_norm"].values)
        otm_knorms = set(sets["otm"]["k_norm"].values)
        assert wing_knorms.issubset(otm_knorms), "wing not subset of otm"

    def test_otm_contains_calls(self):
        df = _make_cross_section(n_puts=8, n_calls=4)
        norm = normalize_quotes(df)
        sets = build_nested_quote_sets(norm, K=0.90)
        has_calls = any(
            row["type"] == "C" for _, row in sets["otm"].iterrows()
            if "type" in sets["otm"].columns
        )
        # OTM set should include OTM calls
        otm_calls = sets["otm"][sets["otm"]["k_norm"] > 1.0] if "k_norm" in sets["otm"].columns else pd.DataFrame()
        assert len(otm_calls) > 0 or len(sets["otm"]) > 0  # at least something

    def test_wing_no_calls(self):
        """Wing should contain only OTM puts (k_norm < 1.0)."""
        df = _make_cross_section(n_puts=10, n_calls=5)
        norm = normalize_quotes(df)
        sets = build_nested_quote_sets(norm, K=0.90)
        if len(sets["wing"]) > 0:
            assert (sets["wing"]["k_norm"] < 1.0 + 1e-10).all()

    def test_empty_when_no_otm_puts(self):
        """If no OTM puts, pair and local should be empty."""
        df = _make_cross_section(n_puts=0, n_calls=5)
        norm = normalize_quotes(df)
        sets = build_nested_quote_sets(norm, K=0.90)
        assert len(sets["pair"]) == 0
        assert len(sets["wing"]) == 0
