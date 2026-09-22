"""Tests for src/cell_partition.py."""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cell_partition import build_cell_partition, call_payoff_coefficients


# ---------------------------------------------------------------------------
# build_cell_partition
# ---------------------------------------------------------------------------

class TestBuildCellPartition:
    def test_basic_structure(self):
        """Partition contains K as boundary and right bound B."""
        strikes = np.array([0.80, 0.85, 0.90, 0.95, 1.00, 1.10])
        p = build_cell_partition(strikes, K=0.90, B=5.0)
        assert 0.0 in p["boundaries"]
        assert 0.90 in p["boundaries"] or any(abs(b - 0.90) < 1e-12 for b in p["boundaries"])
        assert 5.0 in p["boundaries"]

    def test_K_is_boundary(self):
        """K must appear exactly once in boundaries."""
        strikes = np.array([0.80, 0.85, 0.95])
        p = build_cell_partition(strikes, K=0.90, B=5.0)
        count_K = sum(abs(b - 0.90) < 1e-12 for b in p["boundaries"])
        assert count_K == 1, f"K appears {count_K} times in boundaries"

    def test_boundary_count(self):
        """Boundaries = union of {0, K, B} + strikes in (0, B)."""
        strikes = np.array([0.80, 0.85, 0.95, 1.05])
        p = build_cell_partition(strikes, K=0.90, B=5.0)
        # All of 0, 0.80, 0.85, 0.90, 0.95, 1.05, 5.0
        expected_bdys = sorted({0.0, 0.80, 0.85, 0.90, 0.95, 1.05, 5.0})
        np.testing.assert_allclose(p["boundaries"], expected_bdys, atol=1e-10)

    def test_n_cells_equals_boundaries_minus_one(self):
        strikes = np.array([0.80, 0.95])
        p = build_cell_partition(strikes, K=0.90, B=5.0)
        assert p["n_cells"] == len(p["boundaries"]) - 1

    def test_below_K_indices(self):
        """Cells strictly below K should have right endpoint <= K."""
        strikes = np.array([0.80, 0.85, 0.95])
        p = build_cell_partition(strikes, K=0.90, B=5.0)
        for j in p["below_K_indices"]:
            a, b = p["cells"][j]
            assert b <= 0.90 + 1e-12, f"Cell {j} = [{a},{b}] not below K"

    def test_threshold_cell_starts_at_K(self):
        strikes = np.array([0.80, 0.95])
        p = build_cell_partition(strikes, K=0.90, B=5.0)
        thresh = p["threshold_idx"]
        a_thresh, _ = p["cells"][thresh]
        assert abs(a_thresh - 0.90) < 1e-12

    def test_above_K_indices(self):
        """Cells above threshold should start strictly above K."""
        strikes = np.array([0.80, 0.95, 1.10])
        p = build_cell_partition(strikes, K=0.90, B=5.0)
        for j in p["above_K_indices"]:
            a, b = p["cells"][j]
            assert a >= 0.90 - 1e-12

    def test_moment_orders(self):
        """Below-K cells have 4 moments; threshold and above have 3."""
        strikes = np.array([0.80, 0.85, 0.95])
        p = build_cell_partition(strikes, K=0.90, B=5.0)
        for j in p["below_K_indices"]:
            assert p["n_moments_per_cell"][j] == 4
        assert p["n_moments_per_cell"][p["threshold_idx"]] == 3
        for j in p["above_K_indices"]:
            assert p["n_moments_per_cell"][j] == 3

    def test_no_strikes(self):
        """Works with empty strike array (only K and B in partition)."""
        strikes = np.array([])
        p = build_cell_partition(strikes, K=0.90, B=5.0)
        assert len(p["boundaries"]) >= 3  # at least {0, K, B}

    def test_strike_at_K(self):
        """Strike exactly at K is absorbed as boundary without duplication."""
        strikes = np.array([0.80, 0.90, 0.95])
        p = build_cell_partition(strikes, K=0.90, B=5.0)
        count_K = sum(abs(b - 0.90) < 1e-12 for b in p["boundaries"])
        assert count_K == 1

    def test_cells_cover_full_support(self):
        """Union of all cells spans [0, B]."""
        strikes = np.array([0.80, 0.95, 1.20])
        p = build_cell_partition(strikes, K=0.90, B=5.0)
        assert p["cells"][0][0] == pytest.approx(0.0)
        assert p["cells"][-1][1] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# call_payoff_coefficients
# ---------------------------------------------------------------------------

class TestCallPayoffCoefficients:
    def test_OTM_call_coefs_nonempty(self):
        """A call strike below B should produce nonzero coefficients."""
        strikes = np.array([0.80, 0.95])
        p = build_cell_partition(strikes, K=0.90, B=5.0)
        coefs = call_payoff_coefficients(p, k_i=0.95)
        assert len(coefs) > 0

    def test_ITM_coefs_all_cells(self):
        """Strike at 0 means all cells contribute (integral of payoff from 0 to B)."""
        strikes = np.array([0.80, 0.95])
        p = build_cell_partition(strikes, K=0.90, B=5.0)
        coefs = call_payoff_coefficients(p, k_i=0.0)
        # At k=0, all cells contribute
        assert len(coefs) >= p["n_cells"]

    def test_strike_beyond_B_empty(self):
        """Strike >= B means no cell starts at or beyond B: empty coefs."""
        strikes = np.array([0.80, 0.95])
        p = build_cell_partition(strikes, K=0.90, B=5.0)
        coefs = call_payoff_coefficients(p, k_i=5.0)
        assert len(coefs) == 0

    def test_coef_structure(self):
        """Coefficients are (cell_idx, moment_order) -> float."""
        strikes = np.array([0.85, 0.95])
        p = build_cell_partition(strikes, K=0.90, B=5.0)
        coefs = call_payoff_coefficients(p, k_i=0.85)
        for (j, r), c in coefs.items():
            assert isinstance(j, int)
            assert r in (0, 1)  # Only y_0 and y_1 appear in payoff expression

    def test_coef_values_sign(self):
        """y_1 coefficient should be +1, y_0 coefficient should be -k_i."""
        strikes = np.array([0.80, 0.95])
        p = build_cell_partition(strikes, K=0.90, B=5.0)
        k_i = 0.85
        coefs = call_payoff_coefficients(p, k_i=k_i)
        for (j, r), c in coefs.items():
            if r == 1:
                assert c == pytest.approx(1.0)
            elif r == 0:
                assert c == pytest.approx(-k_i)
