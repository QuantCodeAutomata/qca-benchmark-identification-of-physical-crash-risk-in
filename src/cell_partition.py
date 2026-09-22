"""
Cell partition construction for the moment-based SOCP representation.

Partitions [0, B] into cells using the crash threshold K and normalized
option strikes as boundary points, as described in the paper's methodology.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def build_cell_partition(
    normalized_strikes: np.ndarray,
    K: float = 0.90,
    B: float = 5.0,
) -> Dict:
    """
    Build an ordered cell partition of [0, B].

    Boundaries = sorted union of {0, K, B} and all strikes in (0, B).
    Cells: I_j = [b_j, b_{j+1}] for j = 0, ..., n_cells-1.

    Args:
        normalized_strikes: Array of normalized strike values k_i = kappa_i / F.
        K: Crash threshold (default 0.90).
        B: Right support bound (default 5.0).

    Returns:
        Dict with:
        - boundaries: np.ndarray of sorted boundary points
        - j_K: index such that boundaries[j_K] == K
        - n_cells: total number of cells
        - cells: list of (a, b) tuples for each cell
        - below_K_indices: cell indices where b_{j+1} <= K (i.e., j < j_K)
        - threshold_idx: index of cell [K, next boundary] = j_K
        - above_K_indices: cell indices above threshold (j > j_K)
        - n_moments_per_cell: list of int (4 for below-K, 3 for at/above-K)
    """
    # Collect boundary points: {0, K, B} + strikes strictly inside (0, B)
    interior_strikes = normalized_strikes[
        (normalized_strikes > 0) & (normalized_strikes < B)
    ]
    raw_boundaries = np.unique(
        np.concatenate([[0.0, K, B], interior_strikes])
    )
    boundaries = np.sort(raw_boundaries)

    # Verify K appears in boundaries
    j_K_candidates = np.where(np.abs(boundaries - K) < 1e-12)[0]
    if len(j_K_candidates) == 0:
        raise ValueError(f"K={K} not found in boundaries after construction.")
    j_K = int(j_K_candidates[0])

    n_cells = len(boundaries) - 1
    cells = [(boundaries[j], boundaries[j + 1]) for j in range(n_cells)]

    # Cells strictly below K: both endpoints <= K, i.e., j < j_K
    below_K_indices = list(range(j_K))
    # Threshold cell: [K, next] = cell j_K
    threshold_idx = j_K
    # Cells above threshold: j > j_K
    above_K_indices = list(range(j_K + 1, n_cells))

    # Moment orders per cell
    n_moments_per_cell = []
    for j in range(n_cells):
        if j < j_K:
            n_moments_per_cell.append(4)  # orders 0,1,2,3 needed
        else:
            n_moments_per_cell.append(3)  # orders 0,1,2 needed

    return {
        "boundaries": boundaries,
        "j_K": j_K,
        "n_cells": n_cells,
        "cells": cells,
        "below_K_indices": below_K_indices,
        "threshold_idx": threshold_idx,
        "above_K_indices": above_K_indices,
        "n_moments_per_cell": n_moments_per_cell,
        "K": K,
        "B": B,
    }


def call_payoff_coefficients(
    partition: Dict,
    k_i: float,
) -> Dict[Tuple[int, int], float]:
    """
    Compute linear coefficients for the call payoff expression C_i(y).

    Since k_i is a partition boundary, C_i(y) = sum_{j: b_j >= k_i} (y_{1j} - k_i * y_{0j}).

    This is the Charnes-Cooper scaled version:
    C_i(y) = integral_{k_i}^{B} (x - k_i) d(rho*Q)(x)

    Because k_i is always a boundary point (we add all strikes to the partition),
    the expression is a sum over complete cells to the right of k_i.

    Args:
        partition: Output of build_cell_partition.
        k_i: Normalized strike (must be a boundary point).

    Returns:
        Dict mapping (cell_index, moment_order) -> coefficient.
        moment_order 0 corresponds to y_{0j}, order 1 to y_{1j}, etc.
    """
    boundaries = partition["boundaries"]
    n_cells = partition["n_cells"]
    coefs: Dict[Tuple[int, int], float] = {}

    for j in range(n_cells):
        a_j = boundaries[j]
        if a_j >= k_i - 1e-12:
            # Entire cell contributes: integral_{a_j}^{b_j} (x-k_i) dmu_j
            # = y_{1j} - k_i * y_{0j}
            coefs[(j, 1)] = coefs.get((j, 1), 0.0) + 1.0
            coefs[(j, 0)] = coefs.get((j, 0), 0.0) - k_i

    return coefs
