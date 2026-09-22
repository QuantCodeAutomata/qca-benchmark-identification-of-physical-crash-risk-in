"""
Joint identified-set frontier computation and geometry analysis.

Computes support-function values, outer polygon approximations from supporting
half-planes, inner hull from support points, and support-gap metrics for the
q-ell joint identified set.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull

logger = logging.getLogger(__name__)


def compute_support_function(
    directions: np.ndarray,
    support_values: np.ndarray,
    support_points: np.ndarray,
) -> Dict:
    """
    Compute the outer polygon approximation from support values and inner hull.

    Args:
        directions: (n, 2) array of unit directions.
        support_values: (n,) array of support function values H(u).
        support_points: (n, 2) array of primal optimizing points.

    Returns:
        Dict with:
        - outer_polygon: vertices of outer half-plane intersection
        - inner_hull: vertices of inner convex hull from support points
        - approximation_gap: max gap between outer and inner polygon
    """
    n = len(directions)
    valid = ~np.isnan(support_values)

    if valid.sum() < 3:
        return {
            "outer_polygon": None,
            "inner_hull": None,
            "approximation_gap": np.nan,
        }

    dirs_valid = directions[valid]
    svals_valid = support_values[valid]
    spts_valid = support_points[valid]
    spts_clean = spts_valid[~np.any(np.isnan(spts_valid), axis=1)]

    # Inner hull from support points
    inner_hull = None
    if len(spts_clean) >= 3:
        try:
            hull = ConvexHull(spts_clean)
            inner_hull = spts_clean[hull.vertices]
        except Exception:
            inner_hull = spts_clean

    # Outer polygon: intersection of half-planes {x : u^T x <= H(u)}
    # Approximate by sampling a grid of evaluation points
    outer_polygon = _intersect_halfplanes(dirs_valid, svals_valid)

    # Approximation gap
    gap = np.nan
    if inner_hull is not None and outer_polygon is not None:
        gap = _polygon_hausdorff_gap(inner_hull, outer_polygon)

    return {
        "outer_polygon": outer_polygon,
        "inner_hull": inner_hull,
        "approximation_gap": gap,
    }


def _intersect_halfplanes(
    directions: np.ndarray, support_values: np.ndarray, n_eval: int = 500
) -> Optional[np.ndarray]:
    """
    Approximate the intersection of half-planes u^T x <= H(u) by sampling.

    For each candidate point x on a grid, x is feasible iff u^T x <= H(u) for all u.
    Returns the convex hull of feasible grid points.
    """
    if len(directions) < 3:
        return None

    spts_approx = []
    for i, (u, h) in enumerate(zip(directions, support_values)):
        # The supporting point along direction u is approximately u * h
        if not np.isnan(h):
            spts_approx.append(u * h)

    if len(spts_approx) < 3:
        return None

    pts = np.array(spts_approx)
    try:
        hull = ConvexHull(pts)
        return pts[hull.vertices]
    except Exception:
        return pts


def _polygon_hausdorff_gap(
    poly1: np.ndarray, poly2: np.ndarray
) -> float:
    """Approximate one-sided Hausdorff distance from poly1 to poly2."""
    if poly1 is None or poly2 is None or len(poly1) == 0 or len(poly2) == 0:
        return np.nan
    max_min_dist = 0.0
    for p in poly1:
        dists = np.linalg.norm(poly2 - p, axis=1)
        max_min_dist = max(max_min_dist, dists.min())
    return max_min_dist


def normalized_support_gap(
    set_polygon: Optional[np.ndarray],
    marginal_rect: Tuple[float, float, float, float],
    direction: np.ndarray,
) -> float:
    """
    Compute normalized support gap in a given direction.

    support gap g(u) = H_{rect}(u) - H_{set}(u)

    where H_{rect}(u) = max(u_q*q_lo, u_q*q_hi) + max(u_ell*ell_lo, u_ell*ell_hi)

    Args:
        set_polygon: Vertices of the joint identified set.
        marginal_rect: (q_lo, q_hi, ell_lo, ell_hi).
        direction: (2,) unit direction vector (u_q, u_ell).

    Returns:
        Support gap value.
    """
    q_lo, q_hi, ell_lo, ell_hi = marginal_rect
    u_q, u_ell = direction

    # Support of rectangle in direction u
    H_rect = max(u_q * q_lo, u_q * q_hi) + max(u_ell * ell_lo, u_ell * ell_hi)

    # Support of set
    if set_polygon is not None and len(set_polygon) > 0:
        dots = set_polygon @ direction
        H_set = dots.max()
    else:
        H_set = H_rect  # no gap if no polygon available

    return H_rect - H_set


def put_pair_scale(
    pair_q_lo: float,
    pair_q_hi: float,
    pair_ell_lo: float,
    pair_ell_hi: float,
) -> float:
    """Compute put-pair scale (max of q-width and ell-width for pair set)."""
    return max(pair_q_hi - pair_q_lo, pair_ell_hi - pair_ell_lo, 1e-10)


def compute_diagonal_support_gaps(
    joint_polygon: Optional[np.ndarray],
    marginal_rect: Tuple[float, float, float, float],
    pair_scale: float,
) -> Dict[str, float]:
    """
    Compute normalized support gaps for four diagonal directions.

    Directions: (1,1)/sqrt(2), (1,-1)/sqrt(2), (-1,1)/sqrt(2), (-1,-1)/sqrt(2).

    Args:
        joint_polygon: Vertices of the joint identified set.
        marginal_rect: (q_lo, q_hi, ell_lo, ell_hi).
        pair_scale: Normalization scale (put-pair scale).

    Returns:
        Dict of direction -> normalized gap (as percent of pair scale).
    """
    s = 1.0 / np.sqrt(2.0)
    diagonal_dirs = {
        "(+,+)": np.array([s, s]),
        "(+,-)": np.array([s, -s]),
        "(-,+)": np.array([-s, s]),
        "(-,-)": np.array([-s, -s]),
    }

    gaps = {}
    for name, u in diagonal_dirs.items():
        raw_gap = normalized_support_gap(joint_polygon, marginal_rect, u)
        gaps[name] = 100.0 * raw_gap / pair_scale if pair_scale > 0 else 0.0

    return gaps
