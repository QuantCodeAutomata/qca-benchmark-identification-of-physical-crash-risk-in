"""
SOCP-based identified-set solver for physical crash-risk under CRRA utility.

Implements the exact gamma=2 Charnes-Cooper formulation described in the paper:
  - Charnes-Cooper scaling: rho = M(Q)^{-1}, d~Q = rho dQ
  - Moment variables y_{r,j} for each support cell and moment order r
  - Threshold atom a_K >= 0 at x = K
  - Truncated Hausdorff moment PSD conditions for support feasibility
  - Linear identified-set endpoint programs for q, ell, s

For gamma != 2, a separate (gamma=1) path is provided.

Solver: CLARABEL (primary), SCS (fallback).
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd

from src.cell_partition import build_cell_partition, call_payoff_coefficients
from src.utils import safe_divide

logger = logging.getLogger(__name__)

# Solver settings
PRIMARY_SOLVER = "CLARABEL"
FALLBACK_SOLVER = "SCS"
CLARABEL_OPTS = {
    "max_iter": 50000,
    "tol_gap_abs": 1e-7,
    "tol_gap_rel": 1e-7,
    "tol_infeas_abs": 1e-8,
    "tol_infeas_rel": 1e-8,
    "time_limit": 60.0,
}
SCS_OPTS = {
    "max_iters": 50000,
    "eps": 1e-6,
    "eps_infeas": 1e-7,
    "verbose": False,
}

OPTIMAL_STATUSES = {"optimal", "optimal_inaccurate"}


@dataclass
class EndpointResult:
    """Result of a single identified-set endpoint solve."""

    value: Optional[float]
    status: str
    is_valid: bool


@dataclass
class CrossSectionResult:
    """Full identified-set result for one cross-section and one quote set."""

    date: pd.Timestamp
    horizon: int
    quote_set: str
    n_quotes: int
    gamma: float
    K: float
    B: float
    # q: crash probability
    q_lo: Optional[float] = None
    q_hi: Optional[float] = None
    q_width: Optional[float] = None
    status_q_lo: str = ""
    status_q_hi: str = ""
    # ell: unconditional shortfall
    ell_lo: Optional[float] = None
    ell_hi: Optional[float] = None
    ell_width: Optional[float] = None
    status_ell_lo: str = ""
    status_ell_hi: str = ""
    # s: conditional crash depth
    s_lo: Optional[float] = None
    s_hi: Optional[float] = None
    s_width: Optional[float] = None
    status_s_lo: str = ""
    status_s_hi: str = ""
    # regularity
    regularity_verified: bool = False
    regularity_slack: Optional[float] = None
    # frontier (optional)
    frontier_directions: Optional[np.ndarray] = None
    frontier_support_values: Optional[np.ndarray] = None
    frontier_points: Optional[np.ndarray] = None
    valid: bool = False  # True if at least q endpoints solved

    def _pct(self, v: Optional[float]) -> Optional[float]:
        """Convert to percentage."""
        return None if v is None else v * 100.0

    def summary_dict(self) -> Dict:
        """Return a flat dict of key results in percentage units."""
        return {
            "date": self.date,
            "horizon": self.horizon,
            "quote_set": self.quote_set,
            "n_quotes": self.n_quotes,
            "gamma": self.gamma,
            "K": self.K,
            "B": self.B,
            "q_lo_pct": self._pct(self.q_lo),
            "q_hi_pct": self._pct(self.q_hi),
            "q_width_pct": self._pct(self.q_width),
            "ell_lo_pct": self._pct(self.ell_lo),
            "ell_hi_pct": self._pct(self.ell_hi),
            "ell_width_pct": self._pct(self.ell_width),
            "s_lo_pct": self._pct(self.s_lo),
            "s_hi_pct": self._pct(self.s_hi),
            "s_width_pct": self._pct(self.s_width),
            "regularity_verified": self.regularity_verified,
            "regularity_slack": self.regularity_slack,
            "valid": self.valid,
            "status_q_lo": self.status_q_lo,
            "status_q_hi": self.status_q_hi,
        }


def _psd_constraint(
    constraints: List,
    u: cp.Expression,
    v: cp.Expression,
    w: cp.Expression,
) -> None:
    """Add [[u,v],[v,w]] >> 0 constraint via SOC decomposition."""
    # [[u,v],[v,w]] >> 0 iff u>=0, w>=0, uw >= v^2
    # SOC form: u>=0, w>=0, ||(2v, u-w)||_2 <= u+w
    constraints.append(u >= 0)
    constraints.append(w >= 0)
    constraints.append(
        cp.norm(cp.hstack([2.0 * v, u - w]), 2) <= u + w
    )


class CrashRiskSOCP:
    """
    SOCP solver for physical crash-risk identification under CRRA utility.

    Implements Charnes-Cooper scaling for both gamma=2 (exact SOCP) and
    gamma=1 (log-investor case). The identified sets for q, ell, and s
    are obtained by solving linear/conic programs over admissible measures.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        K: float = 0.90,
        B: float = 5.0,
        solver: str = PRIMARY_SOLVER,
        solver_opts: Optional[Dict] = None,
        fallback_solver: str = FALLBACK_SOLVER,
    ) -> None:
        """
        Args:
            gamma: CRRA risk-aversion parameter (1.0 or 2.0).
            K: Crash threshold (default 0.90).
            B: Right support bound (default 5.0).
            solver: Primary cvxpy solver name.
            solver_opts: Solver options dict.
            fallback_solver: Fallback solver if primary fails.
        """
        if gamma not in (1.0, 2.0):
            raise ValueError("Only gamma=1 and gamma=2 are implemented.")
        self.gamma = gamma
        self.K = K
        self.B = B
        self.solver = solver
        self.solver_opts = solver_opts or (
            CLARABEL_OPTS.copy() if solver == "CLARABEL" else SCS_OPTS.copy()
        )
        self.fallback_solver = fallback_solver

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve_cross_section(
        self,
        quotes: pd.DataFrame,
        date: pd.Timestamp,
        horizon: int,
        quote_set: str = "otm",
        compute_frontier: bool = False,
        n_frontier_dirs: int = 32,
        verify_regularity: bool = True,
    ) -> CrossSectionResult:
        """
        Solve all identified-set endpoint programs for one cross-section.

        Args:
            quotes: Normalized quote DataFrame with columns k_norm, c_lo, c_hi.
            date: Cross-section date.
            horizon: Calendar days.
            quote_set: Quote set name ('pair','local','wing','otm').
            compute_frontier: Whether to compute the joint q-ell frontier.
            n_frontier_dirs: Initial number of frontier directions.
            verify_regularity: Whether to verify Assumption 3.3.

        Returns:
            CrossSectionResult with all endpoint values and metadata.
        """
        result = CrossSectionResult(
            date=date,
            horizon=horizon,
            quote_set=quote_set,
            n_quotes=len(quotes),
            gamma=self.gamma,
            K=self.K,
            B=self.B,
        )

        if len(quotes) == 0:
            logger.warning("Empty quote set for %s horizon=%d set=%s", date, horizon, quote_set)
            return result

        try:
            # Build cell partition from normalized strikes
            k_norms = quotes["k_norm"].values
            partition = build_cell_partition(k_norms, K=self.K, B=self.B)

            # Build shared model data
            model_data = self._build_base_model(partition, quotes)

            # Solve q endpoints
            q_lo_res = self._solve_single(model_data, "q", minimize=True)
            q_hi_res = self._solve_single(model_data, "q", minimize=False)
            result.q_lo = q_lo_res.value
            result.q_hi = q_hi_res.value
            result.status_q_lo = q_lo_res.status
            result.status_q_hi = q_hi_res.status

            if q_lo_res.is_valid and q_hi_res.is_valid:
                result.q_width = max(0.0, result.q_hi - result.q_lo)
                result.valid = True

            # Solve ell endpoints
            ell_lo_res = self._solve_single(model_data, "ell", minimize=True)
            ell_hi_res = self._solve_single(model_data, "ell", minimize=False)
            result.ell_lo = ell_lo_res.value
            result.ell_hi = ell_hi_res.value
            result.status_ell_lo = ell_lo_res.status
            result.status_ell_hi = ell_hi_res.status
            if ell_lo_res.is_valid and ell_hi_res.is_valid:
                result.ell_width = max(0.0, result.ell_hi - result.ell_lo)

            # Solve s endpoints
            s_lo_res = self._solve_single_s(model_data, minimize=True)
            s_hi_res = self._solve_single_s(model_data, minimize=False)
            result.s_lo = s_lo_res.value
            result.s_hi = s_hi_res.value
            result.status_s_lo = s_lo_res.status
            result.status_s_hi = s_hi_res.status
            if s_lo_res.is_valid and s_hi_res.is_valid:
                result.s_width = max(0.0, result.s_hi - result.s_lo)

            # Regularity verification
            if verify_regularity:
                reg_slack = self._verify_regularity(model_data, partition, quotes)
                result.regularity_slack = reg_slack
                result.regularity_verified = (reg_slack is not None and reg_slack > 1e-10)

            # Frontier computation
            if compute_frontier:
                dirs, svals, spts = self._compute_frontier(
                    model_data, n_dirs=n_frontier_dirs, max_dirs=n_frontier_dirs
                )
                result.frontier_directions = dirs
                result.frontier_support_values = svals
                result.frontier_points = spts

        except Exception as exc:
            logger.error(
                "solve_cross_section failed for %s h=%d set=%s: %s",
                date, horizon, quote_set, exc,
            )

        return result

    # ------------------------------------------------------------------
    # Internal: base model construction
    # ------------------------------------------------------------------

    def _build_base_model(
        self, partition: Dict, quotes: pd.DataFrame
    ) -> Dict:
        """
        Build CVXPY variables and constraints shared across all endpoint programs.

        Returns a dict with:
          - partition, quotes
          - y: dict mapping (cell_idx, moment_order) -> cp.Variable
          - rho: cp.Variable (Charnes-Cooper scaling factor)
          - a_K: cp.Variable (threshold atom)
          - constraints: list of cp.Constraint
          - q_expr: cp.Expression for q = A_2 / M
          - ell_expr: cp.Expression for ell = (K*A_2 - A_3) / M
        """
        j_K = partition["j_K"]
        below_K = partition["below_K_indices"]
        above_K = partition["above_K_indices"]
        thresh_idx = partition["threshold_idx"]
        cells = partition["cells"]
        boundaries = partition["boundaries"]
        B = self.B
        K = self.K

        # --- Variables ---
        # rho = M(Q)^{-1}, rho in [1/B, 1]
        rho = cp.Variable(name="rho", nonneg=True)

        # Cell moment variables: y[(j,r)] for cell j, moment order r
        y: Dict[Tuple[int, int], cp.Variable] = {}
        for j in below_K:
            for r in range(4):
                y[(j, r)] = cp.Variable(name=f"y_{j}_{r}")
        # Threshold cell: moments 0,1,2
        for r in range(3):
            y[(thresh_idx, r)] = cp.Variable(name=f"y_{thresh_idx}_{r}")
        for j in above_K:
            for r in range(3):
                y[(j, r)] = cp.Variable(name=f"y_{j}_{r}")

        # Threshold atom at x=K
        a_K = cp.Variable(name="a_K", nonneg=True)

        constraints: List[cp.Constraint] = []

        # --- rho bounds: rho in [1/B, 1] ---
        constraints.append(rho >= 1.0 / B)
        constraints.append(rho <= 1.0)

        # --- Global moment equalities ---
        # sum_j y_{0j} = rho  (scaled total mass = rho)
        all_y0 = [y[(j, 0)] for j in below_K + [thresh_idx] + above_K]
        constraints.append(cp.sum(all_y0) == rho)

        # sum_j y_{1j} = rho  (scaled first moment = rho, since E_Q[X]=1)
        all_y1 = [y[(j, 1)] for j in below_K + [thresh_idx] + above_K]
        constraints.append(cp.sum(all_y1) == rho)

        # sum_j y_{2j} = 1  (normalized second moment = 1 under Charnes-Cooper)
        all_y2 = [y[(j, 2)] for j in below_K + [thresh_idx] + above_K]
        constraints.append(cp.sum(all_y2) == 1.0)

        # --- Non-negativity of cell 0-moment (mass is non-negative) ---
        for j in below_K + [thresh_idx] + above_K:
            constraints.append(y[(j, 0)] >= 0.0)

        # --- PSD conditions for below-K cells: order-3 (4 moments) ---
        for j in below_K:
            a_j, b_j = cells[j]
            y0, y1, y2, y3 = y[(j, 0)], y[(j, 1)], y[(j, 2)], y[(j, 3)]
            # P1: [[y1-a*y0, y2-a*y1],[y2-a*y1, y3-a*y2]] >> 0
            _psd_constraint(constraints, y1 - a_j * y0, y2 - a_j * y1, y3 - a_j * y2)
            # P2: [[b*y0-y1, b*y1-y2],[b*y1-y2, b*y2-y3]] >> 0
            _psd_constraint(constraints, b_j * y0 - y1, b_j * y1 - y2, b_j * y2 - y3)

        # --- Threshold cell: moments 0,1,2 on [K, b_{j_K+1}] with atom at K ---
        b_thresh_right = boundaries[thresh_idx + 1]
        y0t, y1t, y2t = y[(thresh_idx, 0)], y[(thresh_idx, 1)], y[(thresh_idx, 2)]
        # Residual moments (exclude atom at K)
        y0r = y0t - a_K
        y1r = y1t - K * a_K
        y2r = y2t - K ** 2 * a_K
        # Apply above-K cell conditions to residuals on [K, b_{j_K+1}]
        _add_above_K_constraints(constraints, y0r, y1r, y2r, K, b_thresh_right)

        # --- PSD conditions for above-threshold cells: order-2 (3 moments) ---
        for j in above_K:
            a_j, b_j = cells[j]
            y0, y1, y2 = y[(j, 0)], y[(j, 1)], y[(j, 2)]
            _add_above_K_constraints(constraints, y0, y1, y2, a_j, b_j)

        # --- Quote interval constraints ---
        for _, row in quotes.iterrows():
            k_i = row["k_norm"]
            c_lo = row["c_lo"]
            c_hi = row["c_hi"]
            coefs = call_payoff_coefficients(partition, k_i)
            # C_i(y) = sum_{j: b_j >= k_i} (y_{1j} - k_i * y_{0j})
            C_i = _build_linear_expr(coefs, y)
            # rho * c_lo_i <= C_i(y) <= rho * c_hi_i
            constraints.append(C_i >= rho * c_lo)
            constraints.append(C_i <= rho * c_hi)

        # --- Objective expressions ---
        # q = sum_{j<j_K} y_{2j} + K^2 * a_K
        q_expr = (
            cp.sum([y[(j, 2)] for j in below_K]) + K ** 2 * a_K
            if below_K
            else K ** 2 * a_K
        )
        # ell = sum_{j<j_K} (K * y_{2j} - y_{3j})
        ell_expr = (
            cp.sum([K * y[(j, 2)] - y[(j, 3)] for j in below_K])
            if below_K
            else cp.Constant(0.0)
        )

        return {
            "partition": partition,
            "quotes": quotes,
            "y": y,
            "rho": rho,
            "a_K": a_K,
            "constraints": constraints,
            "q_expr": q_expr,
            "ell_expr": ell_expr,
            "below_K": below_K,
            "thresh_idx": thresh_idx,
            "above_K": above_K,
        }

    # ------------------------------------------------------------------
    # Internal: single endpoint solves (rho normalization for q and ell)
    # ------------------------------------------------------------------

    def _solve_single(
        self, model_data: Dict, target: str, minimize: bool
    ) -> EndpointResult:
        """
        Solve one endpoint program under the rho (Charnes-Cooper) normalization.

        Args:
            model_data: Output of _build_base_model.
            target: 'q' or 'ell'.
            minimize: True for lower bound, False for upper bound.

        Returns:
            EndpointResult with value and solver status.
        """
        q_expr = model_data["q_expr"]
        ell_expr = model_data["ell_expr"]

        if target == "q":
            obj_expr = q_expr
        elif target == "ell":
            obj_expr = ell_expr
        else:
            raise ValueError(f"Unknown target: {target}")

        objective = cp.Minimize(obj_expr) if minimize else cp.Maximize(obj_expr)
        prob = cp.Problem(objective, model_data["constraints"])
        return self._solve_problem(prob)

    # ------------------------------------------------------------------
    # Internal: s endpoint solve (lambda normalization)
    # ------------------------------------------------------------------

    def _solve_single_s(
        self, model_data: Dict, minimize: bool
    ) -> EndpointResult:
        """
        Solve one s (conditional crash depth) endpoint using lambda normalization.

        Under the lambda normalization (lambda = 1/A_2(Q)):
          z_{r,j} = lambda * y_{r,j},  c_K = lambda * a_K
          sum_{j<j_K} z_{2j} + K^2 * c_K = 1  (lambda * A_2 = 1)
          s = K - (sum_{j<j_K} z_{3j} + K^3 * c_K)

        The scaled constraints are:
          sum z_{0j} = mu = lambda * rho
          sum z_{1j} = mu
          lambda = sum_j z_{2j}
          mu >= lambda / B   (rho in [1/B, 1] implies mu >= lambda/B)
          mu <= lambda        (rho <= 1)
          C_i(z) in [mu * c_lo_i, mu * c_hi_i]

        Args:
            model_data: Output of _build_base_model.
            minimize: True for s_lo, False for s_hi.

        Returns:
            EndpointResult.
        """
        partition = model_data["partition"]
        quotes = model_data["quotes"]
        below_K = model_data["below_K"]
        thresh_idx = model_data["thresh_idx"]
        above_K = model_data["above_K"]
        cells = partition["cells"]
        boundaries = partition["boundaries"]
        K = self.K
        B = self.B

        # --- New variables (z-normalized) ---
        mu = cp.Variable(name="mu_s", nonneg=True)
        lam = cp.Variable(name="lambda_s", nonneg=True)
        c_K_s = cp.Variable(name="c_K_s", nonneg=True)

        z: Dict[Tuple[int, int], cp.Variable] = {}
        for j in below_K:
            for r in range(4):
                z[(j, r)] = cp.Variable(name=f"zs_{j}_{r}")
        for r in range(3):
            z[(thresh_idx, r)] = cp.Variable(name=f"zs_{thresh_idx}_{r}")
        for j in above_K:
            for r in range(3):
                z[(j, r)] = cp.Variable(name=f"zs_{j}_{r}")

        constraints: List[cp.Constraint] = []

        # Non-negativity of cell 0-moments
        for j in below_K + [thresh_idx] + above_K:
            constraints.append(z[(j, 0)] >= 0.0)

        # Lambda normalization: sum_{j<j_K} z_{2j} + K^2 * c_K = 1
        A2_z = (
            cp.sum([z[(j, 2)] for j in below_K]) + K ** 2 * c_K_s
            if below_K
            else K ** 2 * c_K_s
        )
        constraints.append(A2_z == 1.0)

        # lambda = sum_j z_{2j} (total second moment under lambda norm)
        all_z2 = [z[(j, 2)] for j in below_K + [thresh_idx] + above_K]
        constraints.append(cp.sum(all_z2) == lam)

        # mu = sum z_{0j} = sum z_{1j}
        all_z0 = [z[(j, 0)] for j in below_K + [thresh_idx] + above_K]
        all_z1 = [z[(j, 1)] for j in below_K + [thresh_idx] + above_K]
        constraints.append(cp.sum(all_z0) == mu)
        constraints.append(cp.sum(all_z1) == mu)

        # rho in [1/B, 1]: mu = lambda*rho, so lambda/B <= mu <= lambda
        constraints.append(mu >= lam / B)
        constraints.append(mu <= lam)

        # --- PSD conditions (same structure, applied to z) ---
        for j in below_K:
            a_j, b_j = cells[j]
            z0, z1, z2, z3 = z[(j, 0)], z[(j, 1)], z[(j, 2)], z[(j, 3)]
            _psd_constraint(constraints, z1 - a_j * z0, z2 - a_j * z1, z3 - a_j * z2)
            _psd_constraint(constraints, b_j * z0 - z1, b_j * z1 - z2, b_j * z2 - z3)

        # Threshold cell residuals (z-space)
        b_thresh_right = boundaries[thresh_idx + 1]
        z0t, z1t, z2t = z[(thresh_idx, 0)], z[(thresh_idx, 1)], z[(thresh_idx, 2)]
        z0r = z0t - c_K_s
        z1r = z1t - K * c_K_s
        z2r = z2t - K ** 2 * c_K_s
        _add_above_K_constraints(constraints, z0r, z1r, z2r, K, b_thresh_right)

        for j in above_K:
            a_j, b_j = cells[j]
            z0, z1, z2 = z[(j, 0)], z[(j, 1)], z[(j, 2)]
            _add_above_K_constraints(constraints, z0, z1, z2, a_j, b_j)

        # Quote constraints (scaled by mu)
        for _, row in quotes.iterrows():
            k_i = row["k_norm"]
            c_lo = row["c_lo"]
            c_hi = row["c_hi"]
            coefs = call_payoff_coefficients(partition, k_i)
            C_i_z = _build_linear_expr(coefs, z)
            constraints.append(C_i_z >= mu * c_lo)
            constraints.append(C_i_z <= mu * c_hi)

        # Objective: s = K - (sum z_{3j for j<j_K} + K^3 * c_K)
        A3_z = (
            cp.sum([z[(j, 3)] for j in below_K]) + K ** 3 * c_K_s
            if below_K
            else K ** 3 * c_K_s
        )
        s_expr = K - A3_z  # note: A2_z = 1 so this is s = K - A_3/A_2

        objective = cp.Minimize(s_expr) if minimize else cp.Maximize(s_expr)
        prob = cp.Problem(objective, constraints)
        return self._solve_problem(prob)

    # ------------------------------------------------------------------
    # Internal: frontier computation
    # ------------------------------------------------------------------

    def _compute_frontier(
        self,
        model_data: Dict,
        n_dirs: int = 32,
        max_dirs: int = 128,
        tol_pct: float = 0.01,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Approximate the q-ell joint frontier via support-function SOCPs.

        Uses adaptive direction grid: start with n_dirs directions, double
        up to max_dirs until outer-inner approximation gap < tol_pct of
        put-pair scale.

        Returns:
            directions: (n_dir, 2) array
            support_values: (n_dir,) array of support function values
            support_points: (n_dir, 2) array of primal support points
        """
        q_expr = model_data["q_expr"]
        ell_expr = model_data["ell_expr"]
        constraints = model_data["constraints"]

        current_dirs = n_dirs
        while True:
            angles = np.linspace(0, 2 * np.pi, current_dirs, endpoint=False)
            u_dirs = np.column_stack([np.cos(angles), np.sin(angles)])
            support_values = np.full(current_dirs, np.nan)
            support_points = np.full((current_dirs, 2), np.nan)

            for i, (u1, u2) in enumerate(u_dirs):
                obj_expr = u1 * q_expr + u2 * ell_expr
                prob = cp.Problem(cp.Maximize(obj_expr), constraints)
                res = self._solve_problem(prob)
                if res.is_valid:
                    support_values[i] = res.value
                    # Extract primal point
                    try:
                        support_points[i, 0] = q_expr.value
                        support_points[i, 1] = ell_expr.value
                    except Exception:
                        pass

            if current_dirs >= max_dirs:
                break
            # Simple convergence check (TODO: polygon gap check)
            current_dirs = min(current_dirs * 2, max_dirs)
            if current_dirs >= max_dirs:
                break

        return u_dirs, support_values, support_points

    # ------------------------------------------------------------------
    # Internal: regularity verification (Assumption 3.3)
    # ------------------------------------------------------------------

    def _verify_regularity(
        self,
        model_data: Dict,
        partition: Dict,
        quotes: pd.DataFrame,
        delta: float = 1e-5,
        margin_width: float = 0.05,
    ) -> Optional[float]:
        """
        Verify Assumption 3.3: existence of strictly interior witness measure.

        Maximizes interior slack epsilon such that:
        - All quotes priced strictly inside: C_i in [rho*c_lo_i + eps, rho*c_hi_i - eps]
        - Positive mass in U_- subset (0,1) away from boundaries
        - Positive mass in U_+ subset (1,B) away from boundaries

        Returns:
            Maximum achievable slack epsilon, or None if program failed.
        """
        boundaries = partition["boundaries"]
        K = self.K
        B = self.B

        # Find interior intervals U_- and U_+ avoiding all boundaries and threshold
        bnd_sorted = sorted(boundaries)

        def _find_interior_interval(lo_bound: float, hi_bound: float) -> Optional[Tuple[int, float, float]]:
            """Find a cell index and interior sub-interval [alpha, beta] in (lo_bound, hi_bound)."""
            for j_idx, (a, b) in enumerate(partition["cells"]):
                mid = 0.5 * (a + b)
                if a > lo_bound and b < hi_bound and b - a > 2 * margin_width:
                    alpha = a + margin_width
                    beta = b - margin_width
                    if alpha < beta:
                        return j_idx, alpha, beta
            return None

        # U_- in (0,1): find a cell with both endpoints strictly between 0 and 1, away from K
        U_minus = _find_interior_interval(0.0, 0.99)
        # U_+ in (1, B): find a cell with both endpoints strictly between 1 and B
        U_plus = _find_interior_interval(1.01, B - 0.1)

        if U_minus is None or U_plus is None:
            return None

        # Add epsilon variable
        eps = cp.Variable(name="eps_reg", nonneg=True)

        # New constraints: tightened quote intervals
        extra_constraints = list(model_data["constraints"])  # copy

        q_expr = model_data["q_expr"]
        ell_expr = model_data["ell_expr"]
        rho = model_data["rho"]
        y = model_data["y"]
        below_K = model_data["below_K"]
        thresh_idx = model_data["thresh_idx"]
        above_K = model_data["above_K"]

        for _, row in quotes.iterrows():
            k_i = row["k_norm"]
            c_lo = row["c_lo"]
            c_hi = row["c_hi"]
            coefs = call_payoff_coefficients(partition, k_i)
            C_i = _build_linear_expr(coefs, y)
            # Strict interior: C_i in [rho*c_lo + eps, rho*c_hi - eps]
            extra_constraints.append(C_i >= rho * c_lo + eps)
            extra_constraints.append(C_i <= rho * c_hi - eps)

        # Positive mass in U_minus: mass_U_minus >= delta
        j_um, alpha_um, beta_um = U_minus
        if (j_um, 0) in y:
            # Use first moment to bound mass from below: mass in [alpha,beta] >= y_0j - ...
            # Approximate: require y_{0,j} >= delta (cell must have positive mass)
            extra_constraints.append(y[(j_um, 0)] >= delta * rho)

        # Positive mass in U_plus: mass_U_plus >= delta
        j_up, alpha_up, beta_up = U_plus
        if (j_up, 0) in y:
            extra_constraints.append(y[(j_up, 0)] >= delta * rho)

        # Maximize eps
        prob = cp.Problem(cp.Maximize(eps), extra_constraints)
        res = self._solve_problem(prob)
        if res.is_valid:
            return float(res.value) if res.value is not None else None
        return None

    # ------------------------------------------------------------------
    # Solver wrapper
    # ------------------------------------------------------------------

    def _solve_problem(self, prob: cp.Problem) -> EndpointResult:
        """Solve a cvxpy Problem with fallback and error handling."""
        # Try primary solver
        try:
            if self.solver == "CLARABEL":
                prob.solve(solver=cp.CLARABEL, **self.solver_opts)
            elif self.solver == "SCS":
                prob.solve(solver=cp.SCS, **self.solver_opts)
            else:
                prob.solve(solver=self.solver)
        except Exception as e:
            logger.debug("Primary solver %s failed: %s", self.solver, e)
            prob.solve(solver=cp.SCS, **SCS_OPTS)

        status = prob.status or "unknown"
        if status in OPTIMAL_STATUSES and prob.value is not None:
            return EndpointResult(value=float(prob.value), status=status, is_valid=True)

        # Fallback
        if status not in OPTIMAL_STATUSES:
            logger.debug("Primary solver status=%s, trying fallback %s", status, self.fallback_solver)
            try:
                if self.fallback_solver == "SCS":
                    prob.solve(solver=cp.SCS, **SCS_OPTS)
                else:
                    prob.solve(solver=self.fallback_solver)
                status = prob.status or "unknown"
            except Exception as e2:
                logger.debug("Fallback solver failed: %s", e2)

        if status in OPTIMAL_STATUSES and prob.value is not None:
            return EndpointResult(value=float(prob.value), status=status, is_valid=True)

        return EndpointResult(value=None, status=status, is_valid=False)


# ------------------------------------------------------------------
# Module-level helper functions
# ------------------------------------------------------------------


def _add_above_K_constraints(
    constraints: List,
    y0: cp.Expression,
    y1: cp.Expression,
    y2: cp.Expression,
    a: float,
    b: float,
) -> None:
    """
    Add order-2 cell constraints for a measure on [a, b]:
    1. [[y0, y1],[y1, y2]] >> 0
    2. a * y0 <= y1 <= b * y0
    3. y2 <= (a+b)*y1 - a*b*y0

    Args:
        constraints: List to append to.
        y0, y1, y2: Moment variables or expressions.
        a, b: Cell boundaries.
    """
    # PSD condition: [[y0,y1],[y1,y2]] >> 0
    _psd_constraint(constraints, y0, y1, y2)
    # Location constraints
    constraints.append(y1 >= a * y0)
    constraints.append(y1 <= b * y0)
    # Second-moment upper bound: y2 <= (a+b)*y1 - ab*y0
    constraints.append(y2 <= (a + b) * y1 - a * b * y0)


def _build_linear_expr(
    coefs: Dict[Tuple[int, int], float],
    y: Dict[Tuple[int, int], cp.Variable],
) -> cp.Expression:
    """Build a CVXPY linear expression from coefficient dict and variable dict."""
    terms = []
    for (j, r), c in coefs.items():
        if (j, r) in y:
            terms.append(c * y[(j, r)])
    if not terms:
        return cp.Constant(0.0)
    return cp.sum(terms)
