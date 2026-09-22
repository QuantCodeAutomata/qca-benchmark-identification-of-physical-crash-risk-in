# Project: Crash-Risk Identification via SPX Options (AGENTS Memory)

## Overview
Research project identifying crash-risk bounds from SPX option quote intervals via SOCP.
Seven experiment scripts in `exp/` use synthetic fallback data when `data/optionmetrics_spx.csv`
is absent (no real data committed to repo).

## Key Source Files
- `src/socp_solver.py` — core SOCP engine (endpoint solves, frontier, regularity)
- `src/utils.py` — `stratified_head()` for cross-section subsampling
- `exp/exp_1_benchmark.py` through `exp_7_regularity_audit.py` — experiment entry points
- `results/` — all output CSVs and PNGs land here

## Bugs Fixed (completed sessions)

### 1. Clarabel v0.11.x API rename (`src/socp_solver.py` lines 34–41)
Clarabel 0.11.x renamed all tolerance keys. The old keys caused an `AttributeError` inside
`cvxpy` (`'DefaultSettings' object has no attribute 'eps_abs'`), silently falling back to SCS
for every solve (~5× slower).

| Old key (broken) | New key (correct) |
|---|---|
| `eps_abs` | `tol_gap_abs` |
| `eps_rel` | `tol_gap_rel` |
| `eps_prim_inf` | `tol_infeas_abs` |
| `eps_dual_inf` | `tol_infeas_rel` |

Fix applied to `CLARABEL_OPTS` dict. Verify with:
```python
import clarabel; print([a for a in dir(clarabel.DefaultSettings()) if 'tol' in a])
```

### 2. Adaptive frontier doubling in `solve_cross_section` (`src/socp_solver.py` ~line 275)
`_compute_frontier` doubles direction count from `n_dirs` to `max_dirs=128` (default) even
when `n_dirs=32` is passed, tripling frontier solve count. Fixed by passing
`max_dirs=n_frontier_dirs` so the loop runs exactly one pass of `n_dirs` directions.

### 3. `KeyError: 'option_type'` in `exp_5_perturbation.py` (~line 85)
Column name is `"type"` in the synthetic/normalized DataFrame, not `"option_type"`.
Fix: `quotes["option_type"]` → `quotes["type"]` in `drop_six_nearby_puts_keep_every_other()`.

### 4. Matplotlib deprecated API (exp scripts)
Any `plt.tight_layout()` warnings about constrained layout can be ignored; they do not
affect output. If `set_clim` / `cm.ScalarMappable` deprecation warnings appear they are
non-fatal.

## Running the Experiments
All scripts accept `--max_cs N` to cap cross-sections (stratified sample):
```bash
python exp/exp_1_benchmark.py --max_cs 60
python exp/exp_2_quote_density.py --max_cs 60
python exp/exp_3_tail_support.py --max_cs 60
python exp/exp_4_kernel_threshold.py --max_cs 60
python exp/exp_5_perturbation.py --max_cs 60
python exp/exp_6_geometry.py --max_cs 60     # ~10 min with max_cs=60 after fixes
python exp/exp_7_regularity_audit.py --max_cs 60
```

## Approximate Wall-Clock Times (max_cs=60, synthetic data, after fixes)
| Script | Time | Cross-section combos |
|---|---|---|
| exp_1 | ~2.5 min | 240 (60 CS × 4 quote sets) |
| exp_2 | ~7.5 min | 240 × 4 horizons |
| exp_3 | ~11 min | 60 CS × 3 B values |
| exp_4 | ~11 min | 60 CS × 2 kernels + thresholds |
| exp_5 | ~11 min | 60 CS × 3 perturbations |
| exp_6 | ~10 min | 240 CS-qs combos × 38 solves (6 endpoints + 32 frontier dirs) |
| exp_7 | ~50 sec | 60 CS × 3 horizons, regularity verify |

## Expected Results with Synthetic Data
- **Exp 7 regularity**: All 0% verified — synthetic data does not satisfy Assumption 3.3 slack.
  Paper benchmarks on real OptionMetrics: ~405/406, 374/376, 332/333 verified for horizons 23/30/37.
- **Exp 6 frontier**: `pair`, `local`, `wing` quote sets produce NaN support gaps (infeasible
  frontiers on synthetic data); `otm` set converges.
- `optimal_inaccurate` solver status is normal for synthetic data; all bounds still reported.

## Results Directory Contents (after full run, max_cs=60)
```
results/
  exp1_benchmark_results.csv        — per-CS endpoint values
  exp1_benchmark_summary.csv        — median summary by horizon
  exp1_benchmark_q_timeseries.png
  exp2_quote_density_results.csv
  exp2_median_widths.csv
  exp2_width_reductions.csv
  exp2_quote_density_widths.png
  exp3_tail_robustness_results.csv
  exp3_tail_robustness_analysis.csv
  exp3_tail_robustness_q_lo_vs_B.png
  exp4a_kernel_comparison.csv
  exp4a_kernel_summary.csv
  exp4a_kernel_comparison.png
  exp4b_threshold_comparison.csv
  exp4b_threshold_summary.csv
  exp5_perturbation_results.csv
  exp6_geometry_results.csv
  exp6_support_gaps.csv
  exp6_joint_frontiers.png
  exp7_regularity_results.csv
  exp7_regularity_table.csv
  exp7_failed_dates.csv
  exp7_regularity_slack.png
```
