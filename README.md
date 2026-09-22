# Benchmark Identification of Physical Crash Risk in SPX Options

Reproduces the paper's benchmark identification of physical crash-probability intervals,
unconditional shortfall, and conditional crash-depth intervals from finite SPX option
bid/ask quotes using CRRA utility pricing kernels and Second-Order Cone Programming (SOCP).

## Methodology

For each Wednesday SPX option cross-section (Jan 2013 – Aug 2023, exact 23/30/37-day maturities):

1. **Quote normalization**: strike k_i = κ_i / F, call-payoff bounds [c̲_i, c̄_i] via put-call parity, scaled by DF.
2. **Charnes-Cooper transformation**: ρ = M(Q)⁻¹, d̃Q = ρ dQ, converting a nonlinear CRRA transform to linear + SOC constraints.
3. **Cell partition**: support [0, B] split at {0, K, B} ∪ {all retained strikes}, creating bounded sub-cells with truncated Hausdorff moment feasibility.
4. **SOCP endpoints**: minimize/maximize linear functionals of crash probability q_{γ,K}, shortfall ℓ_{γ,K}, and depth s_{γ,K} over the admissible measure set M(𝕀).
5. **Frontier geometry**: adaptive support-function solves over 32–128 directions; outer polygon from intersecting half-planes.

### Parameters

| Symbol | Value |
|--------|-------|
| γ (risk aversion) | 2 |
| K (crash threshold) | 0.90 |
| B (support cap) | 5 |
| Horizons | 23, 30, 37 days |
| Solver | CLARABEL (primary), SCS (fallback) |

## Repository Structure

```
├── data/
│   └── synthetic/          # Empty; place optionmetrics_spx.csv here for real data
├── src/
│   ├── data_loader.py       # Load / parse OptionMetrics CSV; synthetic generator
│   ├── synthetic_data.py    # Black-Scholes synthetic SPX option grid
│   ├── quote_processor.py   # Filtering, normalization, nested quote sets
│   ├── cell_partition.py    # Support cell partitioning + moment PSD constraints
│   ├── socp_solver.py       # SOCP engine (q, ℓ, s endpoints; CLARABEL/SCS)
│   ├── frontier.py          # Joint (q, ℓ) frontier via support functions
│   ├── bootstrap.py         # Circular block bootstrap for exp 2
│   ├── pipeline.py          # End-to-end cross-section pipeline
│   └── utils.py             # Helpers (psd_2x2_constraints, safe_divide, …)
├── exp/
│   ├── exp_1_benchmark.py   # Main benchmark identified-set computation
│   ├── exp_2_quote_density.py # Nested quote-set shrinkage analysis
│   ├── exp_3_tail_support.py  # B=2…10 + unbounded robustness
│   ├── exp_4_kernel_threshold.py # γ=1 vs γ=2; K=0.85 vs K=0.90
│   ├── exp_5_perturbation.py  # Spread doubling / strike thinning / deletions
│   ├── exp_6_geometry.py      # Joint vs marginal geometry, support gaps
│   └── exp_7_regularity_audit.py # Assumption 3.3 verification + solver audit
├── tests/
│   ├── test_data_loader.py
│   ├── test_quote_processor.py
│   ├── test_cell_partition.py
│   └── test_socp_solver.py
├── results/
│   └── RESULTS.md            # Summary tables and figures
├── README.md
├── requirements.txt
└── .gitignore
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run benchmark experiment on synthetic data (fast subset)
python exp/exp_1_benchmark.py --max_cs 60

# Run full experiment pipeline
for i in 1 2 3 4 5 6 7; do
    python exp/exp_${i}_*.py --max_cs 60
done
```

## Using Real OptionMetrics Data

Place the exported CSV at `data/optionmetrics_spx.csv` with columns:

```
date, option_type, strike, maturity_date, bid, ask, volume, D, F
```

Then run experiments **without** the `--max_cs` flag to process all cross-sections.
Expected outputs: ~402, 372, 324 valid cross-sections for 23, 30, 37-day horizons.

## Results Summary

See [`results/RESULTS.md`](results/RESULTS.md) for full tables, figures, and comparison with
paper-expected benchmarks.

Key findings on synthetic data:
- Crash-probability intervals widen monotonically with horizon (23 < 30 < 37 days).
- Lower bounds collapse toward zero under unbounded support (B=∞), confirming Proposition 3.6.
- Joint identified set is strictly smaller than its marginal rectangle in shortfall-positive directions.
- Quote density near the threshold K=0.90 is the primary identification driver.

## Dependencies

See `requirements.txt`. Key packages: `cvxpy`, `clarabel`, `numpy`, `pandas`, `scipy`,
`matplotlib`, `seaborn`, `pytest`.

## Citation

> Benchmark identification of physical crash-risk intervals and frontiers from finite SPX
> option quotes under CRRA utility (γ=2, K=0.90, B=5). This repository implements the full
> SOCP identification approach from the paper's methodology sections.
