# Safeguarded Gibbs-dual feasibility wrapper

Code and data for the paper's numerical experiments.

> **Scope.** This implementation does not aim to be an efficient solver. It is mainly a
> proof of concept for the instances considered in the paper, and it shows how the
> number of iterations (oracle calls) is reduced compared with Hamiltonian Updates when
> the proposals exploit the curvature of the Gibbs dual. Every oracle call is a dense
> eigendecomposition. An efficient general implementation (for example exploiting
> sparsity or using randomized trace estimation) is left as future work.

```
safeguarded_sdp/            the method: Gibbs dual, safeguarded wrapper, L-BFGS proposal
                            rules, calibration of the binary search
benchmark/
  experiments/
    precision_scaling.py      synthetic QUBO-SDPs, N = 16 .. 1024
    logistics.py              logistics QUBOs (ASP, OVRP) of Ostermann et al.
    portfolio.py              portfolio optimisation, a general-constraint SDP
    solvers.py                safeguarded solver, Hamiltonian Updates, the binary searches
    common.py                 grids, sequential run loop, CSV output
    diag_dual.py, synthetic.py, qubo_sdp.py, portfolio_instances.py, lift.py
                              problem instances and their Gibbs duals
  make_figures.py           figures and tables from the results
  figures/                  the figures (PDF, PNG) and tables (CSV) it produces
  results/                  results of the three experiments (CSV)
  data/                     logistics QUBOs and reference optima, portfolio returns
  third_party/              Hamiltonian Updates of Henze et al. and of Ostermann et al.
  hpc/                      optional: the same experiments in parallel on a many-core machine
LICENSE                     MIT License (safeguarded_sdp/ and benchmark/)
NOTICE                      third-party code and data, and their licences
requirements.txt
```

## Installation

Python >= 3.10 (tested with 3.12):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

All data is included; nothing is downloaded.

## Figures and tables

```bash
python benchmark/make_figures.py
```

reads `benchmark/results/` and writes, in a few seconds, `benchmark/figures/` (PDF and
PNG) and `benchmark/figures/tables/` (CSV):

| output | content |
|---|---|
| `precision_N128_hu_vs_lbfgs` | oracle calls vs. precision at N = 128: Hamiltonian Updates and the safeguarded solver |
| `precision_qubo_all_N` | safeguarded solver, oracle calls vs. precision for N = 16 .. 1024 |
| `dimension_qubo` | safeguarded solver, oracle calls vs. N at fixed precision |
| `precision_portfolio` | portfolio, oracle calls vs. precision for n = 6 .. 14 assets |
| `dimension_portfolio` | portfolio, oracle calls vs. SDP dimension at fixed precision |
| `tables/precision_exponents.csv` | fitted exponent a in calls ~ eps^a, per solver and N |
| `tables/logistics_table.csv` | logistics: relative error of the lower bound, oracle calls, time |
| `tables/portfolio_table.csv` | portfolio: the safeguarded solver's threshold, certified interval, interior-point value, calls, time |

## The searches

**The safeguarded solver** (`lbfgs_scipy`) decides each feasibility query with the safeguarded
wrapper (`safeguarded_sdp/wrapper.py`) and the scipy L-BFGS-B proposal
(`ScipyLBFGSProposal`), inside a calibrated binary search that
returns a certified interval of width at most eps for the optimum
(`cert_lo`, `cert_hi` in the results):

* synthetic and logistics SDPs (uniform diagonal, `rho_ii = 1/N`, zero-diagonal cost
  matrix): bracket width eps/2 and query tolerance eta = eps / (2 (N + 2)), i.e. the
  sensitivity constant kappa_sens = N (`safeguarded_sdp/reduction.py`);
* portfolio: query tolerance eta = eps / (2 R_tr Gamma_port), with R_tr the trace bound of
  the lift and Gamma_port the sensitivity constant computed from the dual multipliers of
  the lifted SDP (solved by Clarabel); bracket width eps/2.

Only precisions whose query tolerance is at least 1e-7 are run (`--min-eta`): below
that, double precision no longer resolves the decrease of the Gibbs dual in the line
search. The smallest precision therefore grows with N (7.7e-6 at N = 16, 4.6e-4 at
N = 1024).

**Hamiltonian Updates** (`hu`) uses Henze et al.'s protocol: bisection on [-1, 1] to width
eps, every query at tolerance eps. Henze et al.'s code is used for the synthetic SDPs,
the copy shipped with Ostermann et al.'s benchmark (which produced their published
numbers) for the logistics SDPs; both are included unmodified (see `NOTICE`).

**Interior point** (`clarabel`): the lifted portfolio SDP solved by Clarabel via CVXPY,
for comparison.

## Running the experiments

```bash
python benchmark/experiments/precision_scaling.py
python benchmark/experiments/logistics.py --problem asp
python benchmark/experiments/logistics.py --problem vrp
python benchmark/experiments/portfolio.py
```

Each script runs its binary searches one after another, prints one line per
feasibility query, and appends the results to `benchmark/results/<experiment>/`.
Searches already in the results are skipped, so an interrupted run continues where
it stopped; to recompute everything, move `benchmark/results/` away first, or write
elsewhere with `--out DIR` (and `python benchmark/make_figures.py --results DIR`).
`--help` lists each script's options; the defaults are the grids used in the paper.
A quick subset:

```bash
python benchmark/experiments/precision_scaling.py --k 4-6 --seeds 0-2 --out test_run
```

The full grids take about 4 CPU-hours (the sum of `wall_s` in the results): about
3 hours for Hamiltonian Updates, mostly N = 128 at eps = 1e-4, and about 1 hour for the
safeguarded solver, mostly N = 1024. The optional `benchmark/hpc/` runs the same searches in
parallel (`bash benchmark/hpc/run_all.sh`); the results do not depend on how the
searches are scheduled.

## Results

`benchmark/results/<experiment>/summary.csv` has one row per binary search and
`queries.csv` one row per feasibility query (joined on `run_id`):

| column | meaning |
|---|---|
| `solver` | `lbfgs_scipy` (safeguarded solver), `hu` (Hamiltonian Updates), `clarabel` (interior point, portfolio) |
| `eps` | target precision of the search |
| `lo`, `hi`, `estimate` | final bracket of the threshold and its midpoint |
| `cert_lo`, `cert_hi` | safeguarded solver: certified interval containing the optimum |
| `n_oracle` | oracle calls: eigendecompositions (safeguarded L-BFGS) or matrix exponentials including overshoots (HU) |
| `n_queries`, `n_iter` | feasibility queries; iterations (safeguarded L-BFGS) or updates (HU) |
| `n_uncertified_queries`, `n_fallback` | queries without a certified verdict (0 everywhere); fallback steps of the wrapper |
| `wall_s` | wall time (s) |
| `k`, `N`, `seed` | synthetic instance (N = 2^k) |
| `instance`, `lower_bound`, `optimum`, `rel_error` | logistics instance; QUBO lower bound from `hi`; `rel_error = abs(LB - opt) / abs(opt)` |
| `n`, `D`, `m_constraints`, `R_tr`, `gamma_port` | portfolio: assets, SDP dimension, constraints, trace bound, sensitivity constant |
| `mu_star`, `mu_ipm`, `p_star_socp` | portfolio: the safeguarded solver's threshold, the interior-point value of the same SDP, the SOCP optimum |

`queries.csv` adds, per query, the threshold `mu`, the tolerance `query_eps`, the
`verdict` (`feasible` or `infeasible`, both certified), its `reason`, and the query's
`n_oracle`. For the logistics HU searches only the per-search rows are available.

## Notes

* The comparison is in oracle calls. Instances are generated from fixed seeds. The
  results were computed with one BLAS thread per search
  (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`).
* `benchmark/experiments/diag_dual.py` is a memory-efficient form of the package's
  `GibbsDualObjective` for uniform-diagonal SDPs (the general class would need 309 GB at
  N = 3381); the two agree to machine precision.
* The logistics matrices are gzipped CSV files; the first run caches them as `.npy` in
  `benchmark/data/logistics/cache/`.
* Every certified interval was checked against a classical solver: the exact optimum
  (SCS) for the synthetic SDPs up to N = 128, MOSEK's SDP bounds for the logistics
  instances (Ostermann et al., `data/logistics/reference/`), Clarabel for the portfolio.
