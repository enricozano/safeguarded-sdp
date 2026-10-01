#!/usr/bin/env python3
"""
Experiment 1 -- precision and dimension scaling: safeguarded L-BFGS vs. Hamiltonian
Updates (Henze et al.) on synthetic uniform-diagonal QUBO-SDPs of dimension N = 2^k.

One search finds the SDP optimum gamma* to precision eps. Safeguarded L-BFGS uses the
calibrated binary search (certified interval of width eps, queries at tolerance
eps / (2 (N + 2))); HU uses Henze et al.'s protocol (bisection on [-1, 1] to width eps,
every query at tolerance eps). Defaults are the grid used in the paper:

    safeguarded L-BFGS : k = 4..10, eps from 10 log-spaced values 1e-2 .. 1e-6, those
                         whose calibrated tolerance is at least 1e-7 (--min-eta)
    HU                 : k = 4..7, eps = 6 log-spaced values 1e-2 .. 1e-4
    seeds 0..9, row sparsity 16

    python benchmark/experiments/precision_scaling.py                       # full grid
    python benchmark/experiments/precision_scaling.py --k 4-6 --seeds 0-2   # a quick subset
"""
from __future__ import annotations

import argparse

import common
import solvers
from diag_dual import DiagSDPDual
from synthetic import gen_cost_matrix

EXPERIMENT = "precision_scaling"
EXTRA_FIELDS = ["k", "N", "seed", "sparsity", "cert_lo", "cert_hi"]


def searches(args):
    out = []
    for solver in args.solvers.split(","):
        safeguarded = solver in solvers.SAFEGUARDED_SOLVERS
        for k in common.int_list(args.k if safeguarded else args.hu_k):
            for seed in common.int_list(args.seeds):
                for eps in common.eps_list(args.lbfgs_eps if safeguarded else args.hu_eps):
                    if safeguarded and eps / (2 * (2 ** k + 2)) < args.min_eta:     # calibrated eta
                        continue
                    out.append({
                        "key": common.search_key(solver, f"k={k}|seed={seed}", eps),
                        "solver": solver, "eps": eps,
                        "instance": {"k": k, "N": 2 ** k, "seed": seed, "sparsity": args.sparsity},
                        # expected relative cost, only used to schedule parallel runs (hpc/)
                        "cost": 8 ** k * (eps ** -1.6 if solver == "hu" else 0.05 * eps ** -0.35),
                    })
    return out


def run_search(spec, log=None):
    inst = spec["instance"]
    C = gen_cost_matrix(inst["k"], inst["sparsity"], inst["seed"])
    eps = spec["eps"]
    if spec["solver"] == "hu":
        lo, hi, records = solvers.binary_search(lambda mu: solvers.hu(C, mu, eps, impl="henze"),
                                                -1.0, 1.0, eps, log=log)
        return {"lo": lo, "hi": hi, "records": records}
    kappa, S = solvers.uniform_diagonal_constants(C)
    lo, hi, records, (cert_lo, cert_hi) = solvers.calibrated_search(
        lambda mu, eta: solvers.safeguarded_lbfgs(DiagSDPDual(C, mu), eta),
        eps, kappa, S, log=log)
    return {"lo": lo, "hi": hi, "records": records,
            "extra": {"cert_lo": cert_lo, "cert_hi": cert_hi}}


def parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k", default="4-10", help="log2 N for safeguarded L-BFGS, e.g. 4-10 or 4,6")
    p.add_argument("--hu-k", default="4-7", help="log2 N for HU")
    p.add_argument("--seeds", default="0-9")
    p.add_argument("--solvers", default="lbfgs_scipy,hu",
                   help="comma list of lbfgs_scipy (safeguarded L-BFGS), hu")
    p.add_argument("--lbfgs-eps", default="1e-2:1e-6:10", help="'a:b:n' = n log-spaced values")
    p.add_argument("--hu-eps", default="1e-2:1e-4:6")
    p.add_argument("--sparsity", type=int, default=16)
    p.add_argument("--min-eta", type=float, default=common.MIN_ETA,
                   help="safeguarded L-BFGS: run only the eps whose calibrated query tolerance "
                        "eps / (2 (N + 2)) is at least this (double-precision limit)")
    p.add_argument("--out", default=None,
                   help="results directory (default benchmark/results/precision_scaling)")
    p.add_argument("--quiet", action="store_true", help="no line per query")
    return p


if __name__ == "__main__":
    a = parser().parse_args()
    common.run_searches(EXPERIMENT, searches(a), run_search, EXTRA_FIELDS, a.out, a.quiet)
