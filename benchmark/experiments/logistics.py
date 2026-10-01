#!/usr/bin/env python3
"""
Experiment 2 -- real-world logistics QUBOs: the affinity-based slotting (ASP)
and open vehicle routing (OVRP) instances of Ostermann et al.

Each QUBO is homogenised into the uniform-diagonal SDP (qubo_sdp.py, verbatim from
Ostermann et al., so bounds are on their published scale), then the SDP optimum is found
to precision eps: safeguarded L-BFGS with the calibrated binary search (certified
interval, queries at tolerance eps / (2 (N + 2))), HU with Henze et al.'s protocol
(bisection to width eps, queries at tolerance eps). The lower bound on the QUBO optimum
uses the upper end hi of the final bracket (for safeguarded L-BFGS a certified-infeasible
threshold):

    LB = convert_back(N * hi),    rel_error = |LB - opt| / |opt|

Safeguarded L-BFGS runs at the eps of 1e-2 .. 1e-6 (ASP) and 1e-2 .. 1e-4 (OVRP) whose
calibrated tolerance is at least 1e-7 (--min-eta): 1e-2 .. 1e-4 (ASP), 1e-2, 1e-3 (OVRP).
HU, with the HU code shipped with Ostermann et al.'s benchmark, runs at eps = 1e-2, 1e-3
(ASP) and 1e-2 (OVRP).

    python benchmark/experiments/logistics.py --problem asp
    python benchmark/experiments/logistics.py --problem vrp      # N = 1751 and 3381: slower
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np

import common
import solvers
from diag_dual import DiagSDPDual
from qubo_sdp import convert, convert_back, get_const

DATA_DIR = os.path.join(common.DATA, "logistics")
EXTRA_FIELDS = ["problem", "instance", "size", "penalty", "N", "cert_lo", "cert_hi",
                "lower_bound", "optimum", "abs_gap", "rel_error"]


def instance_files(problem, names=""):
    """The gzipped QUBO matrices of one problem (or the named ones)."""
    folder = os.path.join(DATA_DIR, problem)
    if names:
        return [os.path.join(folder, n.strip() + ".csv.gz") for n in names.split(",")]
    return sorted(glob.glob(os.path.join(folder, "*.csv.gz")))


def name_of(path):
    return os.path.basename(path)[: -len(".csv.gz")]


def load_qubo(path):
    """Parse the gzipped CSV once and keep a binary .npy copy in benchmark/data/logistics/cache/."""
    cache = os.path.join(DATA_DIR, "cache", name_of(path) + ".npy")
    if not os.path.exists(cache):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.save(cache, np.loadtxt(path, delimiter=","))
    return np.load(cache)


def optima():
    with open(os.path.join(DATA_DIR, "reference", "optima.csv")) as fh:
        return {r["size"]: float(r["optimum"]) for r in csv.DictReader(fh)}


def searches(args):
    opt = optima()
    out = []
    for path in instance_files(args.problem, args.instances):
        name = name_of(path)
        size = "_".join(name.split("_")[:2])                          # e.g. ASP_n30
        penalty = name.split("_p")[1].split("_")[0]
        N = load_qubo(path).shape[0] + 1                               # SDP dimension
        for solver in args.solvers.split(","):
            safeguarded = solver in solvers.SAFEGUARDED_SOLVERS
            for eps in common.eps_list(args.lbfgs_eps if safeguarded else args.hu_eps):
                if safeguarded and eps / (2 * (N + 2)) < args.min_eta:          # calibrated eta
                    continue
                out.append({
                    "key": common.search_key(solver, name, eps), "solver": solver, "eps": eps,
                    "instance": {"problem": args.problem, "instance": name, "size": size,
                                 "penalty": penalty},
                    "path": path, "optimum": opt.get(size, float("nan")),
                    "hu_impl": args.hu_impl, "hu_b": args.hu_b,
                    "cost": os.path.getsize(path) * (eps ** -1.6 if not safeguarded else 0.05 * eps ** -0.35),
                })
    return out


def prepare(specs):
    """Parse every instance once before parallel runs (hpc/runner.py)."""
    for path in sorted({s["path"] for s in specs}):
        load_qubo(path)


def run_search(spec, log=None):
    C, info = convert(load_qubo(spec["path"]))
    info["c"] -= get_const(name_of(spec["path"]) + ".csv")     # constant offset in the file name
    N = C.shape[0]
    eps = spec["eps"]
    extra = {}
    if spec["solver"] == "hu":
        kwargs = {"b": spec["hu_b"]} if spec["hu_b"] > 0 else {}  # bits of HU's CNOT-count model
        lo, hi, records = solvers.binary_search(
            lambda mu: solvers.hu(C, mu, eps, impl=spec["hu_impl"], **kwargs), -1.0, 1.0, eps, log=log)
    else:
        kappa, S = solvers.uniform_diagonal_constants(C)
        lo, hi, records, (extra["cert_lo"], extra["cert_hi"]) = solvers.calibrated_search(
            lambda mu, eta: solvers.safeguarded_lbfgs(DiagSDPDual(C, mu), eta),
            eps, kappa, S, log=log)

    opt = spec["optimum"]
    lb = convert_back(N * hi, info)
    extra.update(lower_bound=lb, optimum=opt, abs_gap=lb - opt, rel_error=abs(lb - opt) / abs(opt))
    return {"lo": lo, "hi": hi, "records": records, "instance": {"N": N}, "extra": extra}


def parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--problem", choices=["asp", "vrp"], default="asp")
    p.add_argument("--instances", default="", help="comma list of instance names (default: all)")
    p.add_argument("--solvers", default="lbfgs_scipy,hu",
                   help="comma list of lbfgs_scipy (safeguarded L-BFGS), hu")
    p.add_argument("--lbfgs-eps", default=None,
                   help="default 1e-2,1e-3,1e-4,1e-5,1e-6 (asp) / 1e-2,1e-3,1e-4 (vrp)")
    p.add_argument("--hu-eps", default=None, help="default 1e-2,1e-3 (asp) / 1e-2 (vrp)")
    p.add_argument("--hu-impl", choices=["ostermann", "henze"], default="ostermann")
    p.add_argument("--hu-b", type=int, default=8, help="bits of HU's CNOT-count model (0 = off)")
    p.add_argument("--min-eta", type=float, default=common.MIN_ETA,
                   help="safeguarded L-BFGS: run only the eps whose calibrated query tolerance "
                        "eps / (2 (N + 2)) is at least this (double-precision limit)")
    p.add_argument("--out", default=None,
                   help="results directory (default benchmark/results/logistics_<problem>)")
    p.add_argument("--quiet", action="store_true", help="no line per query")
    return p


def parse(argv=None):
    a = parser().parse_args(argv)
    a.lbfgs_eps = a.lbfgs_eps or ("1e-2,1e-3,1e-4,1e-5,1e-6" if a.problem == "asp" else "1e-2,1e-3,1e-4")
    a.hu_eps = a.hu_eps or ("1e-2,1e-3" if a.problem == "asp" else "1e-2")
    return a


if __name__ == "__main__":
    a = parse()
    common.run_searches(f"logistics_{a.problem}", searches(a), run_search, EXTRA_FIELDS,
                        a.out, a.quiet)
