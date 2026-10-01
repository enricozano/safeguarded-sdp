#!/usr/bin/env python3
"""
Experiment 3 -- portfolio optimisation as a general-constraint SDP.

Instances (portfolio_instances.py): n assets drawn (by seed) from 174 liquid US large caps,
first T = 2n daily returns. The mean-risk SOCP is lifted exactly to an SDP (lift.py:
D = 5n+3, m = 2n^2+5n+2 dense constraints), and the minimal cost mu* is found by binary
search over the cost threshold mu. Each query is decided by safeguarded L-BFGS on the
package's general GibbsDualObjective.

Calibration: with the sensitivity constant
    Gamma_port = beta_obj + sum_j |y_j*| beta_j
(y* the equality-row dual multipliers of the lifted SDP, beta the row normalisations),
an accepted threshold satisfies mu >= mu* - R_tr Gamma_port eta. Every query runs at
eta = eps / (2 R_tr Gamma_port) and the bracket is refined to width eps / 2, so
mu* lies in the certified interval [lo, hi + eps / 2] of width at most eps. y* is taken
from the Clarabel solution of the same lifted SDP.

Reference searches (solver = clarabel, one per (n, seed)) solve the same lifted SDP in
one shot with the interior-point solver Clarabel (via CVXPY), plus the original SOCP (p*).

    python benchmark/experiments/portfolio.py        # n = 6..14, seed 0, eps 1e-2..1e-5
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

import common
import solvers
from lift import build_lifted_sdp
from portfolio_instances import (load_returns_csv, make_instance, portfolio_objective,
                                 solve_portfolio_reference)

from gibbs_dual import GibbsDualObjective  # noqa: E402  (safeguarded_sdp, put on sys.path by solvers)

RETURNS = os.path.join(common.DATA, "portfolio", "returns_liquid_us.csv")
EXTRA_FIELDS = ["n", "seed", "T", "D", "m_constraints", "R_tr", "gamma_port",
                "cert_lo", "cert_hi", "mu_star", "p_star_socp", "mu_ipm", "ipm_wall_s", "ipm_status"]
MAX_EXPAND = 8


def searches(args):
    out = []
    for n in common.int_list(args.n):
        for seed in common.int_list(args.seeds):
            inst = {"n": n, "seed": seed, "T": 2 * n}
            out.append({"key": f"clarabel|n={n}|seed={seed}", "solver": "clarabel", "eps": "",
                        "instance": inst, "cost": float(n) ** 6})
            lift, y = lifted_sdp_ipm(instance_of({"instance": inst}))[3:]
            scale = 2.0 * lift.R * gamma_port(lift, y)              # eta = eps / scale
            for eps in common.eps_list(args.eps):
                if eps / scale < args.min_eta:
                    continue
                out.append({"key": common.search_key(args.solver, f"n={n}|seed={seed}", eps),
                            "solver": args.solver, "eps": eps, "instance": inst,
                            "cost": float(n) ** 6 * eps ** -0.6})
    return out


def instance_of(spec):
    return make_instance(n=spec["instance"]["n"], seed=spec["instance"]["seed"],
                         returns=load_returns_csv(RETURNS))


def lifted_sdp_ipm(inst):
    """The lifted SDP solved by Clarabel. Returns (optimum, wall time, status, lift, y)
    with y[i] = |dual multiplier| of equality row i (indexed like lift.names)."""
    import cvxpy as cp

    lift = build_lifted_sdp(inst, mu=0.0)
    Y = cp.Variable((lift.D, lift.D), symmetric=True)
    rows = [i for i in range(lift.socp.K)] + [i for i, n in enumerate(lift.names) if n.startswith("risk_")]
    eq = {i: cp.sum(cp.multiply(lift.raw_matrices[i], Y)) == lift.raw_targets[i] for i in rows}
    obj_mat = lift.raw_matrices[lift.names.index("objective_threshold")].copy()
    u = int(lift.positions["u"])
    obj_mat[u, u] -= 1.0                        # strip the slack, leaving the objective
    prob = cp.Problem(cp.Minimize(cp.sum(cp.multiply(obj_mat, Y))),
                      [Y >> 0, cp.trace(Y) <= lift.R] + list(eq.values()))
    t0 = time.perf_counter()
    prob.solve(solver="CLARABEL")
    wall = time.perf_counter() - t0
    y = {i: abs(float(c.dual_value)) for i, c in eq.items()}
    return float(prob.value), wall, str(prob.status), lift, y


def gamma_port(lift, y):
    """Gamma_port = beta_obj + sum_j |y_j*| beta_j, with beta the lift's row normalisation."""
    beta = lift.scales
    return float(beta[lift.names.index("objective_threshold")] + sum(v * beta[i] for i, v in y.items()))


def run_search(spec, log=None):
    return run_reference(spec) if spec["solver"] == "clarabel" else run_ours(spec, log)


def run_ours(spec, log=None):
    inst = instance_of(spec)
    eps = spec["eps"]
    y = lifted_sdp_ipm(inst)[4]
    dims = {}

    def query(mu):
        lift = build_lifted_sdp(inst, mu=mu)
        gamma = gamma_port(lift, y)
        eta = eps / (2.0 * lift.R * gamma)
        dims.update(D=lift.D, m_constraints=lift.d, R_tr=lift.R, gamma_port=gamma)
        rec = solvers.safeguarded_lbfgs(GibbsDualObjective(list(lift.observables), lift.targets), eta)
        rec["query_eps"] = eta
        return rec

    hi = float(portfolio_objective(inst, inst.w_bar))       # cost of a feasible portfolio
    lo = float(-np.max(inst.u_hat)) - 1.0                   # below any achievable cost
    records = []
    for _ in range(MAX_EXPAND):                             # make sure lo really is infeasible
        records.append(solvers.timed_query(query, lo, lo, hi, "expand", log, len(records) + 1))
        if not records[-1]["feasible"]:
            break
        hi, lo = lo, lo - 2.0 * (hi - lo)
    lo, hi, records = solvers.binary_search(query, lo, hi, eps / 2.0, feasible_moves="hi", log=log,
                                            records=records)
    return {"lo": lo, "hi": hi, "records": records, "instance": dims,
            "extra": {"mu_star": hi, "cert_lo": lo, "cert_hi": hi + eps / 2.0}}


def run_reference(spec):
    """The lifted SDP solved directly by Clarabel (same feasible set the search explores)."""
    inst = instance_of(spec)
    p_star, _, _ = solve_portfolio_reference(inst)
    mu_ipm, wall, status, lift, y = lifted_sdp_ipm(inst)
    return {"instance": {"D": lift.D, "m_constraints": lift.d, "R_tr": lift.R,
                         "gamma_port": gamma_port(lift, y)},
            "extra": {"p_star_socp": p_star, "mu_ipm": mu_ipm, "estimate": mu_ipm,
                      "ipm_wall_s": wall, "ipm_status": status}}


def parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", default="6,8,10,12,14", help="numbers of assets")
    p.add_argument("--seeds", default="0", help="the paper uses seed 0")
    p.add_argument("--eps", default="1e-2,1e-3,1e-4,1e-5")
    p.add_argument("--solver", default="lbfgs_scipy", choices=solvers.SAFEGUARDED_SOLVERS)
    p.add_argument("--min-eta", type=float, default=common.MIN_ETA,
                   help="run only the eps whose calibrated query tolerance is at least this")
    p.add_argument("--out", default=None, help="results directory (default benchmark/results/portfolio)")
    p.add_argument("--quiet", action="store_true", help="no line per query")
    return p


if __name__ == "__main__":
    a = parser().parse_args()
    common.run_searches("portfolio", searches(a), run_search, EXTRA_FIELDS, a.out, a.quiet)
