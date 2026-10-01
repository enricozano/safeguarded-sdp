"""
The solvers compared in the paper, each as one feasibility query, and the binary
searches that turn feasibility queries into the optimum of an SDP.

    safeguarded_lbfgs(problem, eta)  safeguarded wrapper + L-BFGS proposal
    hu(C, mu, eps)                   Hamiltonian Updates (vendored, unmodified)
    calibrated_search(...)           calibrated binary search (safeguarded solver):
                                     certified interval of width eps for the optimum
    binary_search(...)               bisection until the bracket is narrower than a
                                     width; with width = tolerance = eps it is Henze
                                     et al.'s protocol (HU)

Every query returns a record (dict) with its verdict and its cost in oracle calls.
"""
from __future__ import annotations

import contextlib
import importlib
import io
import os
import sys
import time

BENCHMARK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(BENCHMARK), "safeguarded_sdp"))

import numpy as np  # noqa: E402

from proposals import ScipyLBFGSProposal  # noqa: E402
from reduction import calibrate, kappa_sens_uniform_diagonal, s_obj  # noqa: E402
from wrapper import decide_feasibility  # noqa: E402

# safeguarded L-BFGS: the safeguarded wrapper with the scipy L-BFGS-B proposal (ScipyLBFGSProposal)
SAFEGUARDED_SOLVERS = ("lbfgs_scipy",)

_VERDICT = {"eps_feasible": "feasible", "infeasible_certificate": "infeasible"}


def safeguarded_lbfgs(problem, eps, mem=10, max_iter=50000):
    """One feasibility query at residual tolerance eps: returns an eps-feasible Gibbs state
    or a certificate of infeasibility. `problem` is a Gibbs dual
    (safeguarded_sdp.gibbs_dual.GibbsDualObjective or experiments/diag_dual.py)."""
    # one propose() = one L-BFGS-B run down to (a hair under) the wrapper's stopping
    # threshold 15*eps/16, keeping its curvature memory for the whole run
    proposal = ScipyLBFGSProposal(maxiter=max_iter, maxfun=10 * max_iter, ftol=0.0,
                                  gtol=0.9 * eps, maxcor=mem)
    r = decide_feasibility(problem, eps, proposal, max_iter=max_iter)
    return {
        "feasible": bool(r["feasible"]),
        "verdict": _VERDICT.get(r["reason"], "maxiter"),
        "reason": r["reason"],
        "n_oracle": int(r["n_gibbs"]), "n_iter": int(r["n_iter"]),
        "n_prop_attempts": int(r["n_prop_attempts"]),
        "n_prop_accepted": int(r["n_prop_accepted"]),
        "n_fallback": int(r["n_fallback"]),
        "f_min": float(r["f_min"]), "grad_inf": float(r["grad_inf"]),
        "U_final": float(r["U_final"]), "ham_op_norm": float(r["ham_op_norm"]),
    }


# ----------------------------------------------------------------------------- HU

HU_DIRS = {"henze": os.path.join(BENCHMARK, "third_party", "hu_henze"),        # Henze et al.
           "ostermann": os.path.join(BENCHMARK, "third_party", "hu_ostermann")}  # Ostermann et al.
HU_BETA = {"henze": 0.5, "ostermann": 0.4}      # their built-in momentum parameters
_hu_loaded: dict = {}


def _hamiltonian_update(impl):
    """Import the vendored HU function. Both copies are packaged as a top-level module
    named `src`, so a process can use only one of them."""
    if impl not in _hu_loaded:
        if _hu_loaded:
            raise RuntimeError("only one HU implementation per process")
        sys.path.insert(0, HU_DIRS[impl])
        with contextlib.redirect_stdout(io.StringIO()):          # silence its import banner
            _hu_loaded[impl] = importlib.import_module("src.hamiltonian_updates").hamiltonian_update
    return _hu_loaded[impl]


def hu(C, mu, eps, impl="henze", **kwargs):
    """One HU feasibility query. Oracle calls = matrix exponentials = updates + overshoots."""
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            accepted, H, t, F, qc, logs = _hamiltonian_update(impl)(C, mu, eps, **kwargs)
        except Exception as exc:                                 # raised when HU hits T_max
            if "maximum number of iterations" not in str(exc):
                raise
            return {"feasible": False, "verdict": "maxiter", "reason": "hu_T_max"}
    overshoots = int(logs.get("overshot_c", 0)) + int(logs.get("overshot_d", 0))
    return {
        "feasible": bool(accepted),
        "verdict": "feasible" if accepted else "infeasible",
        "reason": "hu_eps_feasible" if accepted else "hu_free_energy_positive",
        "n_oracle": int(t) + overshoots, "n_iter": int(t), "hu_overshoots": overshoots,
    }


# ----------------------------------------------------------------------------- search

def uniform_diagonal_constants(C):
    """(kappa_sens, S_obj) of the normalised uniform-diagonal SDP max Tr(C rho),
    rho_ii = 1/N: kappa_sens <= N if every C_ii >= 0, else 2N."""
    kappa = kappa_sens_uniform_diagonal(C.shape[0], cii_nonneg=bool(np.all(np.diag(C) >= 0)))
    return kappa, s_obj(1.0, float(np.linalg.norm(C, 2)))


def calibrated_search(query, eps, kappa_sens, S_obj=1.0, log=None):
    """Calibrated binary search for a normalised maximisation threshold on [-1, 1].

    Bracket width w = eps / (2 S_obj); every query at tolerance
    eta = eps / (2 S_obj (kappa_sens + 2)). `query(mu, eta)` returns a record. Returns
    (lo, hi, records, (cert_lo, cert_hi)): the optimum lies in the certified interval
    [S_obj (lo - (kappa_sens + 2) eta), S_obj hi], of width at most eps."""
    w, eta = calibrate(eps, kappa_sens, S_obj)

    def query_eta(mu):
        rec = query(mu, eta)
        rec["query_eps"] = eta
        return rec

    lo, hi, records = binary_search(query_eta, -1.0, 1.0, w, feasible_moves="lo", log=log)
    return lo, hi, records, (S_obj * (lo - (kappa_sens + 2.0) * eta), S_obj * hi)


def binary_search(query, lo, hi, eps, feasible_moves="lo", log=None, records=None):
    """Bisection on [lo, hi] until hi - lo <= eps. `query(mu)` returns a record.

    feasible_moves='lo': maximisation threshold (a feasible mu raises the lower end);
    feasible_moves='hi': minimisation threshold (a feasible mu lowers the upper end).
    Returns (lo, hi, records), one record per query, with mu and timing added; new
    records are appended to `records` if given."""
    records = [] if records is None else records
    while hi - lo > eps:
        mu = 0.5 * (lo + hi)
        records.append(timed_query(query, mu, lo, hi, "bisect", log, len(records) + 1))
        if records[-1]["feasible"] == (feasible_moves == "lo"):
            lo = mu
        else:
            hi = mu
    return lo, hi, records


def timed_query(query, mu, lo, hi, phase, log, q):
    t0 = time.perf_counter()
    rec = query(mu)
    rec.update(q=q, phase=phase, mu=mu, lo_before=lo, hi_before=hi,
               wall_s=round(time.perf_counter() - t0, 4))
    if log is not None:
        log(f"  q{q:>2} {phase:6s} mu={mu:+.6f} {rec['verdict']:10s} "
            f"oracle={rec.get('n_oracle', '')!s:>7} {rec['wall_s']:8.2f}s")
    return rec


def totals(records):
    """Summary columns of one binary search from its query records."""
    s = {k: sum(int(r.get(k) or 0) for r in records)
         for k in ("n_oracle", "n_iter", "n_prop_attempts", "n_prop_accepted", "n_fallback")}
    s["n_queries"] = len(records)
    s["n_uncertified_queries"] = sum(r["verdict"] not in ("feasible", "infeasible") for r in records)
    return s
