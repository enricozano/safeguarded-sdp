"""
Optimization calibration: turns a feasibility oracle
(wrapper.decide_feasibility) into a certified interval for the true SDP
optimum V*, via a tolerant binary search that uses TWO distinct tolerances
-- the outer bracket-stopping width w and the inner feasibility oracle's
tolerance eta are NOT the same quantity, unlike a plain bisection that
reuses one shared eps:

    w   = eps_opt / (2*S_obj)
    eta = eps_opt / (2*S_obj*(kappa_sens + 2))

The outer bracket [L, U] stops once U - L <= w; every inner feasibility call
is made at the FIXED tolerance eta (not shrunk across queries). The
certified interval for V* is:

    V* in S_obj * [L - (kappa_sens+2)*eta, U]

Also provides:
  - s_obj(): the general S_obj = R_tr * max(1, ||C||_op) scaling.
  - kappa_sens_uniform_diagonal(): a dimension-dependent sensitivity bound
    for the direct normalized uniform-diagonal SDP family this package
    targets.
"""
from __future__ import annotations

import math
import numpy as np

from gibbs_dual import GibbsDualObjective
from instances import build_diagonal_sdp_instance
from wrapper import decide_feasibility


def s_obj(R_tr: float, C_op_norm: float) -> float:
    """S_obj = R_tr * max(1, ||C||_op). For the already-normalized
    uniform-diagonal family this package targets (Tr(rho)=1 so R_tr=1, and
    ||C||_op <= 1 after normalization), this correctly evaluates to 1 -- but
    is not hardcoded to 1, so an unnormalized instance can be plugged in."""
    return float(R_tr) * max(1.0, float(C_op_norm))


def kappa_sens_uniform_diagonal(d: int, cii_nonneg: bool = False) -> float:
    """kappa_sens bound for the direct normalized uniform-diagonal SDP
    family (max Tr(C rho) s.t. rho>=0, Tr(rho)=1, rho_ii=1/d for i=1..d,
    ||C||_op<=1): an optimal dual is attained with kappa_sens <= 2*d in the
    direct density formulation's projector coordinates.

    This tightens to kappa_sens <= d if C_ii >= 0 for all i (the common
    zero-diagonal-QUBO normalization case), but this helper defaults
    conservatively to 2*d unless the caller explicitly asserts that
    condition via cii_nonneg=True.

    (A trace-filling formulation would instead admit kappa_sens <= 3*d;
    that bound does not apply here since instances.py uses the direct
    formulation, not a trace-filled one.)
    """
    if d <= 0:
        raise ValueError("d must be a positive integer")
    return float(d) if cii_nonneg else 2.0 * float(d)


def calibrate(eps_opt: float, kappa_sens: float, S_obj: float):
    """Given a target optimization accuracy eps_opt and the
    (kappa_sens, S_obj) calibration constants, return (w, eta): the outer
    bracket-stopping width and the inner feasibility oracle's fixed
    tolerance."""
    if eps_opt <= 0:
        raise ValueError("eps_opt must be positive")
    w = eps_opt / (2.0 * S_obj)
    eta = eps_opt / (2.0 * S_obj * (kappa_sens + 2.0))
    return w, eta


def calibrated_binary_search(
    C: np.ndarray,
    eps_opt: float,
    proposal_factory,
    kappa_sens: float | None = None,
    S_obj: float | None = None,
    cii_nonneg: bool = False,
    c: float = 0.5,
    max_iter: int = 50000,
):
    """A tolerant binary-search procedure that turns repeated feasibility
    queries into a certified optimum interval, specialized to the direct
    normalized uniform-diagonal SDP family of instances.py.

    `proposal_factory` is a zero-argument callable that returns a FRESH
    proposals.py instance for every feasibility query (queries must not
    share mutable proposal state; wrapper.decide_feasibility resets its
    proposal each call, but distinct instances keep failure modes from
    leaking across queries in caller code that inspects them afterward).

    If `kappa_sens`/`S_obj` are not given, they default to
    kappa_sens_uniform_diagonal(N, cii_nonneg) and s_obj(1.0, ||C||_op)
    (the defaults appropriate for this instance family).

    Returns a dict: L, U, w, eta, kappa_sens, S_obj, V_star_interval
    (a (lo, hi) tuple bounding S_obj*[L - (kappa_sens+2)*eta, U]),
    n_calls, history (list of (mu, feasible) tuples for every query,
    excluding the initial mu=-1 query, which is always feasible by
    construction and need not be run).
    """
    C = np.asarray(C, dtype=np.float64)
    N = C.shape[0]
    if kappa_sens is None:
        kappa_sens = kappa_sens_uniform_diagonal(N, cii_nonneg=cii_nonneg)
    if S_obj is None:
        S_obj = s_obj(R_tr=1.0, C_op_norm=float(np.linalg.norm(C, ord=2)))

    w, eta = calibrate(eps_opt, kappa_sens, S_obj)

    L, U = -1.0, 1.0  # L=-1 is feasible by construction (no call needed)
    n_calls = 0
    history = []

    max_queries = 2 + math.ceil(math.log2(max(4.0 * S_obj / eps_opt, 2.0)))
    for _ in range(max_queries):
        if U - L <= w:
            break
        mu = 0.5 * (L + U)
        A_list, b = build_diagonal_sdp_instance(C, mu)
        problem = GibbsDualObjective(A_list, b)
        result = decide_feasibility(
            problem, eta, proposal_factory(), c=c, max_iter=max_iter
        )
        n_calls += 1
        feasible = result["feasible"]
        history.append((mu, feasible))
        if feasible:
            L = mu
        else:
            U = mu

    V_lo = S_obj * (L - (kappa_sens + 2.0) * eta)
    V_hi = S_obj * U
    return {
        "L": L,
        "U": U,
        "w": w,
        "eta": eta,
        "kappa_sens": kappa_sens,
        "S_obj": S_obj,
        "V_star_interval": (V_lo, V_hi),
        "n_calls": n_calls,
        "history": history,
    }
