"""
The Safeguarded Single-Trajectory Gibbs-Dual Wrapper: a single feasibility
oracle loop that maintains a certified upper bound on the Gibbs-dual
objective, with a pluggable proposal rule A(lambda) (proposals.py). Implements:

  - tau = eps/16
  - U0 = log(N) + 2*tau; at each visited iterate the sign test F_hat + tau < 0 and the
    tightening U <- min(U, F_hat + 3*tau), U never reset upward
  - the eps-feasibility stopping test ||g_hat||_inf <= 15*eps/16
  - K_prop, the global proposal-attempt cap, and the permanent switch to the fallback
    after the first rejected proposal
  - the safeguarded acceptance test for a candidate:
    F_hat(lambda') + tau <= U
  - the certified inexact Hamiltonian Update fallback:
    j = argmax_i |g_hat_i|, r = |g_hat_j| - tau,
    lambda^+ = lambda - (r/4)*sign(g_hat_j)*e_j, U-debit D = r^2/8

Every oracle evaluation, including the auxiliary points a proposal rule evaluates,
returns a certificate as soon as F_hat + tau < 0 (a negative dual value at any lambda
certifies infeasibility); only visited iterates tighten U.

`problem` is anything exposing __call__(lam) -> (f, grad), dim(),
hilbert_dim(), and n_gibbs (attribute) -- see gibbs_dual.GibbsDualObjective.
This module uses the *exact* oracle model: no simulated oracle noise is
implemented (see README "Scope and known limitations").
"""
from __future__ import annotations

import math
import numpy as np


class _Infeasible(Exception):
    """Raised when F_b(lambda) certifies a negative objective."""

    def __init__(self, f, lam):
        self.f = f
        self.lam = lam


def decide_feasibility(problem, eps, proposal, c=0.5, max_iter=50000):
    """Run one feasibility query against `problem` with the proposal rule `proposal`.

    `proposal` must expose the interface documented in `proposals.py`
    (reset, can_propose, propose, on_accept, on_reject, on_fallback). It is
    reset at the start of this call, so the same instance may be reused
    across repeated queries without carrying state between calls.

    The wrapper starts in proposal mode and switches permanently to the
    certified fallback after the first rejected proposal or once K_prop
    proposals have been attempted.

    Returns a dict: feasible, reason ("eps_feasible" | "infeasible_certificate"
    | "maxiter"), n_iter, n_gibbs, n_prop_accepted, n_fallback, f_min,
    grad_inf, ham_op_norm, K_bound, n_prop_attempts, U_final.
    """
    dim = problem.dim()
    N = problem.hilbert_dim()
    b = np.asarray(problem.b, dtype=np.float64)
    tau = eps / 16.0
    K_prop = math.ceil(math.log(N) / (c * eps * eps))  # proposal-attempt cap

    state = {
        "f_min": math.inf,
        "lam_best": np.zeros(dim),
        "U": math.log(N) + 2.0 * tau,  # U0
    }

    def evaluate(x):
        f, g = problem(x)  # one Gibbs-dual oracle call (counted in problem.n_gibbs)
        if f < state["f_min"]:
            state["f_min"] = f
            state["lam_best"] = x.copy()
        if f + tau < 0.0:  # certified negative objective
            raise _Infeasible(f, x.copy())
        return f, g

    def visit(f, x):
        """Tightening of the certified bound U at a visited iterate."""
        state["U"] = min(state["U"], f + 3.0 * tau)
        if state["U"] < 0.0:  # certified objective envelope
            raise _Infeasible(f, x.copy())

    def clip_grad(g):
        # g_hat_i <- b_i - Pi_[-1,1](b_i - g_hat_i). This cannot increase
        # the true residual error because b in [-1,1]^m by construction.
        return b - np.clip(b - g, -1.0, 1.0)

    def debit(D):
        """Certified-decrease bookkeeping: charge the guaranteed decrease D
        to U directly (fixed/shared, not pluggable)."""
        state["U"] -= D
        if state["U"] < 0.0:
            raise _Infeasible(state["f_min"], state["lam_best"].copy())

    lam = np.zeros(dim)
    n_prop_accepted = n_fallback = n_iter = n_prop_attempts = 0
    reason = "maxiter"

    proposal.reset()
    mode = "proposal" if proposal.can_propose() else "safe"

    last_ginf = 0.0
    try:
        f, g = evaluate(lam)
        visit(f, lam)
        for _ in range(max_iter):
            g_hat = clip_grad(g)
            ginf = float(np.max(np.abs(g_hat)))
            last_ginf = ginf
            if ginf <= 15.0 * eps / 16.0:  # eps-feasibility stopping test
                reason = "eps_feasible"
                break

            took_proposal = False
            attempt = (mode == "proposal" and n_prop_attempts < K_prop
                       and proposal.can_propose())
            if attempt:
                n_prop_attempts += 1
                lam_new, f_new, g_new = proposal.propose(lam, f, g, evaluate)
                accept = f_new + tau <= state["U"]  # safeguarded acceptance test

                if accept:  # proposal accepted, U unchanged
                    proposal.on_accept(lam, g, lam_new, g_new)
                    lam, f, g = lam_new, f_new, g_new
                    visit(f, lam)
                    n_prop_accepted += 1
                    took_proposal = True
                else:  # rejected: switch permanently to the fallback
                    proposal.on_reject()
                    mode = "safe"

            if not took_proposal:  # certified Hamiltonian Update fallback
                g_hat = clip_grad(g)
                j = int(np.argmax(np.abs(g_hat)))
                r = abs(float(g_hat[j])) - tau  # > 0 since ginf > 15 eps/16 > tau
                s_sign = 1.0 if g_hat[j] >= 0.0 else -1.0
                lam_new = lam.copy()
                lam_new[j] -= (r / 4.0) * s_sign
                debit(r * r / 8.0)  # D = r^2/8, may raise
                f_new, g_new = evaluate(lam_new)  # may raise _Infeasible
                visit(f_new, lam_new)
                proposal.on_fallback(lam, g, lam_new, g_new)
                lam, f, g = lam_new, f_new, g_new
                n_fallback += 1
            n_iter += 1

    except _Infeasible:
        reason = "infeasible_certificate"

    lam_best = state["lam_best"]
    feasible = (state["f_min"] >= 0.0) if reason != "infeasible_certificate" else False
    ginf_best = last_ginf if reason == "eps_feasible" else 0.0
    return {
        "feasible": feasible,
        "reason": reason,
        "n_iter": n_iter,
        "n_gibbs": problem.n_gibbs,
        "n_prop_accepted": n_prop_accepted,
        "n_fallback": n_fallback,
        "f_min": state["f_min"],
        "grad_inf": ginf_best,
        "ham_op_norm": problem.hamiltonian_op_norm(lam_best),
        "K_bound": K_prop,
        "n_prop_attempts": n_prop_attempts,
        "U_final": state["U"],
    }
