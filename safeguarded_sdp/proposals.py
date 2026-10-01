"""
Pluggable proposal rules A(lambda) for wrapper.decide_feasibility: an
arbitrary resource-admissible candidate, checked via the wrapper's own
acceptance test -- the wrapper never has to trust the proposal rule's own
convergence theory.

Rules are ordinary Python classes deriving from the abstract base class
`Proposal` below (Python doesn't need an abstract class for duck-typed
interfaces like this one -- `wrapper.decide_feasibility` only ever calls
methods by name -- but subclassing `Proposal` gets you an explicit contract,
`isinstance` checks, and a `TypeError` at construction time if a required
method is missing, which is closer to the C++-style abstract-base-class
habit than pure duck typing).

Two concrete rules are implemented: `ScipyLBFGSProposal` (an L-BFGS
proposal delegating the quasi-Newton direction and line search to
`scipy.optimize.minimize`) and `NoProposal` (never propose: the wrapper then
runs its certified Hamiltonian Update fallback only). Any other proposal rule
(Newton, a problem-specific heuristic, ...) just needs to subclass `Proposal`
and implement `propose`.

Interface every `Proposal` subclass must expose:

    reset()
        Clear all internal optimizer state (secant-pair history, or
        whatever else the rule tracks). Called once by
        wrapper.decide_feasibility at the start of every call.
        Default: no-op.

    can_propose() -> bool
        False only for NoProposal: the wrapper then never calls
        propose/on_accept/on_reject for this object and always goes
        straight to the fallback step. Default: True.

    propose(lam, f, g, evaluate) -> (lam_trial, f_trial, g_trial)
        Attempt ONE candidate from the current certified iterate (lam, f, g).
        `evaluate` is the wrapper's own oracle-call closure: every oracle
        touch (including line-search probes) must go through it so they are
        counted in n_gibbs and so infeasibility is detected correctly.
        Abstract: every subclass must implement this.

    on_accept(lam_old, g_old, lam_new, g_new)
        Called exactly when a proposed candidate is accepted, so the
        proposal can update its own state (e.g. L-BFGS secant pairs).
        Default: no-op.

    on_reject()
        Called when a proposed candidate is rejected. Default: no-op.

    on_fallback(lam_old, g_old, lam_new, g_new)
        Called after every fallback step, so state that should track the
        realized trajectory (e.g. the
        L-BFGS secant history) stays consistent even when a step came from
        the fixed fallback rule rather than from propose(). Default: no-op.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Proposal(ABC):
    """Abstract base class for proposal rules A(lambda). See the module
    docstring for the full interface contract. Every method except
    `propose` has a sensible no-op/True default, so a minimal concrete
    subclass only needs to implement `propose`."""

    def reset(self):
        pass

    def can_propose(self) -> bool:
        return True

    @abstractmethod
    def propose(self, lam, f, g, evaluate):
        """Attempt ONE candidate from (lam, f, g); return (lam_trial,
        f_trial, g_trial). Must route every oracle touch through
        `evaluate`."""
        raise NotImplementedError

    def on_accept(self, lam_old, g_old, lam_new, g_new):
        pass

    def on_reject(self):
        pass

    def on_fallback(self, lam_old, g_old, lam_new, g_new):
        pass


class ScipyLBFGSProposal(Proposal):
    """L-BFGS proposal rule delegating the quasi-Newton direction and line
    search to `scipy.optimize.minimize(..., method="L-BFGS-B")`.

    Each `propose()` call runs one bounded scipy L-BFGS-B local search from
    the current certified iterate (`lam`), capped at `maxiter`/`maxfun`
    inner steps and stopped once the gradient's inf-norm is at most `gtol`,
    and offers the point it returns as a single candidate for the wrapper's
    acceptance test. Every point scipy evaluates goes through `evaluate`, so
    all of them are counted in `n_gibbs` and infeasibility is still detected
    mid-search; neither scipy's starting point (already evaluated by the
    wrapper) nor the returned point is evaluated a second time. Set `gtol` a little below the
    wrapper's stopping threshold (15*eps/16) and `ftol=0` to let one call run
    to the wrapper's accuracy with its curvature memory intact.

    Requires scipy.
    """

    def __init__(self, maxiter=20, maxfun=50, ftol=0.0, gtol=0.0, maxcor=10):
        self.maxiter = maxiter
        self.maxfun = maxfun
        self.ftol = ftol
        self.gtol = gtol
        self.maxcor = maxcor
        self._stalled = False

    def reset(self):
        self._stalled = False

    def can_propose(self):
        return not self._stalled

    def propose(self, lam, f, g, evaluate):
        from collections import OrderedDict
        from scipy.optimize import minimize

        # recent evaluations keyed by the exact point, seeded with the iterate the
        # wrapper has already evaluated (scipy always starts by evaluating x0)
        recent = OrderedDict([(np.asarray(lam, dtype=np.float64).tobytes(), (f, g))])

        def fun(x):
            hit = recent.get(x.tobytes())
            if hit is not None:
                return hit
            fx, gx = evaluate(x)
            recent[x.tobytes()] = (fx, gx)
            if len(recent) > 8:
                recent.popitem(last=False)
            return fx, gx

        result = minimize(
            fun,
            lam,
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": self.maxiter,
                "maxfun": self.maxfun,
                "ftol": self.ftol,
                "gtol": self.gtol,
                "maxcor": self.maxcor,
            },
        )
        lam_new = np.asarray(result.x, dtype=np.float64)
        cached = recent.get(lam_new.tobytes())
        f_new, g_new = cached if cached is not None else evaluate(lam_new)
        if not f_new < f:
            self._stalled = True
        return lam_new, f_new, g_new


class NoProposal(Proposal):
    """No-op proposal rule ("plain Hamiltonian Updates", propose=False):
    can_propose() is always False, so the wrapper never calls
    propose/on_accept/on_reject for this object and every iteration goes
    straight to the fallback step."""

    def can_propose(self):
        return False

    def propose(self, lam, f, g, evaluate):
        raise NotImplementedError(
            "NoProposal.can_propose() is False; the wrapper must never "
            "call propose() on it"
        )
