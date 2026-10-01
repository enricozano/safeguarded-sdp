# safeguarded_sdp

A standalone, minimal implementation of a safeguarded, single-trajectory
Gibbs-dual feasibility wrapper, together with a calibrated binary search that
turns repeated feasibility queries into a certified interval for an SDP's
optimum.

The package (`gibbs_dual.py`, `wrapper.py`, `proposals.py`, `reduction.py`,
`instances.py`) is built from scratch and has no external
dependency beyond numpy/scipy. The experiments of the paper (`../benchmark/`) use
it as their solver; see the top-level `README.md`.

## Structure

```
safeguarded_sdp/
    README.md
    gibbs_dual.py           GibbsDualObjective: general, O(N^3)-per-call Gibbs dual
    wrapper.py              decide_feasibility(problem, eps, proposal, ...): the feasibility wrapper
    proposals.py            Proposal (ABC), ScipyLBFGSProposal, NoProposal
    reduction.py            calibration: eps_opt -> (w, eta), calibrated binary search, kappa_sens helper
    instances.py            Uniform-diagonal QUBO-SDP instance builder + synthetic cost-matrix generator
```

There is no `__init__.py`/package wrapper: the modules are flat and meant to
be imported with this directory on `sys.path` (as
`../benchmark/experiments/solvers.py` does).

## Core modules

1. **`gibbs_dual.GibbsDualObjective`** implements the Gibbs dual
   `F_b(lambda) = log Tr exp(-sum_i lambda_i A_i) + lambda^T b` and its
   gradient `b - a(rho_lambda)` for arbitrary Hermitian observables
   `A_1..A_m`. This is the exact/noiseless oracle: it does not model
   simulated oracle noise (finite tolerance / failure probability).

2. **`wrapper.decide_feasibility`** is the feasibility wrapper itself:
   given one such dual object and a requested accuracy `eps`, it runs the
   safeguarded single-trajectory loop (maintaining the certified upper
   bound `U`, an acceptance test for candidates, and a certified-decrease
   fallback), returning either an eps-feasible Gibbs parameter or a negative
   dual certificate. The *proposal rule* `A(lambda)` (what candidate to try)
   is a pluggable object, keeping the arbitrary proposal rule cleanly
   separated from the fixed certification wrapper. The wrapper switches
   permanently to the certified fallback after the first rejected proposal.

3. **`proposals.py`** defines the proposal-rule interface as an abstract
   base class, `Proposal` (Python's `abc.ABC`, the closest analogue to a
   C++-style abstract class: `propose()` is `@abstractmethod`, everything
   else has a no-op/True default, and instantiating `Proposal` itself
   raises `TypeError`). Two concrete subclasses are supplied:
   `ScipyLBFGSProposal`, an L-BFGS proposal delegating the quasi-Newton
   direction and line search to `scipy.optimize.minimize(method="L-BFGS-B")`,
   and the trivial `NoProposal`, with which the wrapper runs its certified
   Hamiltonian Update fallback only. Any other proposal rule
   (Newton, a problem-specific heuristic, ...) just needs to subclass
   `Proposal` and implement `propose`.

4. **`instances.py`** builds a concrete feasibility instance: the direct
   normalized uniform-diagonal SDP family (the QUBO relaxation
   `max Tr(C rho)` s.t. `rho >= 0`, `Tr(rho)=1`, `rho_ii = 1/N`), encoded as
   an observable list/target for `GibbsDualObjective`.

5. **`reduction.py`** is the layer that turns a single feasibility oracle
   into a full *optimization* result: a calibrated binary search over the
   threshold `mu`, which computes the correct `(w, eta)` allocation from a
   target `eps_opt` and a sensitivity bound `kappa_sens` (not a single
   shared tolerance -- see "Design notes" below), and returns a certified
   interval `[L, U]` (rescaled by `S_obj`) that is guaranteed to contain the
   true SDP optimum `V*`.

Conceptually: **dual -> wrapper (with pluggable proposal) -> reduction
(repeated calibrated feasibility queries) -> certified optimum interval.**

## Design notes

- **Efficient dual evaluation.** A naive generalization of the Gibbs dual
  loops over the `m` observables and does a dense `N x N` matmul per
  observable to get `Tr(rho A_i)`, which is `O(N^4)` per call.
  `gibbs_dual.GibbsDualObjective` instead forms the density matrix
  `rho = (U * p) @ U.T` **once** after the single `eigh` call (`O(N^3)`,
  no extra asymptotic cost over the eigendecomposition itself), then
  obtains every `Tr(rho A_i)` via one batched `einsum` contraction over the
  stacked observables (`O(m*N^2)`, cheap) -- one general dual class, no
  separate "diagonal-only fast path" needed.
- **Two-tolerance calibration.** A plain bisection over the SDP threshold
  might reuse a single shared `eps` as both the outer bracket-stopping
  width and the inner feasibility oracle's tolerance, which silently
  assumes `kappa_sens ~= 0`. `reduction.py` instead implements the correct
  allocation: `w = eps_opt/(2*S_obj)`,
  `eta = eps_opt/(2*S_obj*(kappa_sens+2))`, with a helper
  (`kappa_sens_uniform_diagonal`) for this instance family's
  `kappa_sens <= 2d` bound (tightening to `<= d` if `C_ii >= 0`).

## Scope and known limitations

- **Exact oracles only.** `GibbsDualObjective` is the noiseless oracle. No
  simulated oracle noise (finite tolerance/failure probability) is
  implemented anywhere in this package.
- **Dense `eigh`-based dual evaluation.** Every oracle call does one dense
  `numpy.linalg.eigh` (`O(N^3)`), not a certified sparse matrix-function
  evaluation. This package targets small-to-medium dense instances, not an
  asymptotic large-scale cost model.
- **One L-BFGS proposal implementation** is supplied (`ScipyLBFGSProposal`),
  plus the trivial no-proposal rule. No other proposal rule (e.g. Newton, a
  problem-specific heuristic) is implemented, but the `Proposal` abstract
  base class in `proposals.py` makes adding one a matter of subclassing it
  and implementing `propose`.
