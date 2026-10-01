"""
SOCP -> SDP lift, and the Gibbs-feasibility problem it produces.

The lift is exact, not a relaxation: every SOCP variable is read off a single
PSD matrix Y of size D=3n+m+3, with w/phi/rho/t on the diagonal (so Y >= 0
gives their nonnegativity for free) and eta_j on the off-diagonal Y[t, eta_j].
Two extra constraint families (risk_diag: the eta block's diagonal equals t;
risk_offdiag: its off-diagonals vanish) turn the (t, eta) submatrix into the
arrow matrix [[t, eta^T], [eta, t*I]] >= 0, which is exactly the SOC
||eta||_2 <= t.

include_offdiag=False drops the risk_offdiag rows -- a genuine relaxation
(||eta||_inf <= t instead of ||eta||_2 <= t), kept only for comparison.

The result is normalized into a density matrix rho = Y/R for an a-priori trace
bound R, giving find rho >= 0, Tr(rho) = 1, Tr(A_i rho) = c_i for i = 1..d.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from portfolio_instances import PortfolioInstance, SocpData, build_socp_matrices


@dataclass(frozen=True)
class LiftedSDP:
    observables: np.ndarray
    targets: np.ndarray
    raw_matrices: np.ndarray
    raw_targets: np.ndarray
    scales: np.ndarray
    names: tuple[str, ...]
    R: float
    D: int
    d: int
    n: int
    m: int
    mu: float
    include_offdiag: bool
    positions: dict[str, object]
    socp: SocpData


def _sym_entry(D: int, i: int, j: int, value: float = 1.0) -> np.ndarray:
    mat = np.zeros((D, D))
    if i == j:
        mat[i, i] = value
    else:
        mat[i, j] = 0.5 * value
        mat[j, i] = 0.5 * value
    return mat


def _positions(n: int, m: int) -> dict[str, object]:
    risk_start = 3 * n
    return {"w": np.arange(0, n), "phi": np.arange(n, 2 * n), "rho": np.arange(2 * n, 3 * n),
            "risk_start": risk_start, "t": risk_start,
            "eta": np.arange(risk_start + 1, risk_start + 1 + m),
            "u": 3 * n + m + 1, "tr": 3 * n + m + 2}


def trace_bound_R(instance: PortfolioInstance, mu: float) -> float:
    """An a-priori upper bound on Tr(Y), used to normalise Y into a density matrix."""
    n, m = instance.n, instance.m
    fixed_scalar_trace = 1.0 + 2.0 * float(np.sum(instance.zeta))
    risk_t_bound = max(float(np.linalg.norm(instance.M[:, i])) for i in range(n))
    risk_trace_bound = (m + 1) * risk_t_bound
    objective_slack_bound = (abs(float(mu)) + float(np.linalg.norm(instance.u_hat, ord=np.inf))
                            + instance.q * risk_t_bound)
    return fixed_scalar_trace + risk_trace_bound + objective_slack_bound + 1.0


def _readout_matrices(instance: PortfolioInstance, positions: dict[str, object]) -> list[np.ndarray]:
    """One matrix per SOCP variable: Tr(readout_k . Y) = x_k."""
    n, m = instance.n, instance.m
    D = 3 * n + m + 3
    mats: list[np.ndarray] = []
    for key in ("w", "phi", "rho"):
        for pos in positions[key]:                       # type: ignore[index]
            mats.append(_sym_entry(D, int(pos), int(pos)))       # diagonal readouts
    mats.append(_sym_entry(D, int(positions["t"]), int(positions["t"])))
    risk_start = int(positions["risk_start"])
    for j in range(m):
        mats.append(_sym_entry(D, risk_start, risk_start + 1 + j))   # eta: OFF-diagonal
    return mats


def build_lifted_sdp(instance: PortfolioInstance, *, mu: float,
                     include_offdiag: bool = True) -> LiftedSDP:
    """Build the exact SOCP-to-SDP feasibility problem at objective threshold `mu`."""
    socp = build_socp_matrices(instance)
    n, m = instance.n, instance.m
    D = 3 * n + m + 3
    positions = _positions(n, m)
    R = trace_bound_R(instance, mu)
    readouts = _readout_matrices(instance, positions)
    u_mat = _sym_entry(D, int(positions["u"]), int(positions["u"]))

    raw: list[np.ndarray] = []
    targets: list[float] = []
    names: list[str] = []

    for row in range(socp.K):                            # the SOCP equalities
        mat = np.zeros((D, D))
        for col, coeff in enumerate(socp.A[row]):
            if coeff:
                mat += coeff * readouts[col]
        raw.append(mat); targets.append(float(socp.b[row])); names.append(f"socp_eq_{row}")

    obj = np.zeros((D, D))                               # c.x + u = mu   (the search knob)
    for col, coeff in enumerate(socp.c):
        if coeff:
            obj += coeff * readouts[col]
    obj += u_mat
    raw.append(obj); targets.append(float(mu)); names.append("objective_threshold")

    risk_start = int(positions["risk_start"])
    for j in range(1, m + 1):                            # eta block diagonal = t
        raw.append(_sym_entry(D, risk_start + j, risk_start + j)
                   - _sym_entry(D, risk_start, risk_start))
        targets.append(0.0); names.append(f"risk_diag_{j}")

    if include_offdiag:                                  # eta block off-diagonals = 0
        for i in range(1, m + 1):
            for j in range(i + 1, m + 1):
                raw.append(_sym_entry(D, risk_start + i, risk_start + j))
                targets.append(0.0); names.append(f"risk_offdiag_{i}_{j}")

    raw_stack = np.stack(raw)
    raw_targets = np.asarray(targets, dtype=float)
    density_targets = raw_targets / R
    scales = np.empty(len(raw))
    observables = np.empty_like(raw_stack)
    scaled_targets = np.empty_like(density_targets)
    for i, mat in enumerate(raw_stack):
        scale = max(float(np.linalg.norm(mat, ord=2)), abs(float(density_targets[i])), 1.0)
        scales[i] = scale
        observables[i] = mat / scale
        scaled_targets[i] = density_targets[i] / scale

    return LiftedSDP(observables=observables, targets=scaled_targets, raw_matrices=raw_stack,
                     raw_targets=raw_targets, scales=scales, names=tuple(names), R=R, D=D,
                     d=len(raw), n=n, m=m, mu=float(mu), include_offdiag=include_offdiag,
                     positions=positions, socp=socp)
