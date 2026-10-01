"""
Builds a uniform-diagonal QUBO-SDP feasibility instance (Hermitian observable
list A_i + target vector b, for gibbs_dual.GibbsDualObjective) from a cost
matrix C, for the "direct normalized uniform-diagonal SDP" family:

    max Tr(C rho)  s.t.  rho >= 0, Tr(rho) = 1, rho_ii = 1/N (i = 1..N), ||C||_op <= 1.

For a feasibility query at threshold mu, this is encoded directly (no
trace-filling ancilla) as

    A_1 = -C,        b_1 = -mu,
    A_{i+1} = -|i><i|,  b_{i+1} = -1/N   (i = 0, ..., N-1),

so that F_b(lambda) with lambda = (lambda_0, d_0, ..., d_{N-1}) reproduces
K(lambda) = lambda_0*C + Diag(d) and f = log Tr exp(K) - lambda_0*mu -
(1/N) sum_i d_i: a state rho with Tr(rho)=1 (implicit; not itself one of the
listed constraints -- this is a "direct density formulation") matching
diag(rho) = 1/N exactly and Tr(C rho) >= mu is feasible for this mu iff
inf_lambda F_b(lambda) >= 0.

Also provides a small synthetic cost-matrix generator for examples/tests.
"""
from __future__ import annotations

import numpy as np


def build_diagonal_sdp_instance(C: np.ndarray, mu: float):
    """Direct density formulation of the feasibility query
    'exists rho >= 0, Tr(rho)=1, rho_ii=1/N for all i, Tr(C rho) >= mu?'.

    Returns (A_list, b) with A_1=-C, b_1=-mu; A_{i+1}=-|i><i|, b_{i+1}=-1/N,
    ready to build a gibbs_dual.GibbsDualObjective(A_list, b). Requires
    ||C||_op <= 1 (the normalized family this package targets).
    """
    C = np.asarray(C, dtype=np.float64)
    N = C.shape[0]
    if C.shape != (N, N):
        raise ValueError("C must be square")
    op_norm = float(np.linalg.norm(C, ord=2))
    if op_norm > 1.0 + 1e-9:
        raise ValueError(
            f"||C||_op = {op_norm:.6f} > 1; normalize C first"
        )

    A_list = [-C]
    b = [-float(mu)]
    for i in range(N):
        E_ii = np.zeros((N, N), dtype=np.float64)
        E_ii[i, i] = 1.0
        A_list.append(-E_ii)
        b.append(-1.0 / N)
    return A_list, np.asarray(b, dtype=np.float64)


def gen_cost_matrix(n: int, sparsity: int | None = None, seed: int = 0) -> np.ndarray:
    """Small synthetic real-symmetric cost matrix for examples/tests: an
    n x n zero-diagonal random symmetric matrix (optionally sparsified to at
    most `sparsity` nonzeros per row), normalized so ||C||_op <= 1 (rescaled
    by the entrywise infinity norm, which upper-bounds the operator norm for
    a symmetric matrix, so no further rescaling is needed).
    """
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, n))
    C = np.triu(M, 1)
    C = C + C.T  # symmetric, zero diagonal (common zero-diagonal QUBO normalization)

    if sparsity is not None and sparsity < n - 1:
        mask = np.zeros((n, n))
        while np.max(np.sum(mask, axis=1)) < sparsity:
            i = int(rng.integers(n))
            j = int(rng.integers(n))
            if i == j:
                continue
            mask[i, j] = 1.0
            mask[j, i] = 1.0
        C = C * mask

    inf_norm = np.linalg.norm(C, ord=np.inf)  # max abs row sum
    if inf_norm > 0:
        C = C / inf_norm
    return C
