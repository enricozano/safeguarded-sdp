"""
General Gibbs dual F_b(lambda) and its gradient for arbitrary Hermitian
observables A_1..A_m, ||A_i||_op <= 1, and target b in [-1,1]^m:

    F_b(lambda) = log Tr exp(-sum_i lambda_i A_i) + lambda^T b
    grad F_b(lambda)_i = b_i - Tr(rho_lambda A_i), rho_lambda the Gibbs state.
"""
from __future__ import annotations

import numpy as np


class GibbsDualObjective:
    """Callable F_b(lambda) -> (f, grad), the Gibbs dual and its gradient.
    """

    def __init__(self, A_list, b):
        if len(A_list) == 0:
            raise ValueError("need at least one observable")
        A_stack = np.stack(
            [np.ascontiguousarray(A, dtype=np.float64) for A in A_list], axis=0
        )
        m, N, N2 = A_stack.shape
        if N2 != N:
            raise ValueError("each A_i must be a square N x N matrix")
        self.A_stack = A_stack
        self.m = m
        self.N = N
        self.b = np.asarray(b, dtype=np.float64)
        if self.b.shape != (self.m,):
            raise ValueError("b must have length m = len(A_list)")
        self.n_gibbs = 0  # counts every oracle call, including line-search probes

    def dim(self) -> int:
        return self.m

    def hilbert_dim(self) -> int:
        """N, the Hilbert-space (matrix) dimension. """
        return self.N

    def _K(self, lam: np.ndarray) -> np.ndarray:
        """K(lambda) = -sum_i lambda_i A_i, the argument of exp() in F_b."""
        lam = np.asarray(lam, dtype=np.float64)
        return -np.tensordot(lam, self.A_stack, axes=(0, 0))

    def __call__(self, lam: np.ndarray) -> tuple[float, np.ndarray]:
        self.n_gibbs += 1
        lam = np.asarray(lam, dtype=np.float64)
        K = self._K(lam)

        w, U = np.linalg.eigh(K)  # the one counted Gibbs computation, O(N^3) classically
        wmax = float(w[-1])
        ew = np.exp(w - wmax)
        Z = float(ew.sum())
        logZ = wmax + np.log(Z)
        p = ew / Z  # eigenvalue populations of rho_lambda

        f = logZ + float(lam @ self.b)

        # rho formed once (O(N^3)); every Tr(rho A_i) then comes from
        # a single batched contraction (O(m*N^2)) instead of m separate
        # O(N^3) matmuls.
        rho = (U * p) @ U.T
        a = np.einsum("jk,ijk->i", rho, self.A_stack)

        g = self.b - a
        return f, np.asarray(g, dtype=np.float64)

    def hamiltonian_op_norm(self, lam: np.ndarray) -> float:
        """||K(lambda)||_op = max |eigenvalue|. Diagnostic only; not a
        counted Gibbs call."""
        K = self._K(lam)
        return float(np.linalg.norm(K, ord=2))
