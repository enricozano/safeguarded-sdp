"""
Specialized implementation of the Gibbs dual for the uniform-diagonal SDP family

    max Tr(C rho)  s.t.  rho >= 0, Tr(rho) = 1, rho_ii = 1/N,  ||C||_op <= 1.

Mathematically identical to safeguarded_sdp's

    GibbsDualObjective(*instances.build_diagonal_sdp_instance(C, mu))

(same lambda = (lambda_0, d_0..d_{N-1}), same F_b, same gradient, same `b`), but it
never materialises the N+1 stacked N x N observables, which cost O(N^3) memory
(8.6 GB at N=1024, 43 GB at N=1751, 309 GB at N=3381). This class needs O(N^2);
it agrees with GibbsDualObjective to machine precision.

    K(lambda) = lambda_0 C + Diag(d)
    F(lambda) = log Tr exp(K) - lambda_0 mu - (1/N) sum_i d_i
    grad      = [Tr(rho C) - mu,  diag(rho) - 1/N]

One call = one dense eigendecomposition = one counted oracle round (n_gibbs).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class DiagSDPDual:
    def __init__(self, C: np.ndarray, mu: float):
        C = np.ascontiguousarray(C, dtype=np.float64)
        if C.ndim != 2 or C.shape[0] != C.shape[1]:
            raise ValueError("C must be square")
        self.C = C
        self.N = C.shape[0]
        self.mu = float(mu)
        self.inv_N = 1.0 / self.N
        self.b = np.concatenate(([-self.mu], np.full(self.N, -self.inv_N)))
        self.n_gibbs = 0
        self._best = (np.inf, None, None)
        self._diag = np.diag_indices(self.N)
        density = np.count_nonzero(C) / C.size
        self._C_mul = sp.csr_matrix(C) if density < 0.05 else C

    def dim(self) -> int:
        return self.N + 1

    def hilbert_dim(self) -> int:
        return self.N

    def _K(self, lam: np.ndarray) -> np.ndarray:
        K = float(lam[0]) * self.C
        K[self._diag] += lam[1:]
        return K

    def __call__(self, lam: np.ndarray) -> tuple[float, np.ndarray]:
        self.n_gibbs += 1
        lam = np.asarray(lam, dtype=np.float64)
        w, U = np.linalg.eigh(self._K(lam))
        wmax = float(w[-1])
        ew = np.exp(w - wmax)
        Z = float(ew.sum())
        p = ew / Z
        f = wmax + np.log(Z) + float(lam @ self.b)
        if f < self._best[0]:
            self._best = (f, lam.tobytes(), float(max(abs(w[0]), abs(w[-1]))))

        g = np.empty(self.N + 1)
        # Tr(rho C) = sum_l p_l u_l^T C u_l ;  diag(rho)_i = sum_l p_l U_il^2
        g[0] = float(p @ np.einsum("il,il->l", U, self._C_mul @ U)) - self.mu
        g[1:] = (U * U) @ p - self.inv_N
        return f, g

    def hamiltonian_op_norm(self, lam: np.ndarray) -> float:
        lam = np.asarray(lam, dtype=np.float64)
        if self._best[1] is not None and lam.tobytes() == self._best[1]:
            return self._best[2]
        w = np.linalg.eigvalsh(self._K(lam))
        return float(max(abs(w[0]), abs(w[-1])))
