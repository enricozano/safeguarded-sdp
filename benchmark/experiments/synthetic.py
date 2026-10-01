"""
Synthetic cost-matrix family for the precision-scaling experiment: the block
off-diagonal family of Henze et al. (their `gen_C_til`), C = [[0, B], [B^T, 0]]
with B an (N/2 x N/2) Gaussian block capped at `sparsity` nonzeros per row,
normalised to ||C||_inf = 1 (so ||C||_op <= 1).

A pure function of (k, sparsity, seed): seed s always gives the same matrix.
"""
from __future__ import annotations

import numpy as np


def gen_cost_matrix(k: int, sparsity: int | None, seed: int) -> np.ndarray:
    """Cost matrix for N = 2^k, row-sparsity `sparsity` of the off-diagonal block."""
    n = 2 ** k
    rng = np.random.default_rng(seed)
    n_half = n // 2

    c_block = rng.normal(size=(n_half, n_half))

    if sparsity is not None and sparsity < n_half:
        mask = np.zeros((n_half, n_half))
        # add symmetric pairs until the busiest row hits `sparsity` (a max, not an average)
        while np.max(np.sum(mask, axis=1)) < sparsity:
            i = int(rng.integers(n_half))
            j = int(rng.integers(n_half))
            if i == j:
                continue
            mask[i, j] = 1.0
            mask[j, i] = 1.0
        c_block = c_block * mask

    C = np.block([[np.zeros((n_half, n_half)), c_block],
                  [c_block.T, np.zeros((n_half, n_half))]])
    inf_norm = np.linalg.norm(C, ord=np.inf)
    if inf_norm > 0:
        C = C / inf_norm
    return C
