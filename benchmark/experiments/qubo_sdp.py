"""
QUBO -> Ising-SDP conversion for the logistics instances. convert/convert_back/
get_const are verbatim from the Ostermann et al. repository (see NOTICE, item 3),
so bounds computed here are on the same scale as the published ones.

QUBO Q (min x^T Q x) -> homogenized C = -Q, normalized so ||C||_inf = 1. The SDP
solved on C is max Tr(C rho) s.t. rho >= 0, diag(rho) = 1/(n+1).
"""
from __future__ import annotations

import numpy as np
from scipy import linalg


def convert(QUBO):
    """QUBO (minimization) -> normalized Ising SDP cost matrix C_til + conversion info."""
    A = -QUBO                                   # minus sign: minimization -> maximization
    B = np.sum(A, axis=1) / 4
    c = (np.sum(A) + np.sum(np.diag(A))) / 4    # constant restored when converting back
    n = A.shape[0]
    C = np.zeros((n + 1, n + 1))                # row/col 0 encodes the linear terms
    C[1:, 0] = B
    C[0, 1:] = B
    C[1:, 1:] = (A - np.diag(np.diag(A))) / 4
    C_til = C / linalg.norm(C, ord=np.inf)      # normalize by operator (inf-)norm
    return C_til, {"A": A, "B": B, "C": C, "c": c}


def convert_back(value, conversion_info):
    """Map an SDP objective value (on C_til scale) back to the QUBO objective."""
    rescaled_value = value * linalg.norm(conversion_info["C"], ord=np.inf)
    return -(rescaled_value + conversion_info["c"])


def get_const(filename):
    """Read the constant offset encoded in the filename tail ..._c<CONST>.csv."""
    i = filename[:-4].rfind("c")
    return float(filename[i + 1:-4])
