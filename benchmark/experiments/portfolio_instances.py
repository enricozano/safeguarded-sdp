"""
Portfolio-optimization instances and their SOCP form.

make_instance_from_returns picks n assets at random from real daily returns
(data/portfolio/returns_liquid_us.csv) and takes the first m=2n days to build a
mean-risk problem

    min_w  -u_hat.w + q*||Mw||_2
    s.t.   sum(w)=1, |w-w_bar|<=zeta, w>=0

build_socp_matrices writes this as a second-order cone program with slack
variables phi, rho, t, eta.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class PortfolioInstance:
    source: str
    n: int
    m: int
    seed: int
    tickers: tuple[str, ...]
    returns_block: np.ndarray
    u_hat: np.ndarray
    M: np.ndarray
    Sigma: np.ndarray
    w_bar: np.ndarray
    q: float
    zeta: np.ndarray


@dataclass(frozen=True)
class SocpData:
    c: np.ndarray
    A: np.ndarray
    b: np.ndarray
    cone_dims: tuple[int, ...]
    offsets: dict[str, slice | int]
    N: int
    K: int
    r: int


def make_instance_from_returns(returns: pd.DataFrame, *, n: int, seed: int,
                               source: str = "real", q: float = 1.0,
                               zeta_value: float = 0.05) -> PortfolioInstance:
    """Pick n assets, take the first m = 2n days, build (u_hat, M, Sigma, w_bar)."""
    m = 2 * n
    usable = returns.dropna(axis=1, how="any")
    if usable.shape[1] < n:
        raise ValueError(f"Need {n} usable tickers, got {usable.shape[1]}")
    if usable.shape[0] < m:
        raise ValueError(f"Need {m} return rows, got {usable.shape[0]}")

    rng = np.random.default_rng(seed)
    sampled = tuple(rng.choice(np.array(usable.columns), size=n, replace=False).tolist())
    block = usable.loc[:, sampled].iloc[:m].to_numpy(dtype=float)
    u_hat = block.mean(axis=0)
    M = block - u_hat[None, :]                 # "raw" scaling, as in the reference code
    w_bar = np.ones(n) / n                     # equal weight
    zeta = zeta_value * np.ones(n)
    return PortfolioInstance(source=source, n=n, m=m, seed=seed, tickers=sampled,
                             returns_block=block, u_hat=u_hat, M=M, Sigma=M.T @ M,
                             w_bar=w_bar, q=q, zeta=zeta)


def load_returns_csv(path: str) -> pd.DataFrame:
    """Read the returns matrix (dates x tickers)."""
    return pd.read_csv(path, index_col=0, parse_dates=True)


def make_instance(*, n: int, seed: int, returns: pd.DataFrame,
                  q: float = 1.0, zeta_value: float = 0.05) -> PortfolioInstance:
    """Build an instance from real returns. `returns` is what load_returns_csv gives."""
    return make_instance_from_returns(returns, n=n, seed=seed, source="real",
                                      q=q, zeta_value=zeta_value)


def build_socp_matrices(instance: PortfolioInstance) -> SocpData:
    """The mean-risk problem as a SOCP: variables x = (w, phi, rho, t, eta)."""
    n, m = instance.n, instance.m
    N = 3 * n + m + 1
    K = 2 * n + m + 1
    w = slice(0, n)
    phi = slice(n, 2 * n)
    rho = slice(2 * n, 3 * n)
    t = 3 * n
    eta = slice(3 * n + 1, N)

    c = np.zeros(N)
    c[w] = -instance.u_hat
    c[t] = instance.q

    A = np.zeros((K, N))
    b = np.zeros(K)
    A[0, w] = 1.0                                    # budget: sum w = 1
    b[0] = 1.0

    row = 1
    A[row: row + n, w] = np.eye(n)                   # w + phi = w_bar + zeta
    A[row: row + n, phi] = np.eye(n)
    b[row: row + n] = instance.w_bar + instance.zeta
    row += n

    A[row: row + n, w] = np.eye(n)                   # w - rho = w_bar - zeta
    A[row: row + n, rho] = -np.eye(n)
    b[row: row + n] = instance.w_bar - instance.zeta
    row += n

    A[row: row + m, w] = instance.M                  # M w - eta = 0
    A[row: row + m, eta] = -np.eye(m)

    return SocpData(c=c, A=A, b=b,
                    cone_dims=tuple([1] * (3 * n) + [m + 1]),
                    offsets={"w": w, "phi": phi, "rho": rho, "t": t, "eta": eta},
                    N=N, K=K, r=3 * n + 1)


def portfolio_objective(instance: PortfolioInstance, w: np.ndarray) -> float:
    return float(-instance.u_hat @ w + instance.q * np.linalg.norm(instance.M @ w))


def solve_portfolio_reference(instance: PortfolioInstance, solver: str = "CLARABEL"):
    """Reference optimum p* via cvxpy -- the ground truth to check against."""
    import cvxpy as cp

    n = instance.n
    w = cp.Variable(n)
    obj = -instance.u_hat @ w + instance.q * cp.norm(instance.M @ w, 2)
    cons = [cp.sum(w) == 1, w >= 0, cp.abs(w - instance.w_bar) <= instance.zeta]
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=solver)
    return float(prob.value), str(prob.status), f"cvxpy_{solver.lower()}"
