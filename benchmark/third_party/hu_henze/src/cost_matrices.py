import numpy as np
from scipy import linalg
import time

import cvxpy as cp
import uuid


def dump_to_uuid(file_prefix, data, time_):
    fileName = file_prefix + str(uuid.uuid4()) + ".npy"

    print(f"Generated in {round(time_, 3)}s. Dumping data to", fileName)
    np.save(fileName, data)


"""generate/solve/save C_til"""


def sample_uniform_dim(n_min, n_max, s, number, save_dir, solve_SDP, solver_eps, solver_verbose=False):
    solver_kwargs = {'solver': cp.SCS, 'verbose': solver_verbose, 'eps': solver_eps}
    for _ in range(number):
        n = 2 * np.random.randint(int(n_min / 2), int(n_max / 2))
        new_C_til_dump(n, s, save_dir=save_dir, solve_SDP=solve_SDP, solver_kwargs=solver_kwargs,
                       include_rho=solve_SDP)


def sample_fixed_dim(n, s, number, save_dir, solve_SDP, solver_eps, solver_verbose=False):
    solver_kwargs = {'solver': cp.SCS, 'verbose': solver_verbose, 'eps': solver_eps}
    for _ in range(number):
        new_C_til_dump(n, s, save_dir=save_dir, solve_SDP=solve_SDP, solver_kwargs=solver_kwargs,
                       include_rho=solve_SDP)


def new_C_til_dump(n, s, save_dir, solve_SDP, solver_kwargs=None, include_rho=False):
    start = time.time()
    C_til = gen_C_til(n, s)
    opt, rho_opt = None, None
    if solve_SDP:
        opt, rho_opt = solve(C_til, n, solver_kwargs=solver_kwargs)
    time_ = time.time() - start
    context = {'n': n, 's': n if s is None else s, 'solver_kwargs': solver_kwargs, 'solver_time': time_}
    if include_rho:
        dump_to_uuid(f'{save_dir}/n_{n}_s_{s}_',
                     {'C_til': C_til, 'opt': opt, 'rho_opt': rho_opt, 'context': context}, time_)
    else:
        dump_to_uuid(f'{save_dir}/n_{n}_s_{s}_',
                     {'C_til': C_til, 'opt': opt, 'rho_opt': np.nan, 'context': context}, time_)
    return {'C_til': C_til, 'opt': opt, 'context': context}


def gen_C_til(n, s=None):  # lazy implementation, gets slow if s is close to n
    if n % 2 != 0:
        raise Exception('Dimension needs to be even to allow block form')
    n_half = int(n / 2)
    C_block = np.random.normal(size=(n_half, n_half))

    # generate a sparsity mask that decide which entries of C_block are non-zero
    if s is not None and s < n_half:
        sparsity_mask = np.zeros((n_half, n_half))
        while np.max(np.sum(sparsity_mask, axis=1)) < s:
            i = np.random.randint(n_half)
            j = np.random.randint(n_half)
            if i == j:
                continue
            sparsity_mask[i, j] = 1
            sparsity_mask[j, i] = 1
        C_block = C_block * sparsity_mask

    C = np.block([[np.zeros_like(C_block), C_block], [C_block.T, np.zeros_like(C_block)]])

    C = C / linalg.norm(C, ord=np.inf)

    return C


def solve(C_til, n, solver_kwargs=None):
    if solver_kwargs is None:
        solver_kwargs = dict()
    cvx_C = C_til
    b = np.ones(n) / n

    X = cp.Variable((n, n), symmetric=True)

    constraints = [X >> 0]
    constraints += [
        X[i, i] == b[i] for i in range(n)
    ]
    prob = cp.Problem(cp.Maximize(cp.sum(cp.multiply(cvx_C, cp.transpose(X)))), constraints)
    return prob.solve(**solver_kwargs), X.value
