import numpy as np
import time

from src.gate_count import QuantumComputer
from scipy import linalg as linalg_cpu

# we try to import cupy to use a GPU to compute Gibbs states. If that fails, we just use the CPU
gpu = False
try:
    import cupy
    from cupyx.scipy import linalg as linalg_gpu

    gpu = True
    print('Using GPU with Cuda')
except ImportError:
    print('Cupy not installed. Using CPU instead.')

""""Helper functions"""

# computation of Gibbs states uses cupy if import was successful, otherwise scipy is used
# the free energy (-np.log(trace_)) is computed at the same time to safe resources
if gpu:
    # The cupy version of expm is less numerically stable than the scipy version. When H becomes to large, the function
    # can output nan or inf values, despite the trace being close to 1 mathematically. We prevent this by checking
    # whether a wrong output is given. If that is the case, H gets scaled down by a constant factor before computing
    # the exponential and correcting it afterwards by taking the power. The new scale is saved and used for subsequent,
    # doubling it again if neccessary. We track the increase in the computation time caused by the correction.
    def matrix_exp(H, logs):
        H_cp = cupy.asarray(H)
        exp_ = linalg_gpu.expm(H_cp / logs['scale'])

        if logs['scale'] != 1:
            time_start = time.time()
            exp_rescale = cupy.linalg.matrix_power(exp_, logs['scale'])
            logs['matrix_power_time'] += time.time() - time_start
        else:
            exp_rescale = exp_

        if not np.isfinite(np.trace(exp_rescale)):
            logs['scale'] *= 2
            print('nan in matrix exponential, doubling scale factor')
            exp_ = linalg_gpu.expm(H_cp / logs['scale'])

            time_start = time.time()
            exp_rescale = cupy.linalg.matrix_power(exp_, logs['scale'])
            logs['matrix_power_time'] += time.time() - time_start

        return cupy.asnumpy(exp_rescale)

else:
    def matrix_exp(H, _):
        return linalg_cpu.expm(H)


# This function outputs both the gibbs state and the free energy, as the can be computed in the same operation the save
# resources.
def gibbs_state(H, logs=None):
    if logs is None:
        logs = dict()
        logs['scale'] = 1
        logs['matrix_power_time'] = 0

    exp_ = matrix_exp(-H, logs)
    trace_ = np.trace(exp_)
    return exp_ / trace_, -np.log(trace_)


# the a-priori maximum number of iterations. This should never be reached in practice
def T_max(n, error_parameter):
    return int(64 * np.ceil(np.log(n) * (1 / error_parameter ** 2)) + 1)


# efficient implementation of trace(A @ B)
def tr_prod(A, B):
    return np.sum(A * B.T)


def get_P_d(diag_dev, n):
    norm_ = np.max(np.abs(diag_dev))
    if norm_ == 0:
        return np.zeros((n, n))
    return np.diag(diag_dev) / norm_


def dist_c(rho, P_c):
    return tr_prod(P_c, rho)


def diag_deviations(rho, n):
    return np.diag(rho) - np.ones(n) / n


"""Hamiltonian updates"""


def hamiltonian_update(
        C_til, gamma, epsilon, b=None,
        lambda_c=10, lambda_d=10, beta=0.4, lambda_increase=1.3,
        print_resolution=False, verbose=False
):
    s = np.max(np.count_nonzero(C_til, axis=1))
    n = C_til.shape[0]
    qc = QuantumComputer(b, n, s)

    # setting initial values
    H, rho, momentum, P_c, F, distance_c, l1_d, logs = initialize_values(n, C_til, gamma)

    distance_c = tr_prod(P_c, np.eye(n) / n)
    l1_d = 0

    logs = init_logs()

    for t in range(T_max(n, epsilon)):
        # printing current state of HU every k-th iteration, k=print_resolution
        if print_resolution:
            print_progress(print_resolution, verbose, t, distance_c, l1_d, F, lambda_c, lambda_d, logs, H,
                           n)
        if F > 0:  # free energy is positive -> there is no feasible solution -> terminate HU
            print_results(verbose, False, gamma, t, F, logs)
            return False, H, t, F, qc, logs

        qc.sample_trace_product()
        distance_c = tr_prod(P_c, rho)

        if distance_c >= epsilon:
            # c update

            Delta_H = distance_c * P_c + beta / lambda_c * momentum
            H, rho, F, lambda_c, logs = update(
                qc, H, Delta_H, lambda_c, epsilon, lambda_increase, F, logs, 'c'
            )
            momentum = lambda_c * Delta_H
            continue

        qc.sample_diag_dev(H, epsilon / 4)
        diag_dev = diag_deviations(rho, n)

        l1_d = np.sum(np.abs(diag_dev))
        if l1_d >= epsilon:
            # d update
            P_d = get_P_d(diag_dev, n)
            if distance_c < 0:
                Delta_H = P_d
            else:
                Delta_H = P_d + beta / lambda_d * momentum

            H, rho, F, lambda_d, logs = update(
                qc, H, Delta_H, lambda_d, epsilon, lambda_increase, F, logs, 'd'
            )
            momentum = lambda_d * Delta_H
            continue

        # We know both distances are below the error tolerance -> HU found a feasible solution
        print_results(verbose, True, gamma, t, F, logs)
        return True, H, t, F, qc, logs

    raise Exception('HU reached maximum number of iterations.')


def update(qc, H, Delta_H, lambda_, epsilon, lambda_increase, F_old, logs, update_type):
    H_new = H + lambda_ * Delta_H
    rho_new, F = gibbs_state(H_new, logs)

    qc.sample_trace_product()
    dist_mom_new = tr_prod(rho_new, Delta_H)

    while dist_mom_new < 0:
        # reduce step size
        lambda_ *= 0.5
        logs[f'overshot_{update_type}'] += 1

        # recompute H and rho
        H_new = H + lambda_ * Delta_H
        rho_new, F = gibbs_state(H_new, logs)

        # compute new distance and track it for the quantum computer
        qc.sample_trace_product()
        dist_mom_new = tr_prod(rho_new, Delta_H)

    logs['c_steps'].append(lambda_)
    # increasing the step factor for the next iteration

    lambda_ *= lambda_increase

    return H_new, rho_new, F, lambda_, logs


def initialize_values(n, C_til, gamma):
    H = np.zeros(shape=(n, n))
    rho = np.eye(n) / n
    momentum = np.zeros(shape=(n, n))
    P_c = - C_til + gamma * np.eye(n)  # constant
    F = -np.log(n)
    distance_c = tr_prod(P_c, rho)
    l1_d = 0
    logs = init_logs()
    return H, rho, momentum, P_c, F, distance_c, l1_d, logs


def init_logs():
    logs = dict()
    logs['c_steps'] = list()
    logs['d_steps'] = list()
    logs['d_overshoots'] = list()
    logs['overshot_c'] = 0
    logs['overshot_d'] = 0
    logs['entro_counter'] = 0  # if this is not 0 at the end, it's a strong indicator that something went wrong
    logs['matrix_power_time'] = 0
    logs['scale'] = 1
    return logs


def print_progress(print_resolution, verbose, t, distance_c, l1_d, F, lambda_c, lambda_d, logs, H, n):
    if t % print_resolution == 0 and t > 0:
        if verbose:
            print(f't: {t},', f'distance_c: {round(distance_c, 6)}', f'l1_distance_d: {round(l1_d, 6)}',
                  f'F: {F}',
                  f'lambda_c: {round(lambda_c, 4)}', f'lambda_d: {round(lambda_d, 4)},',
                  f'overshot_c: {logs["overshot_c"]}',
                  f'overshot_d: {logs["overshot_d"]},',
                  f'entropy_counter: {logs["entro_counter"]}', sep='    \t')
        else:
            print(f't: {t},', f'distance_c: {round(distance_c, 6)}', f'l1_distance_d: {round(l1_d, 6)}',
                  f'F: {round(F, 4)}')


def print_results(verbose, is_success, gamma, t, F, logs):
    if verbose:
        print(f'{"success!" if is_success else "infeasible!"}',
              f'gamma: {round(gamma, 6)}'.ljust(18, ' '), f't: {t},', f'F: {round(F, 4)}',
              ('' if logs['scale'] == 1 else f'matrix power time: {round(logs["matrix_power_time"], 4)}'),
              sep='    \t')
