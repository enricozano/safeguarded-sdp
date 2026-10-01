from src.hamiltonian_updates import hamiltonian_update


def binary_search_HU(C_til, epsilon, gap, gamma_min=-1, gamma_max=1, hu_kwargs=None):
    if hu_kwargs is None:
        hu_kwargs = dict()

    best_H = None  # H of last successful binary search gets returned at the end
    CNOT_total = 0

    # binary search loop
    while gamma_max - gamma_min > gap:
        gamma = (gamma_max + gamma_min) / 2

        # perform HU for current value of gamma
        is_accepted, H, iters, F, qc, logs = hamiltonian_update(
            C_til, gamma, epsilon, **hu_kwargs
        )

        if is_accepted:
            gamma_min = gamma
            best_H = H

        else:
            gamma_max = gamma
        CNOT_total += qc.CNOT_count

    if best_H is None:
        raise Exception('Binary search terminated before finding a feasible H')

    return best_H, gamma_min, gamma_max, CNOT_total
