import numpy as np


def gibbs_cost(b, n, s, H_maxnorm, epsilon):
    # lower bound for procedure discussed in the paper
    return (32 * b + 32 * np.log2(n) - 18) * (4.5 * np.log(7.8 / epsilon) * np.sqrt(n) * s * H_maxnorm - 1)


class QuantumComputer:
    def __init__(self, b, n, s):
        self.CNOT_count = 0
        self.count_diagonal_est = 0
        self.count_trace_products = 0
        self.norms = []
        self.b = b
        self.n = n
        self.s = s

        if self.b is None:
            print('Can\'t count gates without value given for b')
            self.b = 0
        if self.s is None:
            print('Can\'t count gates without value given for s')
            self.s = 0

    def sample_diag_dev(self, H, epsilon):
        if self.s == 0 or self.b == 0:
            return

        H_maxnorm = np.max(np.abs(H))  # conservative estimate for the norm of H_plus
        self.norms.append(H_maxnorm)

        number_samples = int(128 * epsilon ** (-2) * (self.n * np.log(2) + 4))
        if number_samples <= 0 or np.isnan(number_samples):
            raise f'Invalid sample number: {number_samples = }'

        self.CNOT_count += gibbs_cost(self.b, self.n, self.s, H_maxnorm, epsilon / 8) * number_samples
        self.count_diagonal_est += 1

    def sample_trace_product(self, H, epsilon):
        # we choose to ignore the cost from trace products as it is negligible compared to diagonal estimations
        self.count_trace_products += 1
