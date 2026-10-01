import numpy as np


class QuantumComputer:
    def __init__(self, b, n, s):
        self.CNOT_count = 0
        self.count_diagonal_est = 0
        self.count_trace_products = 0
        self.norms = []
        self.b = b
        self.n = n  # int(n)  # we get overflow problems with int32
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

        # eta = epsilon / 8

        H_infnorm = np.max(np.abs(H))  # conservative estimate # infnorm means max norm here
        self.norms.append(H_infnorm)
        number_samples = int(128 * epsilon ** (-2) * (self.n * np.log(2) + 4))
        if number_samples <= 0 or np.isnan(number_samples):
            raise f'Invalid sample number: {number_samples = }'

        self.CNOT_count += self.gibbs_cost(self.b, self.n, self.s, H_infnorm, epsilon / 8) * number_samples
        self.count_diagonal_est += 1

    def sample_trace_product(self):
        # we choose to ignore the cost from trace products as it is negligible compared to diagonal estimations
        self.count_trace_products += 1

    @staticmethod
    def gibbs_cost(b, n, s, H_infnorm, epsilon):
        return (32 * b + 32 * np.log2(n) - 18) * (4.5 * np.log(7.8 / epsilon) * np.sqrt(n) * s * H_infnorm - 1)
