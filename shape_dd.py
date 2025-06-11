import numpy as np
from utilities import mmd
from sklearn.metrics.pairwise import pairwise_kernels as apply_kernel

# tested against original version, this one is correct
class shape_dd:
    def __init__(self, data, w_size, n_perm):
        self.data = data  # full data set
        self.w_size = w_size  # window size
        self.n_perm = n_perm  # number of permutations
        self.stat = []  # stat value
        self.drift_detected = []  # tuple of results
        self.shape()

        # dataset X, l1 = first window size, l2 = second window size, n_perm = number of permutations

    def shape(self):
        w = np.array(self.w_size * [1.] + self.w_size * [-1.]) / float(self.w_size)
        n_size = self.data.shape[0]

        kernel_desc = apply_kernel(self.data, metric="rbf")
        W = np.zeros((n_size - 2 * self.w_size, n_size))

        for i in range(n_size - 2 * self.w_size):
            W[i, i:i + 2 * self.w_size] = w

        self.stat = np.einsum('ij,ij->i', np.dot(W, kernel_desc), W)
        shape = np.convolve(self.stat, w)
        shape_prime = shape[1:] * shape[:-1]

        res = np.zeros((n_size, 3))

        res[:, 2] = 1

        # when res[:, 2] == 1, there is definitely no drift, otherwise check p-value in results
        # res[:, 0] is shape-score
        # res[:, 1] is mmd-value
        # res[:, 2] is p-value

        # this step calculates the drift for the whole dataset at once
        for pos in np.where(shape_prime < 0)[0]:
            if shape[pos] > 0:
                res[pos, 0] = shape[pos]
                a, b = max(0, pos - int(self.w_size / 2)), min(n_size, pos + int(self.w_size / 2))
                res[pos, 1:] = mmd(self.data[a:b], pos - a, self.n_perm)
                self.drift_detected.append((pos,) + tuple(res[pos]))
        return res
