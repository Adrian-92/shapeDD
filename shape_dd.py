import numpy as np
from utilities import mmd
from sklearn.metrics.pairwise import pairwise_kernels as apply_kernel

# dataset X, l1 = first window size, l2 = second window size, n_perm = number of permutations
def shape(X, l1, l2, n_perm):
    w = np.array(l1 * [1.] + l1 * [-1.]) / float(l1)
    n_size = X.shape[0]
    kernel_desc = apply_kernel(X, metric="rbf")
    W = np.zeros((n_size - 2 * l1, n_size))

    for i in range(n_size - 2 * l1):
        W[i, i:i + 2 * l1] = w
    stat = np.einsum('ij,ij->i', np.dot(W, kernel_desc), W)
    shape = np.convolve(stat, w)
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
            a, b = max(0, pos - int(l2 / 2)), min(n_size, pos + int(l2 / 2))
            res[pos, 1:] = mmd(X[a:b], pos - a, n_perm)
    return res

