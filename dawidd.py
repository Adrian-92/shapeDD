import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels as apply_kernel


def get_time_kernel(n_size, n_perm, kernel):
    # init new empty dictionary // changed this from method-argument because it is mutable
    cache = dict()
    # there are currently no keys in dictionary, so why use it anyway?
    if (n_size, n_perm, kernel) not in cache.keys():
        # evenly space out numbers between -1 and 1, reshape to just one column
        T = np.linspace(-1, 1, n_size).reshape(-1, 1)
        # is multiplication with 1-array necessary? I'm pretty sure it is not needed
        # -> output is same with or without np.ones
        H = np.eye(n_size) - (1 / n_size) * np.ones((n_size, n_size))
        # when kernel is passed as "rbf" min will never occur!
        if kernel == "min":
            K = [H @ np.minimum(T[None, :], T[:, None])[:, :, 0] @ H]
            for _ in range(n_perm):
                T = np.random.permutation(T)
                K.append(H @ np.minimum(T[None, :], T[:, None])[:, :, 0] @ H)
        else:
            K = [H @ apply_kernel(T, metric=kernel) @ H]
            for _ in range(n_perm):
                T = np.random.permutation(T)
                ## STAGE 3 & 4
                K.append(H @ apply_kernel(T, metric=kernel) @ H)
        K = np.array(K).reshape(n_perm + 1, n_size * n_size)
        # K = K.reshape(n_perm + 1, n_size * n_size) combined these 2 lines
        cache[(n_size, n_perm, kernel)] = K
        # just return K?
    return cache[(n_size, n_perm, kernel)]


##STAGE 1
# maybe just pass the kernel-testmethod-string directly?
def dawidd(X, t_kernel="rbf", n_perm=2500):
    n_size = X.shape[0]
    # rbf is used multiple times // change to directly passing argument - argument should be renamed to metric?
    # apparently there is a rbf-kernel method in pairwise_kernels
    K_X = apply_kernel(X, metric="rbf")
    s = get_time_kernel(n_size, n_perm, t_kernel) @ K_X.ravel()
    p = (s[0] < s).sum() / n_perm

    return (1 / n_size) ** 2 * s[0], p

#Stage 1:
# take dataset X and pass it to dawidd
#Stage 2:
# using HSIC we compute the kernel matrix of data KX and time KT as descriptor ## TIME NOT AVAILABLE
#Stage 3:
# Observation time not available, thus use precomputed K = H x K_t x H
#Stage 4:
# permutation test