import numpy as np
from utilities import mmd


class shape:
    def w_i(self, i: int):
        """
        Parameters:
        -----------
        i : int
            Index of the oldest element

        Returns:
        --------
        w : numpy array
            Weight vector of length m
        """
        w = np.ones(self.m)
        half_window = self.m // 2
        start_index = i
        end_index = (i + half_window) % self.m

        if start_index <= end_index:
            w[start_index:end_index] = -1
        else:
            w[start_index:] = -1
            w[:end_index] = -1

        return w
    def update(self, x):
        raise NotImplementedError("Should be overwritten by the subclass!")


class shape_online(shape):
    def __init__(self, data, f, n_perm=1000):
        """
        Initialize the online shape instance

        Parameters:
        -----------
        data : numpy 2-D array
            Shape (m,?) expected
        f : function
            kernel function
        n_perm : int
            Number of permutations for independence test
        """

        m = data.shape[0]
        self.m = m  # number of saves elements
        self.kernel_desc = np.zeros((m, m))  # kernel matrix
        self.n_perm = n_perm  # Number of Permutations for the independence test

        self.result = np.zeros(m)  # K @ w_i
        self.kernel_func = f  # kernel function
        self.i = 0  # index of the oldest element
        self.data = data  # saved data points

        # statistical tracking of data
        self.stat = []
        self.shape = []
        self.drift_localized = []  # Saved points (according to number of data point, including the initial)
        # contains position, mmd-value and p-value of possible location

        self._init_kernel_matrix()

        self.result = self.kernel_desc @ self.w_i(self.i)
        self.stat.append(self.w_i(self.i) @ self.result)


    def _init_kernel_matrix(self):
        # fill the initial values of the kernel matrix
        for i in range(self.m):
            for j in range(self.m):
                kernel_val = self.kernel_func(self.data[i], self.data[j])
                # enforce symmetry
                self.kernel_desc[i, j] = kernel_val
                self.kernel_desc[j, i] = kernel_val


    # add new data point. Relevant for the online-scenario
    def update(self, x):
        new_i = (self.i + 1) % self.m
        max_index = (self.i + self.m // 2) % self.m  # calculate, where the last "-1" w_i is located

        # update as described in step 1.1 is executed, avoid adding self.K[self.i,:] twice, which would dbe subtracted in the next update step
        self.result = self.result + self.kernel_desc[self.i, :] - 2 * self.kernel_desc[max_index, :]

        # calculate new kernel matrix
        # self.i is the oldest value, they need to be updated
        self.data[self.i] = x
        for j in range(self.m):
            self.kernel_desc[self.i, j] = self.kernel_func(self.data[self.i], self.data[j])
        self.kernel_desc[:, self.i] = self.kernel_desc[self.i, :]

        # Add the new kernel-row to the matrix, as described in 1.2
        self.result += self.kernel_desc[self.i,
                     :]  # Add M by subtracting the (never added) old kernel value and add the new kernel value
        self.result[self.i] = self.kernel_desc[self.i, :].T @ self.w_i(new_i)  # recalculate row i

        self.i = new_i
        self.stat.append(self.w_i(self.i).T @ self.result)

        if len(self.stat) < self.m:
            return None

        arr = np.array(self.stat[-self.m:])
        self.shape.append(self.w_i(self.m // 2) @ arr)
        if len(self.shape) < 2:
            return None

        # Test, if shape predicts a drift localization
        if self.shape[-1] * self.shape[-2] < 0:
            if self.shape[-1] > 0:
                mmd_result = mmd(np.concatenate([self.data[self.i:], self.data[:self.i]], axis=0), self.m // 2,
                                 self.n_perm)
                self.drift_localized.append((len(self.shape),) + mmd_result)
                return mmd_result
        # No (new) drift localized
        return 1


"""
DEBUG Class to compare the Kernels and the shape values.
"""


class stat_native(shape):

    def __init__(self, data, f):
        m, _ = data.shape
        self.m = m  # number of saves elements
        self.K = np.zeros((m, m))  # kernel matrix
        self.f = f  # kernel function
        self.i = 0  # index of the oldest element
        self.data = data  # saved data points

        self.update_k()
        self.stat = [self.w_i(self.i).T @ self.K @ self.w_i(self.i)]

    def update_k(self):
        # fill the initial values of the kernel matrix
        for i in range(self.m):
            for j in range(self.m):
                self.K[i, j] = self.f(self.data[i], self.data[j])

    def update(self, x):
        self.data[self.i] = x
        self.update_k()
        self.i = (self.i + 1) % self.m
        res = self.w_i(self.i).T @ self.K @ self.w_i(self.i)
        self.stat.append(res)
        return res
