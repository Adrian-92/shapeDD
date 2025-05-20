import numpy as np
from utilities import mmd

def permutation_test(data):
    pass

class shape:
    def w_i(self, i: int):
        """
        Create w_i
        i: (int)
            index of the oldest element
        """
        w = np.ones((self.m))
        max_index = i + self.m//2
        if max_index >= self.m:
            w[i:] = -1
            w[:max_index - self.m] = -1
        else:
            w[i:max_index] = -1
        return w
    
    def update(self, x):
        raise NotImplementedError("Should be overwritten by the subclass!")

class ShapeOnline(shape):
    def __init__(self, data, f, n_perm = 1000):
        """
        Initialize the online shape instance. Used for tracking states, which do not need to be updated each step

        data: numpy 2-D array
            Shape (m,?) is expected, m is the number of data points, ? is the (irrelevant) dimension of each data point.
        f: fuction
            kernel function on the data space
        """
        m, _ = data.shape
        self.m = m                  # number of saves elements
        self.K = np.zeros((m,m))    # kernel matrix
        self.prod = np.zeros((m))   # K @ w_i
        self.f = f                  # kernel function
        self.i = 0                  # index of the oldest element
        self.data = data            # saved data points
        self.stat = []
        self.shape = []
        self.n_perm = n_perm        # Number of Perutations for the independence test
        self.drift_localized = []   # Saved points (according to number of data point, including the initial)
        self.mmd_result = []
        self.debug = 0

        # fill the initial values of the kernel matrix
        for i in range(m):
            for j in range(m):
                self.K[i,j] = f(data[i], data[j])
        self.prod = self.K @ self.w_i(self.i)
        self.stat.append(self.w_i(self.i) @ self.prod)

    # add new data point. Relevant for the online-scenario
    def update(self, x):
        new_i = (self.i + 1) % self.m
        max_index = (self.i + self.m//2) % self.m # the upper bound is exclusive

        # remove old values of prod, update perform update from w_i to w_{i + 1}
        self.prod = self.prod + self.K[self.i,:] - 2 * self.K[max_index, :]

        self.data[self.i] = x
        # calculate new kernel matrix
        # self.i is the oldest value, they need to be updated
        for j in range(self.m):
            self.K[self.i, j] = self.f(self.data[self.i], self.data[j])
        self.K[:,self.i] = self.K[self.i,:]

        self.prod = self.prod + self.K[self.i,:]     # w_i(i + i)[i] = 1
        # and update the new comlumn
        self.prod[self.i] = self.K[self.i, :].T @ self.w_i(new_i)

        self.i = new_i
        self.stat.append(self.w_i(self.i).T @ self.prod)

        self.debug += len(self.stat)
        if len(self.stat) < self.m:
            return None


        arr = np.array(self.stat[-self.m:])
        self.shape.append(self.w_i(self.m//2)@arr)
        if len(self.shape) < 2:
            return None
                
        # Test, if shape predicts a drift localization
        if self.shape[-1] * self.shape[-2] < 0:
            if self.shape[-1] > 0:
                print('testing')
                # is this really correct??
                result = mmd(np.concatenate([self.data[self.i:], self.data[:self.i]], axis=0), self.m//2, self.n_perm)
                # why?
                if result:
                    # added mmd_result for output
                    self.mmd_result.append(result)
                    self.drift_localized.append(len(self.shape))
                    return result
        
        # No (new) drift localized
        return 1

"""
DEBUG Class to compare the Kernels and the shape values.
"""
class ShapeNative(shape):

    def __init__(self, data, f):
        m, _ = data.shape
        self.m = m                  # number of saves elements
        self.K = np.zeros((m,m))    # kernel matrix
        self.f = f                  # kernel function
        self.i = 0                  # index of the oldest element
        self.data = data            # saved data points

        self.update_k()

    def update_k(self):
        # fill the initial values of the kernel matrix
        for i in range(self.m):
            for j in range(self.m):
                self.K[i,j] = self.f(self.data[i], self.data[j])
    
    def update(self, x):
        self.data[self.i] = x
        self.update_k()
        self.i = (self.i + 1) % self.m
        return self.w_i(self.i).T @ self.K @ self.w_i(self.i)
