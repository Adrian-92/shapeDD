import numpy as np

from utilities import mmd


# default kernel function
def _rbf_kernel(x, y, sigma=1.0):
    gamma = 1 / (2 * sigma ** 2)
    return np.exp(-gamma * np.linalg.norm(x - y) ** 2)


class Shape:
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
        # size of first batch determines window size
        window_size = self.m

        w = np.ones(window_size)
        max_index = i + window_size // 2
        if max_index >= window_size:
            w[i:] = -1
            w[:max_index - window_size] = -1
        else:
            w[i:max_index] = -1

        return w

    def update(self, x):
        raise NotImplementedError("Should be overwritten by the subclass!")


class ForgettingList(list):
    def __init__(self, max_length):
        super(ForgettingList, self).__init__()
        self.max_length = max_length

    def append(self, item):
        super(ForgettingList, self).append(item)
        if len(self) > self.max_length:
            # l = l[-max_length:]
            del self[:-self.max_length]


class ShapeOnline(Shape):
    def __init__(self, window_size, *args):
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

        # default parameters
        self.kernel_func = _rbf_kernel  # kernel function
        self.n_perm = 1000  # Number of Permutations for the independence test

        if len(args) == 1:
            if callable(args[0]):
                self.kernel_func = args[0]
            elif isinstance(args[0], (int, float)):
                self.n_perm = args[0]
        elif len(args) == 2:
            if callable(args[0]):
                self.kernel_func, self.n_perm = args

        self.m = 2 * window_size  # number of saves elements (equals to 2 times window size)
        self.kernel_desc = np.zeros((self.m, self.m))  # kernel matrix

        self.prod = np.zeros(self.m)  # K @ w_i

        self.i = 0  # index of the oldest element
        self.data = []  # saved data points
        self.old_data = ForgettingList(self.m + self.m // 2)
        # statistical tracking of data
        self.stat = []
        self.shape = []
        self._last_drift_info = None
        self._drift_detected = False

        self._warning_state = False
        self._warning_list = []

    def _init_kernel(self):
        for i in range(self.m):
            for j in range(self.m):
                kernel_val = self.kernel_func(self.data[i], self.data[j])
                # enforce symmetry
                self.kernel_desc[i, j] = kernel_val
                self.kernel_desc[j, i] = kernel_val

        self.prod = self.kernel_desc @ self.w_i(self.i)
        self.stat.append(self.w_i(self.i) @ self.prod)

    @property
    def drift_detected(self):
        """
        Check if drift was detected and reset flag

        Returns:
        --------
        bool : True if drift was detected since last check
        """
        if self._drift_detected:
            self._drift_detected = False
            return True
        else:
            return False

    def get_last_drift_info(self):
        return self._last_drift_info

    def get_warning_state(self):
        return self._warning_state

    def _calc_warning_state(self, next_stat, min_list_length=15, min_mon_length=5, quantile_value=0.9):

        """check if given stat values are strictly rising.
           any strictly rising values are added to _warning_list
           calculate quantile and check if the next value is bigger than the n-quantile of _warning_list
           resets by any occurring drift-detection
           resets if next_stat is not bigger than last_stat

        Parameters:
        -----------
        next_stat : float
            next given stat value
        min_list_length : int
            threshold of values in list to give warning. higher values are more accurate, but warnings will occur later
        min_mon_length : int
            threshold of values that are strictly rising higher value is more accurate, but it will detect less
        quantile_value : float
            chosen quantile of list
           """

        if not self._warning_list:
            self._warning_list.append(next_stat)
            self._warning_state = False
            return False

        last_stat = self._warning_list[-1]

        if next_stat > last_stat:
            self._warning_list.append(next_stat)
            if len(self._warning_list) > min_list_length and len(self._warning_list) > min_mon_length:
                recent_values = self._warning_list[-min_mon_length:]
                is_strictly_increasing = all(x < y for x, y in zip(recent_values, recent_values[1:]))
                quantile = np.quantile(self._warning_list, quantile_value)
                if next_stat > quantile and is_strictly_increasing:
                    return True


            else:
                return False
        else:
            self._warning_list.clear()
            self._warning_list.append(next_stat)
            return False

    # add new data point. Relevant for the online-scenario
    def update(self, x):
        # reset last detected drift
        self._last_drift_info = None
        self.old_data.append(x)
        # need at least n data points to work
        if len(self.old_data) < self.m:
            self.data.append(x)
            return

        # fill the initial values of the kernel matrix
        if len(self.old_data) == self.m:
            self.data.append(x)
            self._init_kernel()

        # calculate indices
        new_i = (self.i + 1) % self.m
        max_index = (self.i + self.m // 2) % self.m  # calculate, where the last "-1" w_i is located

        # update is executed, avoid adding self.K[self.i,:] twice, which would dbe subtracted in the next update step
        self.prod = self.prod + self.kernel_desc[self.i, :] - 2 * self.kernel_desc[max_index, :]

        # self.i is the oldest value, they need to be updated
        self.data[self.i] = x

        # compute new kernel values
        for j in range(self.m):
            kernel_val = self.kernel_func(self.data[self.i], self.data[j])
            # enforce symmetry
            self.kernel_desc[self.i, j] = kernel_val
            self.kernel_desc[j, self.i] = kernel_val

        # Add the new kernel-row to the matrix, as described in 1.2
        # Add M by subtracting the (never added) old kernel value and add the new kernel value
        self.prod += self.kernel_desc[self.i,
                     :]  # Add M by subtracting the (never added) old kernel value and add the new kernel value
        self.prod[self.i] = self.kernel_desc[self.i, :].T @ self.w_i(new_i)  # recalculate row i

        self.i = new_i

        current_stat = (self.w_i(self.i).T @ self.prod)
        self.stat.append(current_stat)

        if len(self.stat) < self.m:
            return None

        recent_stat = (np.array(self.stat[-self.m:]))
        w_shape = self.w_i(self.m // 2)
        shape_value = w_shape.T @ recent_stat

        self.shape.append(shape_value)

        self._warning_state = self._calc_warning_state(current_stat)

        if len(self.shape) < 2:
            return None
        return self._detect_drift()

    def _detect_drift(self):
        window_size = self.m

        current_shape = self.shape[-1]
        previous_shape = self.shape[-2]
        shape_prime = current_shape * previous_shape
        if shape_prime < 0 < current_shape:
            mmd_result = mmd(self.old_data[:self.m], window_size // 2, self.n_perm)
            self._last_drift_info = mmd_result
            self._drift_detected = True
            self._warning_list.clear()
            self._warning_state = False
            return mmd_result
        else:
            return 1
