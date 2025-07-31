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
    def __init__(self, window_size=100, *args):

        """
        Initialize the online shape instance

        Parameters:
        -----------

        window_size : int
        size of the detection window - default value: 100

        args : expects 0 to 2 arguments

        if the first argument is a function it is used as kernel function
        otherwise it is used as number of permutations
        if both arguments are provided,
        the first argument is the kernel function and second argument is the number of permutations
        if no arguments are provided, default values (rbf_kernel, 1000) are used

        f : function
            kernel function
            default: rbf kernel
        n_perm : int
            Number of permutations for independence test
            default: 1000
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

        # fallback default values if not set manually: min_list_length=8, error_threshold=1, quantile_value=0.9

        self._warning_parameters = [8, 1, 0.9]
        self._warning_state = False
        self._warning_list = []
        self._error_count = 0

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
        """
        return the last drift info
        Returns:
            tuple :
            tuple with resulting mmd value and p value
        """
        return self._last_drift_info

    def get_warning_state(self, min_list_length=8, error_threshold=1, quantile_value=0.9):
        """
        technically this function used the result of the private calculation function calls
        check if given stat values are rising.
        any strictly rising values are added to _warning_list
        otherwise the error counter increases up to set threshold.
        calculate quantile and check if the next value is bigger than the n-quantile of _warning_list
        resets by any occurring drift-detection
        resets if error_threshold is reached

        Parameters:
        -----------
        next_stat : float
            next given stat value
        min_list_length : int
            change use this value to make it more generous oder strict in checking
            threshold of values in list to give warning. higher values are more accurate, but warnings will occur later
        error_threshold : int
            determines how many errors are allowed before clearing the warning list and resetting
            use this value careful, higher values make this function highly inaccurate
        quantile_value : float
            chosen quantile of list

        Returns:
            bool : True if a drift might be occurring

        """

        self._warning_parameters = [min_list_length, error_threshold, quantile_value]
        return self._warning_state

    def _calc_warning_state(self, next_stat):

        """
        this function is private. so no one should see this, right?
        description of this function is in get_warning_state
        Parameters:
        -----------
        next_stat : float
            next given stat value
        """

        min_list_length = self._warning_parameters[0]
        error_threshold = self._warning_parameters[1]
        quantile_value = self._warning_parameters[2]

        if not self._warning_list:
            self._warning_list.append(next_stat)
            return False

        last_stat = self._warning_list[-1]

        self._warning_list.append(next_stat)

        if next_stat < last_stat:
            self._error_count += 1

        if self._error_count >= error_threshold:
            self._warning_list.clear()
            self._error_count = 0
            return False

        if len(self._warning_list) > min_list_length:
            quantile = np.quantile(self._warning_list, quantile_value)
            if next_stat > quantile:
                return True

        return False

    # add new data point. Relevant for the online-scenario
    def update(self, x):

        """
        Updates the online detector with a new data point and performs drift detection.

        Parameters:
        -----------

        x : float
        New data point to process

        Returns:
        None : If no drift detection is performed (insufficient data or no drift)

        int : 1 if no drift is detected

        tuple : (float, float) if drift is detected

        **Behavior:**

        1. Adds the new data point to the buffer

        2. Initializes the kernel matrix when sufficient data is available (`m` points)

        3. Updates the kernel matrix and statistics for subsequent points

        4. Performs drift detection based on SHAPE value analysis

        5. Returns drift detection results
        """

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
