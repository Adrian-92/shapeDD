import shape_dd as shape
import shape as s_online
from sklearn.metrics.pairwise import rbf_kernel as rbf_kernel

'''
This is the online version of the shape-based drift detection algorithm.
    It uses the ShapeOnline class from the shape module.
    Note that this version calculates the shape score iteratively for each new datapoint 
    but this function takes the whole dataset as input.
    
    If you want to use this function, you need to provide a kernel function.
    The kernel function should take two data points as input and return the kernel value.
    The default kernel is the RBF kernel.
    If you want to use a different kernel, you can provide it as a parameter.
    If you want to use a datastream as input, provide a initial dataset and a window size, than use the update function for each new datapoint.
    '''


def calculate_shape_online(X, kernel_func=None, window_size=100, n_perm=1000):
    # kernel function
    if kernel_func is None:
        kernel_func = lambda X, y: rbf_kernel([X], [y])[0, 0]

    init_data = X[:window_size]

    so = s_online.ShapeOnline(init_data, kernel_func, n_perm=n_perm)

    # calculate drift for new datapoints in dataset
    for t, x in enumerate(X[window_size:]):
        so.update(x)

    # array with position, mmd-value and p-value
    drift_loc = so.drift_localized
    return drift_loc


'''
This is the experimental version of the shape-based drift detection algorithm.
    It uses the shape function from the shape_dd module.
    Note that this version calculates the shape score for the whole dataset at once.
    This method uses the RBF kernel from scikit-learn
    If you want to use a different kernel, you can provide it as a parameter.
'''


def calculate_shape_full(X, window_size=100, n_perm=1000):
    result_full = shape.shape(X, l1=window_size, l2=window_size, n_perm=n_perm)
    # when result_full[:,2] is 1 there is definitively no drift
    return result_full
