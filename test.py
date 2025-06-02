import numpy as np
from experimental import gen_data
from experimental import shape_dd
from experimental.shape import shape_online

window_size = 100


# gaussian kernel
def g_kernel(sig):
    return lambda x, y: np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sig ** 2))


def test_shape():
    print("generating data")
    X, y = gen_data.gen_random(dist="unif")
    print("start testing")

    '''
    This is the experimental version of the shape-based drift detection algorithm.
        It uses the shape function from the shape_dd module.
        Note that this version calculates the shape score for the whole dataset at once.
        This method uses the RBF kernel from scikit-learn
        If you want to use a different kernel, you can provide it as a parameter.
    '''

    print("use experimental version")

    # TODO: make this cleaner
    s_dd = shape_dd.shape_dd(X, window_size, 1000)
    print("shape_dd:")
    for i in s_dd.drift_detected:
        if i[3] != 1:
            mmd_value = i[2]
            p_value = i[3]
            print(f"possibly drift at {i[0]}: "
                  f"mmd-value={mmd_value:.3f}, "
                  f"p-value={p_value:.3f}"
                  )

    '''
    This is the online version of the shape-based drift detection algorithm.
        It uses the ShapeOnline class from the shape module.
        Note that this version calculates the shape score iteratively for each new datapoint 
        
        If you want to use this function, you need to provide a kernel function.
        The kernel function should take two data points as input and return the kernel value.
        The default kernel is the gaussian kernel.
        If you want to use a different kernel, you can provide it as a parameter.
        If you want to use a datastream as input, provide a initial dataset and a window size, than use the update function for each new datapoint.
        '''
    print("use shape online")
    so = shape_online(X[0:window_size], g_kernel(1))

    for i in range(len(X) - window_size):
        so.update(X[i])

    for pos, mmd_val, p_val in so.drift_localized:
        print(f"possibly drift at {pos}:"
              f" mmd-value={mmd_val:3f},"
              f" p-value={p_val:3f}")

    print("done")


if __name__ == '__main__':
    test_shape()
