import shape_online
import numpy as np

window_size = 100


def rbf_kernel(x, y, sigma=1.0):
    gamma = 1 / (2 * sigma ** 2)
    return np.exp(-gamma * np.linalg.norm(x - y) ** 2)


def test_shape_online():
    np.random.seed(50)

    # Segment 1: normal distribution
    seg1 = np.random.normal(0, 1, (1000, 2))

    # Segment 2: data with drift
    seg2 = np.random.normal(3, 1, (1000, 2))

    # Segment 3: another drift
    seg3 = np.random.normal(0, 2, (1000, 2))

    test_data = np.vstack([seg1, seg2, seg3])

    # you can vary the arguments freely

    # only window_size
    #shape = shape_online.ShapeOnline(100)

    # OR window_size and permutations
    #shape = shape_online.ShapeOnline(100, 1000)

    # OR window_size and kernel function
    #shape = shape_online.ShapeOnline(100, rbf_kernel)

    # OR window_size, kernel function and permutations
    shape = shape_online.ShapeOnline(100, rbf_kernel, 1000)


    for i, val in enumerate(test_data):
        # note, that you need  at least 2 * window_size datapoints to get any result
        shape.update(val)

        if shape.get_warning_state():
            # you need to subtract window_size to get the right position of data, because of shift of that size
            print(f"warning at position {i - window_size}")
        if shape.drift_detected:
            drift_result = shape.get_last_drift_info()
            # you need to subtract 2 * window_size to get the right position of data, because of shift of that size
            print(
                f"shape detected at position {i - 2 * window_size}, mmd-test: {drift_result[0]}, p-value: {drift_result[1]}")


if __name__ == "__main__":
    test_shape_online()
