import shape_online
import numpy as np

window_size = 100


def rbf_kernel(x, y, sigma=1.0):
    gamma = 1 / (2 * sigma ** 2)
    return np.exp(-gamma * np.linalg.norm(x - y) ** 2)


def test_shape_online():
    np.random.seed(69)

    # Segment 1: normal distribution
    seg1 = np.random.normal(0, 1, (500, 2))

    # Segment 2: data with drift
    seg2 = np.random.normal(3, 1, (500, 2))

    # Segment 3: another drift
    seg3 = np.random.normal(0, 2, (500, 2))

    test_data = np.vstack([seg1, seg2, seg3])

    shape = shape_online.ShapeOnline(window_size,rbf_kernel,  1000)

    for i, val in enumerate(test_data):
        shape.update(val)
        if shape.drift_detected:
            drift_result = shape.get_last_drift_info()
            # u need to subtract 2 * window to get the right position of data, because of shift of that size
            print(
                f"shape detected at position {i - 2 * window_size}, mmd-test: {drift_result[0]}, p-value: {drift_result[1]}")


if __name__ == "__main__":
    test_shape_online()
