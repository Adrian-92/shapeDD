import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_kernels as apply_kernel
import shape_online as shape_online
import shape_dd


def compare_online_vs_batch(data, window_size=100, n_perm=1000):
    """
    Compare one and batch version
    """

    def rbf_kernel(x, y, sigma=1.0):
        gamma = 1 / (2 * sigma ** 2)
        return np.exp(-gamma * np.linalg.norm(x - y) ** 2)

    # Batch-Version
    batch_shape = shape_dd.ShapeDD(data, window_size, n_perm)

    # Online-Version
    online_results = simulate_online_shapes(data, window_size, rbf_kernel, n_perm)

    return batch_shape, online_results


def simulate_online_shapes(data, window_size, kernel_func, n_perm):
    m = 2 * window_size
    n_total = len(data)

    results = {
        'stats': [],
        'shape_values': [],
        'drift_positions': [],
        'positions': []
    }

    online_shape = shape_online.ShapeOnline(window_size, kernel_func, n_perm)

    for i in range(n_total):
        drift_result = online_shape.update(data[i])
        # only collect stat values when initialization finished
        if len(online_shape.stat) > 0:
            results['stats'].append(online_shape.stat[-1])
            results['positions'].append(i)
        # only collect shape values when initialization finished
        if len(online_shape.shape) > 0:
            results['shape_values'].append(online_shape.shape[-1])
        # get drift positions
        if drift_result is not None and drift_result != 1:
            results['drift_positions'].append((i - m, drift_result))

    return results


def detailed_comparison_analysis(data, window_size=100):
    print("=" * 80)
    print("ANALYSIS: ONLINE vs BATCH SHAPES")
    print("=" * 80)

    test_data = data[:4]

    # compare kernels just to be sure

    # Batch Kernel
    sigma = 1
    gamma = 1 / (2 * sigma ** 2)
    batch_kernel = apply_kernel(test_data, metric="rbf", gamma=gamma)
    print("Batch Kernel Matrix:")
    print(batch_kernel)

    # Online Kernel
    def rbf_kernel(x, y, sigma=1.0):
        return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))

    online_kernel = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            online_kernel[i, j] = rbf_kernel(test_data[i], test_data[j])

    print("\nOnline Kernel Matrix:")
    print(online_kernel)

    print("\nComparison")
    print("-" * 40)

    batch_shape, online_results = compare_online_vs_batch(data, window_size)

    print(f"Batch Drifts: {len(batch_shape.drift_detected)}")
    print(f"Online Drifts: {len(online_results['drift_positions'])}")

    if batch_shape.drift_detected:
        print("\nBatch Drift-Positions:")
        for drift in batch_shape.drift_detected:
            print(f"  Position {drift[0]}: Shape={drift[1]:.4f}, MMD={drift[2]:.4f}, p-value={drift[3]:.4f}")

    if online_results['drift_positions']:
        print("\nOnline Drift-Positions:")
        for drift in online_results['drift_positions']:
            pos, (mmd_val, p_val) = drift
            print(f"  Position {pos}: MMD={mmd_val:.4f}, p-value={p_val:.4f}")


def plot_comprehensive_comparison(data, window_size=100):
    batch_shape, online_results = compare_online_vs_batch(data, window_size)

    fig, axes = plt.subplots(1, 2, figsize=(36, 12))

    # Plot 1: Batch SHAPES statistics
    axes[0].plot(range(len(batch_shape.stat)), batch_shape.stat, 'b-', linewidth=2)
    axes[0].set_title('Batch SHAPES statistic')
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('SHAPES statistic')
    axes[0].grid(True, alpha=0.3)

    # mark Batch-Drifts
    if batch_shape.drift_detected:
        drift_positions = [d[0] - window_size for d in batch_shape.drift_detected]
        drift_values = [batch_shape.stat[d[0] - window_size] if d[0] < len(batch_shape.stat) else 0
                        for d in batch_shape.drift_detected]
        axes[0].scatter(drift_positions, drift_values, c='red', s=100,
                           marker='x', label='Detected Drifts', linewidth=3)
        axes[0].legend()

    # Plot 2: Online SHAPES statistics
    if online_results['stats']:
        axes[1].plot(online_results['positions'], online_results['stats'], 'g-', linewidth=2)
        axes[1].set_title('Online SHAPES statistic')
        axes[1].set_xlabel('Position')
        axes[1].set_ylabel('SHAPES statistic')
        axes[1].grid(True, alpha=0.3)

        # mark online drift
        if online_results['drift_positions']:
            drift_pos = [d[0] + window_size for d in online_results['drift_positions']]
            # find stat values
            drift_stats = []
            for pos in drift_pos:
                if pos in online_results['positions']:
                    idx = online_results['positions'].index(pos)
                    drift_stats.append(online_results['stats'][idx])
                else:
                    drift_stats.append(0)

            axes[1].scatter(drift_pos, drift_stats, c='red', s=100,
                               marker='x', label='Detected Drifts', linewidth=3)
            axes[1].legend()

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Size of dataset: {len(data)}")
    print(f"Window Size: {window_size}")
    print(f"Batch detected Drifts: {len(batch_shape.drift_detected)}")
    print(f"Online detected Drifts: {len(online_results['drift_positions'])}")

    if batch_shape.drift_detected and online_results['drift_positions']:
        batch_pos = set(d[0] for d in batch_shape.drift_detected)
        online_pos = set(d[0] for d in online_results['drift_positions'])
        overlap = batch_pos.intersection(online_pos)
        print(f"Overlapping Drift-Positions: {len(overlap)}")
        print(f"Only Batch: {len(batch_pos - online_pos)}")
        print(f"Only Online: {len(online_pos - batch_pos)}")


if __name__ == "__main__":
    # test data
    np.random.seed(42069)

    # Segment 1: normal distribution
    seg1 = np.random.normal(0, 1, (500, 2))

    # Segment 2: data with drift
    seg2 = np.random.normal(3, 1, (500, 2))

    seg3 = np.random.normal(0, 2, (500, 2))

    test_data = np.vstack([seg1, seg2, seg3])

    print("checking detectors...")
    detailed_comparison_analysis(test_data, window_size=100)

    print("\nmaking plots...")
    plot_comprehensive_comparison(test_data, window_size=100)
