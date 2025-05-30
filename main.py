import numpy as np
from experimental import gen_data
from experimental.ShapeWrapper import calculate_shape_full, calculate_shape_online


def do_something():
    print("generating data")
    X, y = gen_data.gen_random(dist="unif")
    print("start testing")
    # experimental version
    print("use experimental version")
    result_full = calculate_shape_full(X, 100, 1000)

    mask = result_full[:,2] != 1

    for i in np.where(mask)[0]:
        mmd_value = result_full[i, 1]
        p_value = result_full[i, 2]
        print(f"possibly drift at {i}: "
              f"mmd-value={mmd_value:.3f}, "
              f"p-value={p_value:.3f}")

    print("use shape online")
    result_online = calculate_shape_online(X)

    for pos,mmd_val,p_val in result_online:
        print(f"possibly drift at {pos}:"
              f" mmd-value={mmd_val:3f},"
              f" p-value={p_val:3f}")

    print("done")


if __name__ == '__main__':
    do_something()