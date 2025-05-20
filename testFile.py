from matplotlib import pyplot as plt
import numpy as np
import gen_data
import shape_dd as shape
import shape as shape_online
from sklearn.metrics.pairwise import rbf_kernel as rbf_kernel

# some kind of wrapper to get it running - either this will be implemented or the class itself needs to be refactored
def test_new_shape(X):
    # kernel function -- maybe it's better to refactor the initialization
    f = lambda X, y: rbf_kernel([X], [y])[0, 0]

    init_data = X[:20]

    so = shape_online.ShapeOnline(init_data, f, n_perm=1000)
    

    drift = None
    for t, x in enumerate(X[20:]):
        drift = so.update(x)
        if drift is not None:
            if drift != 1:
                print(f"possibly drift at {t + 20} MMD={drift[0]:.3f}, "
                      f"p-value={drift[1]:.3f}"
                      )

    #mmd = so.mmd_result
    print(len(so.mmd_result))
    print(len(so.drift_localized))

    #for i in np.where(mmd) > 0.001[0]:



# this is from the experimental code, just for testing
def test_shape_old(X):
    res_old_shape = shape.shape(X, l1=5, l2=20, n_perm=1000)

    for i in np.where(res_old_shape[:, 0] > 0.001)[0]:
        shape_score = res_old_shape[i, 0]
        mmd_value = res_old_shape[i, 1]
        p_value = res_old_shape[i, 2]

        print(f"possibly drift at {i}: "
              f"Shape-Score={shape_score:.3f}, "
              f"MMD={mmd_value:.3f}, "
              f"p-value={p_value:.3f}")


if __name__ == '__main__':
    print("generating data")
    X, y = gen_data.gen_random(dist="unif")
    # experimental version
    print("use experimental version")
    test_shape_old(X)
    # new version
    print("use shape online")
    test_new_shape(X)