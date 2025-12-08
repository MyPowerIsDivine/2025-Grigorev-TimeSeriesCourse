import numpy as np

from modules.utils import *


def top_k_discords(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k discords based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k: number of discords

    Returns
    --------
    discords: top-k discords (indices, distances to its nearest neighbor and the nearest neighbors indices)
    """
 
    discords_idx = []
    discords_dist = []
    discords_nn_idx = []

    mp = matrix_profile['mp'].copy()
    mpi = matrix_profile['mpi']
    excl_zone = matrix_profile['excl_zone']

    mp = mp.astype(np.float64)

    mp[np.isinf(mp)] = -np.inf
    mp[np.isnan(mp)] = -np.inf

    for _ in range(top_k):
        idx = np.argmax(mp)
        max_dist = mp[idx]

        if np.isinf(max_dist):
            break

        discords_idx.append(int(idx))
        discords_dist.append(max_dist)
        discords_nn_idx.append(int(mpi[idx]))

        apply_exclusion_zone(mp, idx, excl_zone, -np.inf)

    return {
        'indices' : discords_idx,
        'distances' : discords_dist,
        'nn_indices' : discords_nn_idx
        }
