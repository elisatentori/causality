import networkx as nx
import numpy as np
from joblib import Parallel, delayed


def compute_shortest_paths(G, i, n_nodes):
    return [nx.shortest_path_length(G, source=i, target=j, weight='weight', method='dijkstra') for j in range(n_nodes)]


def find_SP(arr, indices, dist_mat=False, outf: str = None, n_jobs: int = -1, verbose=False):
    if verbose:
        print('\n\nComputing shortest paths....')
    SP_mat = np.zeros_like(arr)

    if dist_mat:
        arr_distance = np.copy(arr)
    else:
        arr_distance = np.ones(arr.shape) * np.inf
        arr_distance[arr != 0] = 1 / arr[arr != 0]

    G = nx.from_numpy_array(arr_distance)

    n_nodes = len(arr)
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_shortest_paths)(G, i, n_nodes) for i in indices
    )

    for idx, i in enumerate(indices):
        SP_mat[i, :] = results[idx]

    if outf:
        np.save(outf, SP_mat)

    if verbose:
        print('.........done')

    return SP_mat
