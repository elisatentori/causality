import networkx as nx
import numpy as np
from joblib import Parallel, delayed


def find_SP(arr, indices, dist_mat=False, outf: str = None, verbose=False, return_hops=False, eps=1e-12):
    """
    Fast shortest-paths (directed) from given sources.

    Parameters
    ----------
    arr : (N, N) array-like
        If dist_mat=False: edge weights (can be positive/negative; only finite & nonzero create edges).
        If dist_mat=True : edge distances (only finite & >0 create edges).
    indices : 1D array-like of int
        Source node indices (rows to compute).
    dist_mat : bool, default False
        Treat `arr` as distances if True; as weights otherwise.
    outf : str or None
        If given, save SP matrix to this path (.npy). If return_hops=True, also saves *_hops.npy.
    verbose : bool
        Print progress.
    return_hops : bool
        If True, also return number of hops (unweighted shortest-path).
    eps : float
        Small constant to avoid division by zero when converting weights to costs.

    Returns
    -------
    If return_hops=False:
        SP_mat : (N, N) float64
            Weighted shortest-path distances from sources (rows in `indices`); others = +inf.
    If return_hops=True:
        (SP_mat, Hops_mat)
            Hops_mat contains the minimum number of edges (topological distance); unreachable = +inf.
    """
    if verbose:
        print("\nComputing shortest paths (SciPy csr dijkstra)...")

    A = np.asarray(arr, dtype=np.float64)
    idx = np.asarray(indices, dtype=int)
    N = A.shape[0]

    if dist_mat:
        # Interpret A as distances: edges only where finite and > 0
        cost = A.copy()
        # Zero-out invalids and non-edges so csr graph has no edge there
        invalid = ~np.isfinite(cost) | (cost <= 0.0)
        cost[invalid] = 0.0
        edge_mask = np.isfinite(A) & (A > 0.0)
    else:
        # Interpret A as weights: edges only where finite and nonzero
        cost = np.zeros_like(A, dtype=np.float64)
        mask = np.isfinite(A) & (A != 0.0)
        cost[mask] = 1.0 / (np.abs(A[mask]) + eps)
        edge_mask = mask

    # No self cost
    np.fill_diagonal(cost, 0.0)

    # Weighted graph for Dijkstra
    Gcsr = csr_matrix(cost)
    dist = dijkstra(Gcsr, directed=True, indices=idx)

    # Pack into full (N,N) with +inf everywhere except requested rows
    SP_mat = np.full((N, N), np.inf, dtype=np.float64)
    SP_mat[idx, :] = dist

    if not return_hops:
        if outf:
            np.save(outf, SP_mat)
        if verbose:
            print("...done.")
        return SP_mat

    # Unweighted graph for hop count (topological distance)
    Gbin = csr_matrix(edge_mask.astype(np.int8))
    hops = shortest_path(Gbin, directed=True, unweighted=True, indices=idx)

    Hops_mat = np.full((N, N), np.inf, dtype=np.float64)
    Hops_mat[idx, :] = hops

    if outf:
        np.save(outf, SP_mat)
        np.save(outf.replace('.npy', '_hops.npy') if outf.endswith('.npy') else outf + '_hops.npy', Hops_mat)
    if verbose:
        print("...done.")
    return SP_mat, Hops_mat




'''
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
'''