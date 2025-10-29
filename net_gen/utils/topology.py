import numpy as np


#=================================================================================================#
# EMPIRICAL MAP

# reorder empirical channel map by cluster ID
def _reorder_clusters_by_x(pos, cluster):
    pos = np.asarray(pos)
    cluster = np.asarray(cluster)

    unique_clusters, inverse_indices = np.unique(cluster, return_inverse=True)
    x_means = np.zeros(len(unique_clusters))

    for i, uc in enumerate(unique_clusters):
        x_means[i] = pos[cluster == uc, 0].mean()

    sorted_order = np.argsort(x_means)

    new_cluster_ids = np.zeros_like(unique_clusters)
    new_cluster_ids[sorted_order] = np.arange(len(unique_clusters))

    dir_clusters = new_cluster_ids[inverse_indices]
    return dir_clusters


# subsample indexes of empirical map
def _subsample_indices(channel, cluster, rate, rng=None):
    
    if rng is None:
        rng = np.random.default_rng()
        
    # Extract info from view (original or sorted)    
    cluster = np.asarray(cluster)
    n_clusters = cluster.max() + 1
    best_indices = []
    
    for c in range(n_clusters):
        indices = np.where(cluster == c)[0]
        if len(indices) == 0:
            continue
        cluster_rates = rate[indices]
        max_rate = np.max(cluster_rates)
        max_pos  = np.where(cluster_rates == max_rate)[0]
        chosen   = indices[rng.choice(max_pos)]
        best_indices.append(chosen)
    
    return np.array(best_indices)


#=================================================================================================#
# GENERATE RANDOM UNIFORM MAP
# Poisson disk sampling to generate a random uniform 2D map with non overlapped points

# min radius for n_points
def estimate_r_min(n_points, x_min, x_max, y_min, y_max, c=0.9):
    area = (x_max - x_min) * (y_max - y_min)
    r_min = np.sqrt(area / (n_points * c))
    return r_min

#-------------------------------------------------------------------------------------------------#

# generate coordinates with poisson disk sampling
def generate_poisson_disk_samples(x_min, x_max, y_min, y_max, r_min, k=30, seed=None):
    
    rng = np.random.default_rng(seed)

    width  = x_max - x_min
    height = y_max - y_min

    cell_size   = r_min / np.sqrt(2)
    grid_width  = int(np.ceil(width / cell_size))
    grid_height = int(np.ceil(height / cell_size))
    grid        = -np.ones((grid_height, grid_width), dtype=int)

    samples     = []
    active_list = []

    # first random point
    x0 = rng.uniform(x_min, x_max)
    y0 = rng.uniform(y_min, y_max)
    samples.append((x0, y0))

    gi           = int((y0 - y_min) / cell_size)
    gj           = int((x0 - x_min) / cell_size)
    grid[gi, gj] = 0
    active_list.append(0)

    while active_list:
        idx   = rng.choice(active_list)
        x, y  = samples[idx]
        found = False

        for _ in range(k):
            radius = rng.uniform(r_min, 2 * r_min)
            angle  = rng.uniform(0, 2 * np.pi)
            new_x  = x + radius * np.cos(angle)
            new_y  = y + radius * np.sin(angle)

            # check: is the dot in the rectangle?
            if not (x_min <= new_x <= x_max and y_min <= new_y <= y_max):
                continue

            gi = int((new_y - y_min) / cell_size)
            gj = int((new_x - x_min) / cell_size)

            too_close = False
            for ii in range(max(0, gi - 2), min(grid_height, gi + 3)):
                for jj in range(max(0, gj - 2), min(grid_width, gj + 3)):
                    neighbor_idx = grid[ii, jj]
                    if neighbor_idx != -1:
                        nx, ny = samples[neighbor_idx]
                        if np.hypot(new_x - nx, new_y - ny) < r_min:
                            too_close = True
                            break
                if too_close:
                    break

            if not too_close:
                samples.append((new_x, new_y))
                grid[gi, gj] = len(samples) - 1
                active_list.append(len(samples) - 1)
                found = True

        if not found:
            active_list.remove(idx)

    return np.array(samples)

#-------------------------------------------------------------------------------------------------#

# generate coordinates
def poisson_disk_until_n(x_min, x_max, y_min, y_max, n_points, r_min_init=0.1, max_iter=20, seed=None):
    
    rng = np.random.default_rng(seed)
        
    r_min = r_min_init
    for _ in range(max_iter):
        samples = generate_poisson_disk_samples(x_min, x_max, y_min, y_max, r_min)
        if len(samples) >= n_points:
            rng.shuffle(samples)
            return samples[:n_points], r_min
        r_min *= 0.95  # Reduce r_min to augment density
    raise RuntimeError(f"Impossible to generate {n_points} points after {max_iter} iterations.")


#=================================================================================================#
# FROM A STARTING MAP, ADD OTHER POINTS WITH A MINIMUM DISTANCE BETWEEN POINTS

def add_nearby_points(samples, n_new=100, scale=0.02, min_dist=0.015, seed=None):
    
    rng = np.random.default_rng(seed)
        
    """
    Generates new points near existing ones, avoiding edge accumulation and overlap.

    Parameters
    ----------
    samples : array, shape (N, 2)
        Existing coordinates (in mm)
    n_new : int
        Number of new points to generate
    scale : float
        Standard deviation of the jitter
    min_dist : float
        Minimum distance between new points and existing ones (in mm)
    rng : int or None
        Random seed
    """

    x_min, x_max = samples[:, 0].min(), samples[:, 0].max()
    y_min, y_max = samples[:, 1].min(), samples[:, 1].max()

    new_points = []

    while len(new_points) < n_new:
        # Select an existing point
        idx = rng.integers(len(samples))
        base = samples[idx]
        # Add jitter
        p = base + rng.normal(loc=0, scale=scale, size=2)

        # Check bounds
        if not (x_min <= p[0] <= x_max and y_min <= p[1] <= y_max):
            continue

        # Compute distances from existing and already generated new points
        all_points = np.vstack([samples, new_points]) if new_points else samples
        dists = np.linalg.norm(all_points - p, axis=1)

        # Accept only if minimum distance constraint is met
        if np.all(dists > min_dist):
            new_points.append(p)

    return np.array(new_points)

#================================================================================================#
# FROM A STARTING MAP, GENERATE A RANDOM TOPOLOGY

def random_EI_connectivity(samples_mm, net, frac_out_E=0.10, frac_out_I=0.125, sigma_exc=1.0, sigma_inh=1.0, seed=None):
    """
    Build a random EI connectivity test matrix:
      - E presyn: connect to both E and I (no self), out-degree ≈ frac_out_E * N
      - I presyn: connect to E only (no self), out-degree ≈ frac_out_I * N_E
      - No I->I connections
      - Weights: truncated Gaussians
          * E:  N(mean=+w_mean_exc,  sd=sigma_exc) truncated to >0
          * I:  N(mean=-w_mean_inh, sd=sigma_inh) truncated to <0
    Returns:
      W (NxN), dist_matrix (NxN)
    """
    rng = np.random.default_rng(seed)

    types = np.asarray(net['ntypes']).astype(bool)  # True=E, False=I
    mu_E  = float(net['w_mean_exc'])                # e.g., 6.0
    mu_I  = float(net['w_mean_inh'])                # e.g., 5.0  (used as magnitude; mean is -mu_I)

    N = samples_mm.shape[0]

    # --- distance matrix (mm)
    dist_matrix = np.linalg.norm(samples_mm[:, None, :] - samples_mm[None, :, :], axis=2)
    np.fill_diagonal(dist_matrix, 0.0)

    exc_idx = np.where(types)[0]
    inh_idx = np.where(~types)[0]

    W = np.zeros((N, N), dtype=float)

    # --- helper: truncated normal sampling (vectorized rejection)
    def trunc_pos(mean, sd, size):
        x = rng.normal(mean, sd, size=size)
        mask = x <= 0.0
        while np.any(mask):
            x[mask] = rng.normal(mean, sd, size=mask.sum())
            mask = x <= 0.0
        return x

    def trunc_neg(mean, sd, size):
        # mean here should be negative (e.g., -mu_I)
        x = rng.normal(mean, sd, size=size)
        mask = x >= 0.0
        while np.any(mask):
            x[mask] = rng.normal(mean, sd, size=mask.sum())
            mask = x >= 0.0
        return x

    # --- E presyn: connect to (all except self)
    kE = max(1, int(frac_out_E * N))
    for pre in exc_idx:
        allowed = np.arange(N)
        # remove self
        if kE >= N:
            targets = allowed[allowed != pre]
        else:
            # sample without self
            allowed = allowed[allowed != pre]
            k = min(kE, allowed.size)
            targets = rng.choice(allowed, size=k, replace=False)

        # weights: truncated positive around +mu_E
        W[pre, targets] = trunc_pos(mu_E, sigma_exc, size=targets.size)

    # --- I presyn: connect to E only
    nE = exc_idx.size
    if nE > 0:
        kI_base = max(1, int(frac_out_I * nE))
        for pre in inh_idx:
            # remove self if pre is in exc_idx (it isn't), so just use exc_idx
            k = min(kI_base, nE)
            targets = rng.choice(exc_idx, size=k, replace=False)

            # weights: truncated negative around -mu_I
            W[pre, targets] = trunc_neg(-mu_I, sigma_inh, size=targets.size)

    # --- no self-connections (already avoided for E; enforce anyway)
    np.fill_diagonal(W, 0.0)

    # --- no I->I (enforce explicitly)
    if inh_idx.size:
        W[np.ix_(inh_idx, inh_idx)] = 0.0

    return W, dist_matrix

#================================================================================================#
# EDR topology (optionally with degree modulated by the number of neighbours inside a circle)
from scipy.stats import truncnorm

# see Song et al. 2005; Lefort et al. 2009; Ikegaya et al., 2004, Science; Buzsáki & Mizuseki, 2014, Nat Rev Neurosci 
def sample_trunc_lognorm(mean, sigma, lower, upper, size=None):
    mu = np.log(mean**2 / np.sqrt(sigma**2 + mean**2))
    sigma_log = np.sqrt(np.log(1 + (sigma**2 / mean**2)))
    lo = max(lower, 1e-12)  # avoid log(0)
    a, b = (np.log(lo) - mu) / sigma_log, (np.log(upper) - mu) / sigma_log if np.isfinite(upper) else np.inf
    z = truncnorm.rvs(a, b, loc=mu, scale=sigma_log, size=size)
    z = np.exp(z)
    return float(z) if size is None else z # Returns a scalar if size=None, otherwise returns an array

#================================================================================================#
# FROM A STARTING MAP, GENERATE A SPATIALLY EMBEDDED NETWORK WITH EDR
#
# from the set of coordinates of nodes/neurons (samples_mm) and the neuron types (net['ntypes'] - boolean: True=E False=I)
#
# Connection probability:
# E –> * flollow an EDR
# I –> E short range connections (r-cutoff)
# I –> I optional (defauld: NO I–>I)
#
# Once the binary matrix of connections is done, weights are assignes:
# Weights strength: decay with distance
#

def EDR_Connections_clean(samples_mm, net,
                          p0_E=0.75, L_E=0.13, c_E=0.0,             # EDR (E presyn only): choose by units of samples_mm (mm!) 
                          r_IE=0.14, p_FS2E=0.75, p_LTS2E=0.75,     # I–>E rules
                          r_II=0.18, p_FS2FS=0.75, p_LTS2LTS=0.75,  # I->I rules
                          local_II=False, LTS_th=0.7,               # I–>I local rule
                          W_decay_exc=1.5, W_decay_inh=1.0,         # Weight means: exponential distance decay (mm⁻¹)
                          cv_exc=0.35, cv_inh=0.10, seed=None):     # Weight variability (CV)
    """
    Minimal, numerically robust EI connectivity for cultures.

    Topology:
      - E presyn: P_E(d) = c_E + (p0_E - c_E) * exp(-d / L_E) toward (E and I).
      - I presyn: local to E within r_I with prob p_I (if none, force nearest E).
      - No self; No I->I.

    Weights:
      - μ_E(d) = +w_mean_exc * exp(-W_decay_exc * d)
      - μ_I(d) = -w_mean_inh * exp(-W_decay_inh * d)
      - Gaussian with σ = CV * |μ(d)|, truncated to sign.
    """
    
    rng   = np.random.default_rng(seed)
    types = np.asarray(net['ntypes']).astype(bool)   # True=E, False=I
    muE0  = float(net['w_mean_exc'])                 # e.g., 6.0
    muI0  = float(net['w_mean_inh'])                 # e.g., 5.0
    N     = samples_mm.shape[0]

    # --- distances (mm)
    dist = np.linalg.norm(samples_mm[:, None, :] - samples_mm[None, :, :], axis=2)
    np.fill_diagonal(dist, 0.0)

    exc_idx = np.where(types)[0]
    inh_idx = np.where(~types)[0]

    # subtype split (FS vs LTS)
    nrands = net['nrands']
    inh_FS_idx  = inh_idx[nrands[inh_idx] <= LTS_th]
    inh_LTS_idx = inh_idx[nrands[inh_idx] >  LTS_th]

    # ----------------- 1) Topology
    binW = np.zeros((N, N), dtype=float)

    # --------- E presyn: EDR to all posts
    if exc_idx.size:
        # Safe probabilities in [0,1] if 0 <= c_E <= p0_E <= 1
        P_E = c_E + (p0_E - c_E) * np.exp(-dist[exc_idx, :] / max(L_E, 1e-9))
        P_E[np.arange(exc_idx.size), exc_idx] = 0.0
        P_E = np.clip(P_E, 0.0, 1.0)
        R   = rng.random(P_E.shape)
        binW[exc_idx, :] = (R < P_E).astype(float)

    # --------- FS presyn onto E
    for pre in inh_FS_idx:
        dE = dist[pre, exc_idx]
        candidates = exc_idx[dE <= r_IE]
        if candidates.size == 0:
            nearest = exc_idx[np.argmin(dE)]
            binW[pre, nearest] = 1.0
        else:
            sel = rng.random(candidates.size) < p_FS2E
            if not np.any(sel):
                nearest_local = candidates[np.argmin(dE[candidates])]
                binW[pre, nearest_local] = 1.0
            else:
                binW[pre, candidates[sel]] = 1.0

    # --------- LTS presyn onto E
    for pre in inh_LTS_idx:
        dE = dist[pre, exc_idx]
        candidates = exc_idx[dE <= r_IE]
        if candidates.size == 0:
            nearest = exc_idx[np.argmin(dE)]
            binW[pre, nearest] = 1.0
        else:
            sel = rng.random(candidates.size) < p_LTS2E
            if not np.any(sel):
                nearest_local = candidates[np.argmin(dE[candidates])]
                binW[pre, nearest_local] = 1.0
            else:
                binW[pre, candidates[sel]] = 1.0

    # --------- I presyn onto I (optional)
    if local_II:
        # FS->FS
        for pre in inh_FS_idx:
            dI = dist[pre, inh_FS_idx]
            candidates = inh_FS_idx[(dI <= r_II) & (inh_FS_idx != pre)]
            if candidates.size == 0:
                nearest = inh_FS_idx[np.argmin(dI)]
                binW[pre, nearest] = 1.0
            else:
                sel = rng.random(candidates.size) < p_FS2FS
                if not np.any(sel):
                    nearest_local = candidates[np.argmin(dist[pre, candidates])]
                    binW[pre, nearest_local] = 1.0
                else:
                    binW[pre, candidates[sel]] = 1.0

        # LTS->LTS
        for pre in inh_LTS_idx:
            dI = dist[pre, inh_LTS_idx]
            candidates = inh_LTS_idx[(dI <= r_II) & (inh_LTS_idx != pre)]
            if candidates.size == 0:
                nearest_local = inh_LTS_idx[np.argmin(dI)]
                binW[pre, nearest_local] = 1.0
            else:
                sel = rng.random(candidates.size) < p_LTS2LTS
                if not np.any(sel):
                    nearest_local = candidates[np.argmin(dist[pre, candidates])]
                    binW[pre, nearest_local] = 1.0
                else:
                    binW[pre, candidates[sel]] = 1.0
    
    np.fill_diagonal(binW, 0.0)

    # ----------------- 2) Weights (Gaussian w/ exp-decay mean, CV sigma, sign-truncated)
    W = np.zeros_like(binW, dtype=float)
    rows, cols = np.nonzero(binW)
    if rows.size:
        d_edges = dist[rows, cols]
        isE = types[rows]

        # Means
        muE = muE0 * np.exp(-W_decay_exc * d_edges[isE])     # > 0
        muI = -muI0 * np.exp(-W_decay_inh * d_edges[~isE])   # < 0
        # Sigmas
        sigE = np.maximum(cv_exc * muE, 1e-9)
        sigI = np.maximum(cv_inh * (-muI), 1e-9)

        # E edges (>0), rejection sampling
        if muE.size:
            x    = np.empty_like(muE)
            mask = np.ones_like(muE, dtype=bool)
            while np.any(mask):
                x[mask] = rng.normal(muE[mask], sigE[mask])
                mask = x <= 0.0
            W[rows[isE], cols[isE]] = x

        # I edges (<0), rejection sampling
        if muI.size:
            x    = np.empty_like(muI)
            mask = np.ones_like(muI, dtype=bool)
            while np.any(mask):
                x[mask] = rng.normal(muI[mask], sigI[mask])
                mask = x >= 0.0
            W[rows[~isE], cols[~isE]] = x

    np.fill_diagonal(W, 0.0)
    return W, dist

#================================================================================================#

