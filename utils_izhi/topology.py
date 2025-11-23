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
# RANDOM TOPOLOGY

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
# LAST VERSION 2 SEPT 2025 – MAYBE THE BEST

# funziona ma non connette selettivamente FS–FS, LTS-LTS, una o l'altra o nessuna o entrambe.
# questa decide solo se connettere i FS tra loro (funziona, comunque)
def EDR_Connections_clean_old(samples_mm, net,
                          p0_E=0.75, L_E=0.13, c_E=0.0,          # EDR (E presyn only): choose by units of samples_mm (mm!)
                          r_IE=0.14, p_IE=0.5,                     # I–>E local rule
                          r_II=0.75, p_II=0.75, local_II=False, LTS_th=0.7,            # I–>I local rule
                          W_decay_exc=1.5, W_decay_inh=1.0,      # Weight means: exponential distance decay (mm⁻¹)
                          cv_exc=.35, cv_inh=0.10, seed=None):  # Weight variability (CV)
    """

    con questa ottieni riverbero e tante altre belle cose che ottieni in coltura
    W, dist_mat = tp.EDR_Connections_clean_old(samples, net, p0_E=0.75, L_E=0.13, c_E=0.0, r_IE=0.18, p_IE=0.75, 
                                       r_II=0.18,  p_II=0.75, local_II=local_II, LTS_th=0.7,
                                       W_decay_exc=1.5, W_decay_inh=1., cv_exc=0.35, cv_inh=0.10, seed=None)
                                       
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

    # ----------------- 1) Topology
    binW = np.zeros((N, N), dtype=float)

    # E presyn: EDR to all posts
    if exc_idx.size:
        # Safe probabilities in [0,1] if 0 <= c_E <= p0_E <= 1
        P_E = c_E + (p0_E - c_E) * np.exp(-dist[exc_idx, :] / max(L_E, 1e-9))
        P_E[np.arange(exc_idx.size), exc_idx] = 0.0
        P_E = np.clip(P_E, 0.0, 1.0)
        R   = rng.random(P_E.shape)
        binW[exc_idx, :] = (R < P_E).astype(float)

    if local_II:
        nrands   = net['nrands']
        # I presyn: local I->E within r_I (Bernoulli p_I); ensure at least one
        inh_FS_idx = np.where( (types==False) & (nrands<=LTS_th) )[0]
        if inh_FS_idx.size and exc_idx.size:
            for pre in inh_FS_idx:
                dE = dist[pre, exc_idx]
                candidates = exc_idx[dE <= r_IE]
                if candidates.size == 0:
                    nearest = exc_idx[np.argmin(dE)]
                    binW[pre, nearest] = 1.0
                else:
                    sel = rng.random(candidates.size) < p_IE
                    if not np.any(sel):
                        nearest_local = candidates[np.argmin(dist[pre, candidates])]
                        binW[pre, nearest_local] = 1.0
                    else:
                        binW[pre, candidates[sel]] = 1.0
                    
        # I presyn: local I->I within r_II (fast spiking neurons); ensure at least one
        inh_FS_idx = np.where( (types==False) & (nrands<=LTS_th) )[0]
        if inh_FS_idx.size:
            for pre in inh_FS_idx:
                dI = dist[pre, inh_FS_idx]
                candidates = inh_FS_idx[(dI <= r_II) & (inh_FS_idx != pre)]
                if candidates.size == 0:
                    nearest = inh_FS_idx[np.argmin(dI)]
                    binW[pre, nearest] = 1.0
                else:
                    sel = rng.random(candidates.size) < p_II
                    if not np.any(sel):
                        nearest_local = candidates[np.argmin(dist[pre, candidates])]
                        binW[pre, nearest_local]   = 1.0
                    else:
                        binW[pre, candidates[sel]] = 1.0
    else:
        # I presyn: local I->E within r_I (Bernoulli p_I); ensure at least one
        if inh_idx.size and exc_idx.size:
            for pre in inh_idx:
                dE = dist[pre, exc_idx]
                candidates = exc_idx[dE <= r_IE]
                if candidates.size == 0:
                    nearest = exc_idx[np.argmin(dE)]
                    binW[pre, nearest] = 1.0
                else:
                    sel = rng.random(candidates.size) < p_IE
                    if not np.any(sel):
                        nearest_local = candidates[np.argmin(dist[pre, candidates])]
                        binW[pre, nearest_local] = 1.0
                    else:
                        binW[pre, candidates[sel]] = 1.0
                        
        # Forbid I->I and self
        if inh_idx.size:
            binW[np.ix_(inh_idx, inh_idx)] = 0.0
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


#--------------------------------------------------------------------------------------------------------------------------------#
    
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
    
def EDR_Connections(samples_mm, net, b_exc_factor=2.5, b_inh_factor=2.5,                                   # contro prob decay steepness
                    EI_boost = 1.5,                                                      # EI_boost: >1 makes E->I denser than E->E
                    IE_boost = 1.2,
                    enforce_inh_nearest=True, frac_inh_targets=0.125,                    # control out-deg of inh neus
                    enforce_inh_local_input=True, frac_inh_inputs=0.1,                   # control in-deg of inh neus
                    allow_I_to_I=True, frac_I_I=0.03,
                    w_decay=False, w_decay_ampli=2, W_decay_exc = 2.0, W_decay_inh = 2., # weights decay with distance
                    neighbors_prob=False, r_thresh=0.5, mod='out', seed=None):
    
    rng = np.random.default_rng(seed)
    
    # exc/inh
    types_mat      = net['ntypes']
    exc_idx        = np.where(types_mat)[0]
    inh_idx        = np.where(types_mat == False)[0]
    
    # mean value of exc and inh weights
    exc_links_mean = net['w_mean_exc']
    inh_links_mean = net['w_mean_inh']

    N_neurons      = len(samples_mm)
    
    # -------------------------------------------------------------------------------
    # Compute distance matrix
    dist_matrix = np.sqrt(np.sum((samples_mm[:, None, :] - samples_mm[None, :, :])**2, axis=2))
    np.fill_diagonal(dist_matrix, 0)

    # -------------------------------------------------------------------------------
    # Define exponential decay parameters (pre-fitted from empirical data)
    a, b, c = 0.18, 2.0, 0.0006
    
    # inhibitory neurons connect in a different way (in a closest ray) than excitatory ones
    base_prob_matrix = np.zeros((N_neurons, N_neurons))
    for pre in range(N_neurons):
        for post in range(N_neurons):
            d = dist_matrix[pre, post]
            if types_mat[pre]:  # exc
                base_prob_matrix[pre, post] =  a * np.exp(-b * b_exc_factor * d) + c 
            else:               # inh
                base_prob_matrix[pre, post] =  a * np.exp(-b * b_inh_factor * d) + c 
                
    # rescaling prob for inh
    exc_row_sums   = base_prob_matrix[exc_idx].sum(axis=1)
    inh_row_sums   = base_prob_matrix[inh_idx].sum(axis=1)
    exc_mean       = np.mean(exc_row_sums)
    inh_mean       = np.mean(inh_row_sums)
    scaling_factor = exc_mean / inh_mean
    base_prob_matrix[inh_idx] *= scaling_factor
    np.fill_diagonal(base_prob_matrix, 0)

    # -------------------------------------------------------------------------------
    # After building base_prob_matrix (or while building it), apply:
    if EI_boost != 1.0 and len(exc_idx) > 0 and len(inh_idx) > 0:
        # boost probabilities from excitatory rows to inhibitory columns
        base_prob_matrix[np.ix_(exc_idx, inh_idx)] *= EI_boost
    base_prob_matrix[np.ix_(inh_idx, exc_idx)] *= IE_boost

    # -------------------------------------------------------------------------------
    if neighbors_prob:
        # Count neighbors within threshold distance
        num_neighbors  = np.sum((dist_matrix < r_thresh) & (dist_matrix > 0), axis=1)
        norm_neighbors = num_neighbors / np.max(num_neighbors)
    else:
        norm_neighbors = np.ones(N_neurons)

    # -------------------------------------------------------------------------------
    # probability matrix
    prob_matrix = np.copy(base_prob_matrix)
    if mod=='in':
        # Modulate probability by in-degree weight (receiver-based modulation)
        for j in range(N_neurons):
            prob_matrix[:, j] *= norm_neighbors[j]
    else:
        # Modulate probability by out-degree weight (sender-based modulation)
        for i in range(N_neurons):
            prob_matrix[i, :] *= norm_neighbors[i]

    # normalization to keep probabilities in [0,1]
    prob_matrix /= np.max(prob_matrix)

    # Sample connections from probability matrix
    rand_mat    = rng.random((N_neurons, N_neurons))
    np.fill_diagonal(rand_mat, 1)  # Prevent self-connections
    bin_weights = (rand_mat < prob_matrix).astype(float)

    # -------------------------------------------------------------------------------
    # Out-degree constraint (rows): inhibitory neurons connect to nearest excitatory targets
    if enforce_inh_nearest:
        mask_out = np.zeros_like(bin_weights, dtype=float)
        n_targets = max(1, int(N_neurons * frac_inh_targets))
        for i in inh_idx:
            dist_order = np.argsort(dist_matrix[i, :])
            exc_sorted_by_dist = [j for j in dist_order if j in exc_idx and j != i]
            mask_out[i, exc_sorted_by_dist[:n_targets]] = 1.0
        # Overwrite inhibitory rows with the row-constraint mask
        bin_weights[inh_idx, :] = mask_out[inh_idx, :]
    
    # -------------------------------------------------------------------------------
    # In-degree constraint (columns): inhibitory neurons receive from nearest sources
    if enforce_inh_local_input:
        mask_in = np.zeros_like(bin_weights, dtype=float)
        n_inputs = max(1, int(N_neurons * frac_inh_inputs))
        for j in inh_idx:
            dist_order = np.argsort(dist_matrix[:, j])
            local_sources = [i for i in dist_order if i != j][:n_inputs]
            mask_in[local_sources, j] = 1.0
        # Overwrite inhibitory columns with the column-constraint mask
        bin_weights[:, inh_idx] = mask_in[:, inh_idx]
    
    # Always forbid I->I
    bin_weights[np.ix_(inh_idx, inh_idx)] = 0.0

    # --- Optional: add local I -> I connections (sparse and short-range) respecting mask_in
    if allow_I_to_I and len(inh_idx) > 0:
        n_ii_targets = max(1, int(len(inh_idx) * frac_I_I))
        for i in inh_idx:
            drow = dist_matrix[i, inh_idx]
            inh_targets_order = inh_idx[np.argsort(drow)]
            inh_targets_order = [j for j in inh_targets_order if j != i]
            for j in inh_targets_order[:n_ii_targets]:
                if (not enforce_inh_local_input) or (mask_in[i, j] == 1.0):
                    bin_weights[i, j] = 1.0

    #-----------------------------------------------------------------------------------#
    # Generate weights    
    weights = np.zeros_like(bin_weights)

    if w_decay:  
        # sampling log-normal distribution
        alpha_exc = exc_links_mean * w_decay_ampli
        alpha_inh = abs(inh_links_mean) * w_decay_ampli
    
        conn_idx = np.argwhere(bin_weights == 1)
        for pre, post in conn_idx:
            d = dist_matrix[pre, post]
            if types_mat[pre]:  # E → *
                mean_w = alpha_exc * np.exp(-W_decay_exc * d)
                weights[pre, post] = sample_trunc_lognorm(mean=mean_w, sigma=mean_w * 0.6,
                                                          lower=0, upper=np.inf, size=None)  # CV_E=0.6
            else:               # I → E
                mean_w = alpha_inh * np.exp(-W_decay_inh * d)
                weights[pre, post] = -sample_trunc_lognorm(mean=mean_w, sigma=mean_w * 0.4,
                                                           lower=0, upper=np.inf, size=None) # CV_I=0.4
    else:
        # exc weights (pos)
        exc_weights = sample_trunc_lognorm(mean=exc_links_mean, sigma=exc_links_mean * 0.6,
                                           lower=0, upper=np.inf, size=(N_neurons, N_neurons))
        # inh weights (neg)
        inh_weights = -sample_trunc_lognorm(mean=abs(inh_links_mean), sigma=abs(inh_links_mean) * 0.4,
                                            lower=0, upper=np.inf, size=(N_neurons, N_neurons))
        for pre in range(N_neurons):
            if types_mat[pre]:
                weights[pre, :] = exc_weights[pre, :]
            else:
                weights[pre, :] = inh_weights[pre, :]

        weights *= bin_weights

        ## Renormalize weights after probability modulation to avoid shrinking too much
        exc_mask = (types_mat[:, None] == 1) & (bin_weights == 1)
        inh_mask = (types_mat[:, None] == 0) & (bin_weights == 1)

        if np.any(exc_mask):
            mean_exc_after = np.mean(weights[exc_mask])
            if mean_exc_after > 0:
                weights[exc_mask] *= (exc_links_mean / mean_exc_after)

        if np.any(inh_mask):
            mean_inh_after = np.mean(np.abs(weights[inh_mask]))
            if mean_inh_after > 0:
                weights[inh_mask] *= (abs(inh_links_mean) / mean_inh_after)

    np.fill_diagonal(weights, 0)

    return weights, dist_matrix


#================================================================================================#

def EDR_Connections_V2(samples_mm, net, b_exc_factor = 1., b_inh_factor=2.5, w_decay_ampli=2, W_decay_exc = 2.0, W_decay_inh = 2., 
                       enforce_inh_nearest=True, frac_inh_targets=0.125,                    # control out-deg of inh neus
                       enforce_inh_local_input=True, frac_inh_inputs=0.1,                   # control in-deg of inh neus
                       neighbors_prob=False, r_thresh=0.5, mod='out', seed=None):
    
    rng = np.random.default_rng(seed)
    
    # exc/inh
    types_mat      = net['ntypes']
    exc_idx        = np.where(types_mat)[0]
    inh_idx        = np.where(types_mat == False)[0]
    # mean value of exc and inh weights
    exc_links_mean = net['w_mean_exc']
    inh_links_mean = net['w_mean_inh']

    N_neurons     = len(samples_mm)

    # Compute distance matrix
    dist_matrix = np.sqrt(np.sum((samples_mm[:, None, :] - samples_mm[None, :, :])**2, axis=2))
    np.fill_diagonal(dist_matrix, 0)

    # Define exponential decay parameters (pre-fitted from empirical data)
    a, b, c = 0.18, 2.0, 0.0006
    #base_prob_matrix = a * np.exp(-b * dist_matrix) + c
    
    # inhibitory neurons connect more densely in a closest ray than excitatory ones
    base_prob_matrix = np.zeros((N_neurons, N_neurons))
    for pre in range(N_neurons):
        for post in range(N_neurons):
            d = dist_matrix[pre, post]
            if types_mat[pre]:  # exc
                base_prob_matrix[pre, post] =  a * np.exp(-b * b_exc_factor * d) + c 
            else:              # inh, faster decay
                base_prob_matrix[pre, post] =  a * np.exp(-b * b_inh_factor * d) + c 
    # rescaling prob for inh
    exc_row_sums   = base_prob_matrix[exc_idx].sum(axis=1)
    inh_row_sums   = base_prob_matrix[inh_idx].sum(axis=1)
    exc_mean       = np.mean(exc_row_sums)
    inh_mean       = np.mean(inh_row_sums)
    scaling_factor = exc_mean / inh_mean
    base_prob_matrix[inh_idx] *= scaling_factor
    np.fill_diagonal(base_prob_matrix, 0)

    if neighbors_prob:
        # Count neighbors within threshold distance
        num_neighbors  = np.sum((dist_matrix < r_thresh) & (dist_matrix > 0), axis=1)
        norm_neighbors = num_neighbors / np.max(num_neighbors)
    else:
        norm_neighbors = np.ones((N_neurons, N_neurons))
        
    prob_matrix = np.copy(base_prob_matrix)
    if mod=='in':
        # Modulate probability by in-degree weight (receiver-based modulation)
        for j in range(N_neurons):
            prob_matrix[:, j] *= norm_neighbors[j]
    else:
        # Modulate probability by out-degree weight (sender-based modulation)
        for i in range(N_neurons):
            prob_matrix[i, :] *= norm_neighbors[i]

    # normalization to keep probabilities in [0,1]
    prob_matrix /= np.max(prob_matrix)

    # Sample connections from probability matrix
    rand_mat    = rng.random((N_neurons, N_neurons)) 
    np.fill_diagonal(rand_mat, 1)  # Prevent self-connections
    bin_weights = (rand_mat < prob_matrix).astype(float)

    # -------------------------------------------------------------------------------
    # Out-degree constraint (rows): inhibitory neurons connect to nearest excitatory targets
    if enforce_inh_nearest:
        mask_out = np.zeros_like(bin_weights, dtype=float)
        n_targets = max(1, int(N_neurons * frac_inh_targets))
        for i in inh_idx:
            dist_order = np.argsort(dist_matrix[i, :])
            exc_sorted_by_dist = [j for j in dist_order if j in exc_idx and j != i]
            mask_out[i, exc_sorted_by_dist[:n_targets]] = 1.0
        # Overwrite inhibitory rows with the row-constraint mask
        bin_weights[inh_idx, :] = mask_out[inh_idx, :]
    
    # -------------------------------------------------------------------------------
    # In-degree constraint (columns): inhibitory neurons receive from nearest sources
    '''if enforce_inh_local_input:
        mask_in = np.zeros_like(bin_weights, dtype=float)
        n_inputs = max(1, int(N_neurons * frac_inh_inputs))
        for j in inh_idx:
            dist_order = np.argsort(dist_matrix[:, j])
            local_sources = [i for i in dist_order if i != j][:n_inputs]
            mask_in[local_sources, j] = 1.0
        # Overwrite inhibitory columns with the column-constraint mask
        bin_weights[:, inh_idx] = mask_in[:, inh_idx]'''
    
    # Always forbid I->I
    bin_weights[np.ix_(inh_idx, inh_idx)] = 0.0
    # -------------------------------------------------------------------------------

    # Generate weights  
    weights = np.zeros_like(bin_weights)
    
    from scipy.stats import truncnorm

    def sample_truncated_normal(mean, scale, lower, upper):
        '''Samples a cutted normal betw [lower, upper]'''
        a, b = (lower - mean) / scale, (upper - mean) / scale
        return truncnorm.rvs(a, b, loc=mean, scale=scale)

    # exp decay params
    alpha_exc = exc_links_mean * w_decay_ampli     # avg weight at 0 dist
    alpha_inh = inh_links_mean * w_decay_ampli

    conn_idx = np.argwhere(bin_weights == 1)
    for pre, post in conn_idx:
        d = dist_matrix[pre, post]
        if types_mat[pre]:  # E → *
            mean_w = alpha_exc * np.exp(-W_decay_exc * d)
            weights[pre, post] = sample_truncated_normal(mean_w, 1, 0, np.inf)
        else:               # I → E
            mean_w = -alpha_inh * np.exp(-W_decay_inh * d)
            weights[pre, post] = sample_truncated_normal(mean_w, 1, -np.inf, 0)

    np.fill_diagonal(weights, 0)

    return weights, dist_matrix

#================================================================================================#
# EDR topology with log-normal degree distribution

def EDR_Connections_LogNormal(samples_mm, net, r_thresh=0.5, prob_scale=1,
                                  b_exc_factor=1., b_inh_factor=2.5,
                                  w_decay_ampli=2, W_decay_exc=2.0, W_decay_inh=2.,
                                  mu_ln=3.2, sigma_ln=0.6, seed=None):
    
    rng = np.random.default_rng(seed)
    
    from scipy.stats import truncnorm

    # exc/inh
    types_mat      = net['ntypes']
    exc_links_mean = net['w_mean_exc']
    inh_links_mean = net['w_mean_inh']
    N_neurons      = len(samples_mm)

    # Distance matrix
    dist_matrix = np.sqrt(np.sum((samples_mm[:, None, :] - samples_mm[None, :, :])**2, axis=2))
    np.fill_diagonal(dist_matrix, 0)

    # --- Parametri decadimento esponenziale
    a, b, c = 0.18, 2.0, 0.0006

    # --- Log-normal out-degrees
    out_deg = rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=N_neurons)
    out_deg = np.round(out_deg).astype(int)
    out_deg = np.clip(out_deg, 0, N_neurons - 1)

    bin_weights = np.zeros((N_neurons, N_neurons))

    for pre in range(N_neurons):
        probs = np.zeros(N_neurons)
        for post in range(N_neurons):
            d = dist_matrix[pre, post]
            if types_mat[pre]:  # exc
                probs[post] = a * np.exp(-b * b_exc_factor * d) + c
            else:               # inh
                b_inh = b * b_inh_factor
                probs[post] = a * np.exp(-b_inh * b_exc_factor * d) + c

        probs /= probs.sum()*prob_scale

        k = out_deg[pre]
        if k > 0:
            posts = rng.choice(N_neurons, size=min(k, N_neurons-1), replace=False, p=probs)
            bin_weights[pre, posts] = 1.0

    # Remove I→I
    inh_idx = np.where(types_mat == False)[0]
    bin_weights[np.ix_(inh_idx, inh_idx)] = 0

    # --- Generate weights
    from scipy.stats import truncnorm
    def sample_truncated_normal(mean, scale, lower, upper):
        a, b = (lower - mean) / scale, (upper - mean) / scale
        return truncnorm.rvs(a, b, loc=mean, scale=scale)

    alpha_exc = exc_links_mean * w_decay_ampli
    alpha_inh = inh_links_mean * w_decay_ampli

    weights = np.zeros_like(bin_weights)
    for pre, post in np.argwhere(bin_weights == 1):
        d = dist_matrix[pre, post]
        if types_mat[pre]:
            mean_w = alpha_exc * np.exp(-W_decay_exc * d)
            weights[pre, post] = sample_truncated_normal(mean_w, 1, 0, np.inf)
        else:
            mean_w = -alpha_inh * np.exp(-W_decay_inh * d)
            weights[pre, post] = sample_truncated_normal(mean_w, 1, -np.inf, 0)

    np.fill_diagonal(weights, 0)
    np.fill_diagonal(dist_matrix, 0)

    return weights, dist_matrix
    
#================================================================================================#
# delay matrix
'''
def set_delays(distance_mat,weights_mat,delay_max=20,deltat=1,topoType='out', seed=None):
    
    rng = np.random.default_rng(seed)
        
    delay_max = delay_max*deltat
    if topoType == 'random':
        del_mat = rng.integers(1*deltat, high=delay_max+1, size=distance_mat.shape, dtype=int)
    else:
        del_mat = np.copy(distance_mat)/np.max(distance_mat.flatten())
        del_mat = (del_mat*delay_max).astype(int)+1
    del_mat[weights_mat<0]  = 1*deltat
    del_mat[weights_mat==0] = 0
    return del_mat


def set_delays(distance_mat, weights_mat, deltat=1.0, topoType='distance', delay_max=20, conduction_speed=None, use_lognormal_speed=False, 
               mean_speed=0.10, cv_speed=0.30, separate_inhibitory_speed=True, inh_speed=0.12, min_delay_steps=1, 
               force_inhibitory_min_delay=True, delay_jitter_steps=0, seed=None):
    
    rng = np.random.default_rng(seed)

    assert distance_mat.shape == weights_mat.shape, "distance_mat and weights_mat must have same shape"

    # Masks
    no_conn = (weights_mat == 0)
    has_conn = ~no_conn
    is_inhib = (weights_mat < 0)

    if topoType == 'random':
        delay_matrix = rng.integers(1, delay_max + 1, size=distance_mat.shape, dtype=int)
    
    else:
        # Determine speed matrix
        if conduction_speed is not None:
            speed_matrix = np.full_like(distance_mat, float(conduction_speed), dtype=float) \
                           if np.isscalar(conduction_speed) else np.array(conduction_speed, dtype=float)
        else:
            if use_lognormal_speed:
                sigma_log = np.sqrt(np.log(1.0 + cv_speed**2))
                mu_log    = np.log(mean_speed) - 0.5 * sigma_log**2
                speed_matrix = rng.lognormal(mean=mu_log, sigma=sigma_log, size=distance_mat.shape)
            else:
                speed_matrix = np.full_like(distance_mat, float(mean_speed), dtype=float)

            if separate_inhibitory_speed:
                speed_matrix[is_inhib] = float(inh_speed)

        # Avoid division by zero and compute delays
        speed_matrix = np.clip(speed_matrix, 1e-6, None)
        delay_matrix = np.rint((distance_mat / speed_matrix) / deltat).astype(int)

        # Enforce minimum delay where connection exists
        delay_matrix[has_conn] = np.maximum(delay_matrix[has_conn], min_delay_steps)

    # Force inhibitory delays to min
    if force_inhibitory_min_delay:
        delay_matrix[is_inhib] = min_delay_steps

    # Optional jitter
    if delay_jitter_steps > 0:
        jitter = rng.integers(-delay_jitter_steps, delay_jitter_steps + 1, size=delay_matrix.shape, dtype=int)
        jitter[no_conn] = 0
        delay_matrix += jitter
        delay_matrix[has_conn] = np.maximum(delay_matrix[has_conn], min_delay_steps)

    # Ensure non-links are 0
    delay_matrix[no_conn] = 0

    return delay_matrix
'''

def set_delays(distance_mat, weights_mat, delay_max=20, deltat=1, min_delay_steps=1,  jitter_exc_steps=1, topoType='out', seed=None):
    
    rng = np.random.default_rng(seed)

    # Masks
    no_conn = (weights_mat == 0)
    has_conn = ~no_conn
    is_inhib_edge = (weights_mat < 0)
    is_excit_edge = (weights_mat > 0)
    
    delay_max = delay_max*deltat
    
    if topoType == 'random':
        del_mat = rng.integers(1*deltat, high=delay_max+1, size=distance_mat.shape, dtype=int)
    else:
        del_mat = np.copy(distance_mat)/np.max(distance_mat.flatten())
        del_mat = (del_mat*delay_max).astype(int)+1
    del_mat[weights_mat<0]  = int(min_delay_steps/deltat)

    # Optional excitatory jitter (0..jitter_exc_steps)
    if jitter_exc_steps > 0:
        jitter = rng.integers(0, jitter_exc_steps + 1, size=del_mat.shape, dtype=int)
        jitter[~is_excit_edge] = 0
        del_mat += jitter
        # Re-enforce minimum and clip by delay_max if provided
        delay_matrix[has_conn] = np.maximum(del_mat[has_conn], int(min_delay_steps/deltat))
        if delay_max is not None:
            del_mat[has_conn] = np.minimum(del_mat[has_conn], delay_max)

    del_mat[weights_mat==0] = 0
    
    return del_mat





# LAST VERSION 2 SEPT 2025
def set_delays(distance_mat, weights_mat, deltat=1.0, topoType='distance', delay_max=20,
               conduction_speed=None, use_lognormal_speed=False,
               mean_speed=0.30, cv_speed=0.30, separate_inhibitory_speed=True, inh_speed=0.30,
               min_delay_steps=1, force_inhibitory_min_delay=True, jitter_exc_steps=1, seed=None):
    """Build an integer delay matrix from distances and optional conduction speeds.

    Notes:
      - `weights_mat` orientation must match your code: rows=presyn, cols=postsyn.
      - Inhibitory edges are detected as weights < 0.
      - If `topoType` is 'out', it's treated like 'distance'.
      - Jitter is applied to excitatory edges only (0..jitter_exc_steps).
    """
    rng = np.random.default_rng(seed)

    assert distance_mat.shape == weights_mat.shape, "distance_mat and weights_mat must have same shape"

    # Masks
    no_conn = (weights_mat == 0)
    has_conn = ~no_conn
    is_inhib_edge = (weights_mat < 0)
    is_excit_edge = (weights_mat > 0)

    topoType = 'distance' if topoType in ('distance', 'out') else topoType

    if topoType == 'random':
        delay_matrix = rng.integers(min_delay_steps, delay_max + 1, size=distance_mat.shape, dtype=int)
    else:
        # Determine speed matrix
        if conduction_speed is not None:
            if np.isscalar(conduction_speed):
                speed_matrix = np.full_like(distance_mat, float(conduction_speed), dtype=float)
            else:
                speed_matrix = np.array(conduction_speed, dtype=float)
        else:
            if use_lognormal_speed:
                sigma_log = np.sqrt(np.log(1.0 + cv_speed**2))
                mu_log    = np.log(mean_speed) - 0.5 * sigma_log**2
                speed_matrix = rng.lognormal(mean=mu_log, sigma=sigma_log, size=distance_mat.shape)
            else:
                speed_matrix = np.full_like(distance_mat, float(mean_speed), dtype=float)

            if separate_inhibitory_speed:
                speed_matrix[is_inhib_edge] = float(inh_speed)

        # Avoid division by zero and compute delays
        speed_matrix = np.clip(speed_matrix, 1e-6, None)
        delay_matrix = np.rint((distance_mat / speed_matrix) / max(deltat, 1e-9)).astype(int)

        # Enforce minimum delay where connection exists
        delay_matrix[has_conn] = np.maximum(delay_matrix[has_conn], min_delay_steps)

    # Force inhibitory delays to minimum (fast GABA), if requested — acts on outgoing I edges
    if force_inhibitory_min_delay:
        delay_matrix[is_inhib_edge] = int(min_delay_steps/deltat)

    # Optional excitatory jitter (0..jitter_exc_steps)
    if jitter_exc_steps > 0:
        jitter = rng.integers(0, jitter_exc_steps + 1, size=delay_matrix.shape, dtype=int)
        jitter[~is_excit_edge] = 0
        delay_matrix += jitter
        # Re-enforce minimum and clip by delay_max if provided
        delay_matrix[has_conn] = np.maximum(delay_matrix[has_conn], int(min_delay_steps/deltat))
        if delay_max is not None:
            delay_matrix[has_conn] = np.minimum(delay_matrix[has_conn], delay_max)

    # Zero out non-links
    delay_matrix[no_conn] = 0

    return delay_matrix







def EDR_Connections_V3_balanced(
    samples_mm, ntypes,
    # out-degree fractions
    frac_out_E=0.04,            # fraction of total neurons targeted by each excitatory neuron
    frac_out_I_E=0.06,          # fraction of excitatory neurons targeted by each inhibitory neuron
    frac_out_I_I=0.01,          # fraction of inhibitory neurons targeted by each inhibitory neuron
    # spatial decay constants [mm]
    lambda_E=0.35, lambda_I=0.30,
    # baseline weights at zero distance
    w0_E=1.0,  w0_I=0.8,
    # exponential decay of weights with distance
    wdecay_E=1.5, wdecay_I=1.8,
    # weight variability
    sigma_E=0.35, sigma_I=0.30,
    # balance control: limit total inhibition relative to excitation per postsynaptic neuron
    cap_I_over_E=0.80,
    seed=None
):
    rng = np.random.default_rng(seed)
    N = samples_mm.shape[0]
    types = np.asarray(ntypes).astype(bool)  # True = excitatory
    E_idx = np.where(types)[0]
    I_idx = np.where(~types)[0]
    N_E, N_I = E_idx.size, I_idx.size

    # Pairwise distances between neurons (mm)
    D = np.linalg.norm(samples_mm[:, None, :] - samples_mm[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)  # exclude self-connections in probability calculation

    # --- Helper functions -------------------------------------------------------

    def row_probs(j, allowed, lam):
        """Compute connection probability from presynaptic j to allowed targets with exponential decay."""
        if allowed.size == 0:
            return np.empty(0)
        p = np.exp(-D[j, allowed] / lam)
        s = p.sum()
        return p / s if s > 0 else np.ones(allowed.size) / allowed.size

    def trunc_pos(mean, sd, size):
        """Sample positive weights from a truncated normal distribution."""
        mean = np.asarray(mean)
        x = rng.normal(mean, sd, size=size)
        m = x <= 0.0
        if mean.ndim == 0:
            while np.any(m):
                x[m] = rng.normal(mean, sd, size=m.sum())
                m = x <= 0.0
        else:
            while np.any(m):
                x[m] = rng.normal(mean[m], sd, size=m.sum())
                m = x <= 0.0
        return x

    def trunc_neg(mean_pos, sd, size):
        """Sample negative weights by flipping the sign of a positive truncated normal."""
        mean_pos = np.asarray(mean_pos)
        x = -rng.normal(mean_pos, sd, size=size)
        m = x >= 0.0
        if mean_pos.ndim == 0:
            while np.any(m):
                x[m] = -rng.normal(mean_pos, sd, size=m.sum())
                m = x >= 0.0
        else:
            while np.any(m):
                x[m] = -rng.normal(mean_pos[m], sd, size=m.sum())
                m = x >= 0.0
        return x

    # --- Connectivity matrix ----------------------------------------------------

    W = np.zeros((N, N), dtype=float)

    # Excitatory presynaptic neurons
    kE = max(1, int(frac_out_E * N))
    for j in E_idx:
        allowed = np.arange(N)
        allowed = allowed[allowed != j]
        p = row_probs(j, allowed, lambda_E)
        k = min(kE, allowed.size)
        tgt = rng.choice(allowed, size=k, replace=False, p=p)
        mean_w = w0_E * np.exp(-wdecay_E * D[j, tgt])
        W[j, tgt] = trunc_pos(mean_w, sigma_E, size=k)

    # Inhibitory presynaptic neurons
    kI_E = max(0, int(frac_out_I_E * N_E)) if N_E > 0 else 0
    kI_I = max(0, int(frac_out_I_I * N_I)) if N_I > 1 else 0
    for j in I_idx:
        # I -> E
        if kI_E > 0 and N_E > 0:
            allowedE = E_idx
            pE = row_probs(j, allowedE, lambda_I)
            k = min(kI_E, allowedE.size)
            tE = rng.choice(allowedE, size=k, replace=False, p=pE)
            mean_wE = w0_I * np.exp(-wdecay_I * D[j, tE])
            W[j, tE] = trunc_neg(mean_wE, sigma_I, size=k)
        # I -> I
        if kI_I > 0 and N_I > 1:
            allowedI = I_idx[I_idx != j]
            if allowedI.size > 0:
                pI = row_probs(j, allowedI, lambda_I)
                k = min(kI_I, allowedI.size)
                tI = rng.choice(allowedI, size=k, replace=False, p=pI)
                mean_wI = 0.3 * w0_I * np.exp(-wdecay_I * D[j, tI])  # weaker I->I
                W[j, tI] = trunc_neg(mean_wI, sigma_I, size=k)

    np.fill_diagonal(W, 0.0)

    # --- Balance step: rescale inhibition if too strong -------------------------

    if cap_I_over_E is not None:
        E_mask = types
        I_mask = ~types
        eps = 1e-12
        for i in range(N):
            E_sum = W[E_mask, i].sum()
            I_mag = -W[I_mask, i].sum()
            if E_sum > eps and I_mag > cap_I_over_E * E_sum:
                scale = (cap_I_over_E * E_sum) / (I_mag + eps)
                W[I_mask, i] *= scale

    # Distance matrix with zeros on the diagonal (used for delays)
    D_dist = D.copy()
    np.fill_diagonal(D_dist, 0.0)

    return W, D_dist

'''
# usable version end augurst – with the 2 functions of pruning
def EDR_Connections_V2(samples_mm, net, b_exc_factor = 1., b_inh_factor=2.5, 
                       w_decay_ampli=2, W_decay_exc = 2.0, W_decay_inh = 2., 
                       sigma_E=1., sigma_I=1.,
                       enforce_inh_nearest=True, frac_inh_targets=0.125,                    # control out-deg of inh neus
                       enforce_inh_local_input=True, frac_inh_inputs=0.1,                   # control in-deg of inh neus
                       neighbors_prob=False, r_thresh=0.5, mod='out', seed=None):
    
    rng = np.random.default_rng(seed)
    
    # exc/inh
    types_mat      = net['ntypes']
    exc_idx        = np.where(types_mat)[0]
    inh_idx        = np.where(types_mat == False)[0]
    # mean value of exc and inh weights
    exc_links_mean = net['w_mean_exc']
    inh_links_mean = net['w_mean_inh']

    N_neurons     = len(samples_mm)

    # Compute distance matrix
    dist_matrix = np.sqrt(np.sum((samples_mm[:, None, :] - samples_mm[None, :, :])**2, axis=2))
    np.fill_diagonal(dist_matrix, 0)

    # Define exponential decay parameters (pre-fitted from empirical data)
    a, b, c = 0.18, 2.0, 0.0006
    #base_prob_matrix = a * np.exp(-b * dist_matrix) + c
    
    # inhibitory neurons connect more densely in a closest ray than excitatory ones
    base_prob_matrix = np.zeros((N_neurons, N_neurons))
    for pre in range(N_neurons):
        for post in range(N_neurons):
            d = dist_matrix[pre, post]
            if types_mat[pre]:  # exc
                base_prob_matrix[pre, post] =  a * np.exp(-b * b_exc_factor * d) + c 
            else:              # inh, faster decay
                base_prob_matrix[pre, post] =  a * np.exp(-b * b_inh_factor * d) + c 
    # rescaling prob for inh
    exc_row_sums   = base_prob_matrix[exc_idx].sum(axis=1)
    inh_row_sums   = base_prob_matrix[inh_idx].sum(axis=1)
    exc_mean       = np.mean(exc_row_sums)
    inh_mean       = np.mean(inh_row_sums)
    scaling_factor = exc_mean / inh_mean
    base_prob_matrix[inh_idx] *= scaling_factor
    np.fill_diagonal(base_prob_matrix, 0)

    if neighbors_prob:
        # Count neighbors within threshold distance
        num_neighbors  = np.sum((dist_matrix < r_thresh) & (dist_matrix > 0), axis=1)
        norm_neighbors = num_neighbors / np.max(num_neighbors)
    else:
        norm_neighbors = np.ones((N_neurons, N_neurons))
        
    prob_matrix = np.copy(base_prob_matrix)
    if mod=='in':
        # Modulate probability by in-degree weight (receiver-based modulation)
        for j in range(N_neurons):
            prob_matrix[:, j] *= norm_neighbors[j]
    else:
        # Modulate probability by out-degree weight (sender-based modulation)
        for i in range(N_neurons):
            prob_matrix[i, :] *= norm_neighbors[i]

    # normalization to keep probabilities in [0,1]
    prob_matrix /= np.max(prob_matrix)

    # Sample connections from probability matrix
    rand_mat    = rng.random((N_neurons, N_neurons)) 
    np.fill_diagonal(rand_mat, 1)  # Prevent self-connections
    bin_weights = (rand_mat < prob_matrix).astype(float)

    # -------------------------------------------------------------------------------
    # Out-degree constraint (rows): inhibitory neurons connect to nearest excitatory targets
    if enforce_inh_nearest:
        mask_out = np.zeros_like(bin_weights, dtype=float)
        n_targets = max(1, int(N_neurons * frac_inh_targets))
        for i in inh_idx:
            dist_order = np.argsort(dist_matrix[i, :])
            exc_sorted_by_dist = [j for j in dist_order if j in exc_idx and j != i]
            mask_out[i, exc_sorted_by_dist[:n_targets]] = 1.0
        # Overwrite inhibitory rows with the row-constraint mask
        bin_weights[inh_idx, :] = mask_out[inh_idx, :]
    
    # Always forbid I->I
    bin_weights[np.ix_(inh_idx, inh_idx)] = 0.0
    
    # -------------------------------------------------------------------------------

    # Generate weights  
    weights = np.zeros_like(bin_weights)
    
    from scipy.stats import truncnorm

    def sample_truncated_normal(mean, scale, lower, upper):
        a, b = (lower - mean) / scale, (upper - mean) / scale
        return truncnorm.rvs(a, b, loc=mean, scale=scale)

    # exp decay params
    alpha_exc = exc_links_mean * w_decay_ampli     # avg weight at 0 dist
    alpha_inh = inh_links_mean * w_decay_ampli

    conn_idx = np.argwhere(bin_weights == 1)
    for pre, post in conn_idx:
        d = dist_matrix[pre, post]
        if types_mat[pre]:  # E → *
            mean_w = alpha_exc * np.exp(-W_decay_exc * d)
            weights[pre, post] = sample_truncated_normal(mean_w, sigma_E, 0, np.inf)
        else:               # I → E
            mean_w = -alpha_inh * np.exp(-W_decay_inh * d)
            weights[pre, post] = sample_truncated_normal(mean_w, sigma_I, -np.inf, 0)

    np.fill_diagonal(weights, 0)

    return weights, dist_matrix
'''



def boost_EI_motifs(
    W, pos, ntypes, 
    frac_I_fast=0.12,         # frazione I che diventa "I-hub" (locali/fast)
    frac_I_long=0.10,         # frazione I con proiezioni più lunghe (surround)
    r_fast=0.12,              # raggio (mm) per potenziamento locale I-hub
    r_EI_in=0.12,             # raggio (mm) per potenziare E->I verso hub
    f_EI_in=1.5,              # moltiplicatore pesi E->I (locali) verso hub
    f_IE_out=1.6,             # moltiplicatore pesi I->E (locali) dagli hub
    add_IE_k=5,               # max nuovi target E per ciascun I-hub (locali)
    long_k=3,                 # nuovi target E lontani per ciascun I-long
    rng=None
):
    """
    Potenzia motivi E–I per generare risposte sito-dipendenti:
      - I-hub locali (feed-forward e perisomatic): E->I↑ e I->E↑ entro r_fast/r_EI_in
      - I-long (surround): pochi archi I->E a lunga distanza
      - Un filo di I->I tra I-hub per terminazione rapida
    Mantiene segni: E +, I -
    """
    W = np.asarray(W, dtype=np.float64).copy()
    pos = np.asarray(pos, dtype=np.float64)
    E = np.asarray(ntypes, dtype=bool); I = ~E
    N = W.shape[0]
    if rng is None: rng = np.random.default_rng()

    # utilità
    def nonzero_rows_cols(M):
        rr, cc = np.where(M)
        return rr, cc

    # magnitudini tipiche correnti (per nuovi archi)
    IE = np.where(W[I][:, E] < 0, -W[I][:, E], 0.0)
    existing_IE = IE[IE > 0]
    med_IE = np.median(existing_IE) if existing_IE.size else 1.0

    II = np.where(W[I][:, I] < 0, -W[I][:, I], 0.0)
    med_II = np.median(II[II > 0]) if II.size else 0.1 * med_IE

    # --- punteggi per identificare I-hub (inibizione verso E) ---
    score_I_toE = (-W[:, E]).clip(min=0).sum(axis=1)   # out-strength |neg| verso E
    deg_I_toE   = (W[:, E] < 0).sum(axis=1)

    I_idx = np.where(I)[0]
    if I_idx.size == 0:
        return W

    # ordina I per (score, degree)
    order_I = I_idx[np.lexsort((-deg_I_toE[I], -score_I_toE[I]))]

    n_fast = max(1, int(frac_I_fast * I_idx.size))
    n_long = max(1, int(frac_I_long * I_idx.size))

    I_fast = order_I[:n_fast]
    rest   = np.setdiff1d(I_idx, I_fast, assume_unique=False)
    if rest.size < n_long:
        n_long = rest.size
    I_long = rng.choice(rest, size=n_long, replace=False) if n_long > 0 else np.array([], dtype=int)

    # precompute distanze
    # (per evitare NxN completo, usiamo differenze contro set mirati)
    # --- 1) Potenziamento locale attorno agli I-hub ---
    for j in I_fast:
        d = np.linalg.norm(pos - pos[j], axis=1)

        # potenzia E->I su E vicini (feed-forward forte)
        e_near = np.where((E) & (d <= r_EI_in))[0]
        if e_near.size:
            W[np.ix_(e_near, [j])] *= f_EI_in  # E->j (positivo * >1)

        # potenzia I->E locali (perisomatic/somatic targeting)
        e_out = np.where((E) & (d <= r_fast))[0]
        if e_out.size:
            W[np.ix_([j], e_out)] *= f_IE_out  # j->E (negativo * >1 -> più negativo)

        # aggiungi pochi nuovi I->E locali (se mancanti)
        if add_IE_k > 0 and e_out.size:
            # candidati non connessi
            mask_missing = W[j, e_out] == 0.0
            cand = e_out[mask_missing]
            if cand.size:
                k = min(add_IE_k, cand.size)
                # prendi i più vicini
                cand = cand[np.argsort(d[cand])[:k]]
                # pesi nuovi: negativi ~ med_IE, con piccola variabilità
                W[j, cand] = - med_IE * rng.lognormal(mean=0.0, sigma=0.25, size=k)

    # --- 2) I "long-range": pochi archi a lunga distanza per surround ---
    if I_long.size:
        # soglia distanza: prendiamo top 30% più lontani
        all_d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)  # N x N
        # (se N grande e vuoi risparmiare, sostituisci con topk approx)
        for j in I_long:
            d = all_d[j, :]
            e_far = np.where(E)[0]
            if e_far.size == 0: 
                continue
            # ordina per distanza decrescente e scegli target non già connessi
            order = e_far[np.argsort(d[e_far])[::-1]]
            order = order[W[j, order] == 0.0]
            k = min(long_k, order.size)
            if k > 0:
                tgt = order[:k]
                W[j, tgt] = - 0.7 * med_IE * rng.lognormal(mean=0.0, sigma=0.25, size=k)

    # --- 3) un filo di I->I tra I-hub per trimming rapido ---
    # collega ogni I_fast ai 1–2 I più vicini (se non connessi)
    for j in I_fast:
        dI = np.linalg.norm(pos[I_idx] - pos[j], axis=1)
        ord_local = I_idx[np.argsort(dI)]
        ord_local = ord_local[ord_local != j]
        if ord_local.size:
            k = min(2, ord_local.size)
            tgt = ord_local[:k]
            # aggiungi solo se mancano
            mask = W[j, tgt] == 0.0
            tgt = tgt[mask]
            if tgt.size:
                W[j, tgt] = - 0.5 * med_II * rng.lognormal(mean=0.0, sigma=0.25, size=tgt.size)

    # no self
    np.fill_diagonal(W, 0.0)
    return W



def prune_long_range_EE(W, pos, ntypes, keep_frac=0.70):
    """
    Keep only the shortest E->E edges: remove the longest (1-keep_frac) fraction.
    keep_frac in [0,1]: e.g., 0.70 keeps the 70% shortest E->E connections.
    """
    import numpy as np
    W = W.copy()
    E = ntypes.astype(bool)
    rows, cols = np.where(W > 0.0)
    mask_EE = E[rows] & E[cols]
    if not np.any(mask_EE):
        return W  # nothing to prune

    r = rows[mask_EE]; c = cols[mask_EE]
    d = np.linalg.norm(pos[r] - pos[c], axis=1)
    cutoff = np.quantile(d, keep_frac)  # keep the short ones
    kill = d > cutoff
    W[r[kill], c[kill]] = 0.0
    return W



def prune_inhibitory_long_range(W, pos, ntypes, r_I_max=0.60, only_IE=False, renorm=True, D=None):
    """
    Prune inhibitory outgoing edges (I->*) beyond a distance cutoff and (optionally) renormalize
    the remaining inhibitory out-strength per inhibitory neuron (keeps total inhibitory magnitude).

    Parameters
    ----------
    W : (N,N) array
        Weighted adjacency (rows = presyn, cols = postsyn). E weights >0, I weights <0.
    pos : (N,2) array
        Neuron coordinates in mm.
    ntypes : (N,) bool array
        True = excitatory, False = inhibitory.
    r_I_max : float
        Distance cutoff in mm; I edges longer than this are removed.
    only_IE : bool
        If True prune only I->E; if False prune both I->E and I->I.
    renorm : bool
        If True rescale each inhibitory row to preserve its total inhibitory magnitude after pruning.
    D : (N,N) array or None
        Optional precomputed distance matrix (mm). If None it will be computed.

    Returns
    -------
    Wp : (N,N) array
        Pruned (and possibly renormalized) weight matrix.
    stats : dict
        Small summary: number of pruned edges and fractions.
    """
    Wp   = np.asarray(W, dtype=np.float64).copy()
    pos  = np.asarray(pos, dtype=np.float64)
    typ  = np.asarray(ntypes).astype(bool)  # True=E, False=I
    N    = Wp.shape[0]

    if D is None:
        D = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
        np.fill_diagonal(D, 0.0)

    I_pre   = ~typ                      # inhibitory rows
    post_ok = typ[None, :] if only_IE else np.ones((1, N), dtype=bool)

    edges      = (Wp != 0.0)
    mask_far_I = (I_pre[:, None]) & post_ok & (D > r_I_max) & edges

    cut_before = int(np.sum(mask_far_I))

    # old inhibitory magnitude per I neuron (sum of absolute negative weights)
    old_mag = -np.sum(np.where(Wp[I_pre, :] < 0.0, Wp[I_pre, :], 0.0), axis=1)

    # prune
    Wp[mask_far_I] = 0.0

    # optional renormalization
    if renorm:
        new_mag = -np.sum(np.where(Wp[I_pre, :] < 0.0, Wp[I_pre, :], 0.0), axis=1)
        scale   = np.divide(old_mag, np.maximum(new_mag, 1e-12),
                            out=np.ones_like(old_mag), where=new_mag > 0)
        Wp[I_pre, :] *= scale[:, None]

    np.fill_diagonal(Wp, 0.0)

    # stats
    edges_after = (Wp != 0.0)
    cut_after   = int(np.sum(mask_far_I))  # should be 0 now
    stats = {
        "pruned_edges": cut_before,
        "frac_pruned_over_all_I_edges": float(cut_before / max(1, np.sum(edges & (I_pre[:,None])))),
        "only_IE": bool(only_IE),
        "r_I_max_mm": float(r_I_max),
    }
    return Wp, stats
