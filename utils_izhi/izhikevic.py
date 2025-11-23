import numpy as np
from numba import njit
from timeit import default_timer as timer

#------------------------------------------------------------------------------------------------------------#
# izhikevic parameters
def set_izhikevic_parameters(regime):
    if regime=='RS':
        izhi_exc  = {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 8, 'c_ran': 0, 'd_ran': 0}    #RS
        #izhi_inh  = {'a': 0.1,'a_ran': 0,'b': 0.2,'b_ran': 0,'c': -65,'d': 2}         #FS
        izhi_inh  = {'a': 0.1,'a_ran': -0.08,'b': 0.2,'b_ran': +0.05,'c': -65,'d': 2}  #FS-LTS
        #reg_i='FS'
        reg_i='FS-LTS'
    elif regime=='RS-CH':
        izhi_exc  = {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 8, 'c_ran': +15, 'd_ran': -6} #RS-CH
        izhi_inh  = {'a': 0.1,'a_ran': -0.08,'b': 0.2,'b_ran': +0.05,'c': -65,'d': 2}  #FS-LTS
        reg_i='FS-LTS'
    elif regime=='IB':
        izhi_exc  = {'a': 0.02, 'b': 0.2, 'c': -55, 'd': 4, 'c_ran': 0, 'd_ran': 0}    #IB
        #izhi_inh  = {'a': 0.1,'a_ran': 0,'b': 0.2,'b_ran': 0,'c': -65,'d': 2}         #FS
        izhi_inh  = {'a': 0.1,'a_ran': -0.08,'b': 0.2,'b_ran': +0.05,'c': -65,'d': 2}  #FS-LTS
        #reg_i='FS'
        reg_i='FS-LTS'
    elif regime=='RS-IB':
        izhi_exc  = {'a': 0.02, 'b': 0.2, 'c': -55, 'd': 4, 'c_ran': -10, 'd_ran': +4} #IB-RS
        izhi_inh  = {'a': 0.1,'a_ran': -0.08,'b': 0.2,'b_ran': +0.05,'c': -65,'d': 2}  #FS-LTS
        reg_i='FS-LTS'
    else:
        print('\n\nProblem!!!!')

    return izhi_exc, izhi_inh


    
#------------------------------------------------------------------------------------------------------------#
# generate Poisson processes for currents

def generate_poisson_spikes(N, rate, T, dt, return_dense=False, seed=None):
    """Generate N independent Poisson spike trains.

    Parameters
    ----------
    N : int
    rate : float (Hz)
    T : float (seconds)
    dt : float (seconds per step)
    return_dense : bool -> if True returns (N, runtime) bool array

    Returns
    -------
    list of integer arrays (steps) or dense boolean array
    """
    rng = np.random.default_rng(seed)
    runtime = int(np.round(T / dt))
    if rate <= 0:
        if return_dense:
            return np.zeros((N, runtime), dtype=bool)
        return [np.empty(0, dtype=int) for _ in range(N)]

    spikes = []
    for _ in range(N):
        # Draw number of events, then place them uniformly in [0, T)
        n_events = rng.poisson(rate * T)
        if n_events == 0:
            spikes.append(np.empty(0, dtype=int))
            continue
        times = rng.random(n_events) * T
        steps = np.clip((times / dt).astype(int), 0, runtime-1)
        steps.sort()
        spikes.append(steps)

    if return_dense:
        spike_matrix = np.zeros((N, runtime), dtype=bool)
        for i, steps in enumerate(spikes):
            if steps.size:
                spike_matrix[i, steps] = True
        return spike_matrix

    return spikes


#------------------------------------------------------------------------------------------------------------#
# Prepare parameters for simulation

def prepare_numba_parameters(net, izhi_exc, izhi_inh, path_connectivity : str = None, seed=None):
    
    rng = np.random.default_rng(seed)
    
    # Neuron types
    ntypes   = net['ntypes']              # (N,) boolean array (True=exc, False=inh)
    ns       = ntypes.shape[0]            # number N of neurons

    # Simulation details
    runtime  = net['runtime']
    deltat   = net['deltat']              # [ms]
    
    # Izhikevich parameters per neuron
    nrands   = net['nrands']              # (N,) random N numbers
    if path_connectivity:
        a,b,c,d = np.loadtxt(path_connectivity+'Parameters/model_parameters.txt', unpack=True, usecols=(0,1,2,3))
    else:
        a = ntypes*izhi_exc['a'] + (~ntypes)*(izhi_inh['a']+izhi_inh['a_ran']*nrands)    # (N,)
        b = ntypes*izhi_exc['b'] + (~ntypes)*(izhi_inh['b']+izhi_inh['b_ran']*nrands)    # (N,)
        nrsquared = nrands*nrands                                                        # (N,)
        c = ntypes*(izhi_exc['c']+izhi_exc['c_ran']*nrsquared) + (~ntypes)*izhi_inh['c'] # (N,)
        d = ntypes*(izhi_exc['d']+izhi_exc['d_ran']*nrsquared) + (~ntypes)*izhi_inh['d'] # (N,)
        
     # Connectivity (receiver,sender) -> transpose rows=postsyn, cols=presyn
    S = np.array(net['weights'], dtype=np.float64).T
    S[:,ntypes==True]  *= net['g_E']
    S[:,ntypes==False] *= net['g_I']

    # Delays
    D = np.array(net['delays'], dtype=np.int32).T
    
    # Noise matrix (Poisson)
    T_sec    = runtime * deltat / 1000.0  # duration in sec
    rate_exc = float(net.get('rate_exc', 1.0))   # Hz
    rate_inh = float(net.get('rate_inh', 1.0))   # Hz
    
    # Generate dense Poisson spikes separated for E e I
    spikes_exc = generate_poisson_spikes(np.sum(ntypes),  rate_exc, T_sec, deltat/1000., return_dense=True, seed=seed)
    spikes_inh = generate_poisson_spikes(np.sum(~ntypes), rate_inh, T_sec, deltat/1000., return_dense=True, seed=seed)
    
    # Combines in final matrix
    I_noise_all = np.zeros((runtime, ns), dtype=np.float32)
    I_noise_all[:, ntypes]  = spikes_exc.T * float(net['I_intensity_exc'])
    I_noise_all[:, ~ntypes] = spikes_inh.T * float(net['I_intensity_inh'])
    #print()

    v_peak = net['v_peak']

    return ns, v_peak, a, b, c, d, S, D, ntypes, I_noise_all

#------------------------------------------------------------------------------------------------------------#
# GET OTHER PARAMETERS

def get_STD_parameters(net):
    tau_AMPA = net['tau_AMPA']
    tau_GABA = net['tau_GABA']
    tau_R  = net['tau_R']
    beta_E = net['beta_E']
    beta_I = net['beta_I']
    j_AMPA = net['j_AMPA']
    j_GABA = net['j_GABA']
    return tau_AMPA, tau_GABA, tau_R, beta_E, beta_I, j_AMPA, j_GABA


def net_runtime(net, time_sec=5, deltat=1, verbose=True):
    fs = int(1/deltat*1000)  # ms_in_1sec = 1000
    net['runtime'] = int(time_sec*fs)
    net['deltat']  = deltat
    net['fs']      = fs
    if verbose:
        print('\nruntime: ',net['runtime'],' [timesteps]','\ntime: ',time_sec,' [sec]','\nsampling frequency: ',fs,' [Hz]')
    return net  

#------------------------------------------------------------------------------------------------------------#


#------------------------------------------------------------------------------------------------------------#
# Izhikevic model with AMPA and GABA currents + STD (synaptic resources)
    
def _csr_by_presyn(S, D):
    # ---------- build CSR by presynaptic column ----------
    """
    Build CSR-like arrays organized by presynaptic neuron j.
    Returns:
      indptr: (N+1,) int32
      indices: (nnz,) int32       -> postsynaptic i for each edge
      weights: (nnz,) float64     -> S[i,j] for each edge (keep your signs)
      delays:  (nnz,) int32       -> D[i,j] in timesteps
      max_delay: int
    """
    N          = S.shape[0]
    counts     = (S != 0).sum(axis=0).astype(np.int32)
    indptr     = np.zeros(N + 1, dtype=np.int32)
    indptr[1:] = np.cumsum(counts)

    nnz        = int(indptr[-1])
    indices    = np.empty(nnz, dtype=np.int32)
    weights    = np.empty(nnz, dtype=np.float64)
    delays     = np.empty(nnz, dtype=np.int32)

    ptr   = 0
    max_d = 0
    for j in range(N):
        rows = np.nonzero(S[:, j])[0]
        k    = rows.size
        if k:
            indices[ptr:ptr+k] = rows.astype(np.int32)
            weights[ptr:ptr+k] = S[rows, j].astype(np.float64)
            Dj                 = D[rows, j].astype(np.int32)
            delays[ptr:ptr+k]  = Dj
            if Dj.size:
                md = int(Dj.max())
                if md > max_d:
                    max_d = md
            ptr += k

    return indptr, indices, weights, delays, int(max_d)

# ---------- Izhikevich + STD (Muñoz) + delays with ring buffers (Numba) + refractory ----------
@njit
def _static_munoz_izhikevich_numba_core(runtime, deltat, ns, v_peak, a, b, c, d,
                                        indptr, indices, weights, delays, max_delay,
                                        ntypes, I_noise_all,
                                        tau_AMPA, tau_GABA, tau_R, beta_E, beta_I,
                                        j_AMPA, j_GABA):
    """
    Izhikevich with AMPA/GABA currents + STD (Muñoz), event-scheduled via ring buffers.
    - Spike detection AFTER integration, then schedule to t+delay.
    - Absolute refractory (E=3 ms, I=1 ms), v clamped to c during refractory.
    - Resource R: at spike R <- beta*R; between spikes τ_R dR/dt = 1 - R.
    """
    v = -65.0 * np.ones(ns)
    u = b * v
    firings = np.zeros((ns, runtime), dtype=np.bool_)

    # Synaptic states
    I_AMPA = np.zeros(ns)
    I_GABA = np.zeros(ns)
    R      = np.ones(ns)

    # Precomputed decays
    decay_AMPA = np.exp(-deltat / tau_AMPA)
    decay_GABA = np.exp(-deltat / tau_GABA)
    decay_R    = np.exp(-deltat / tau_R)

    # Delay ring buffers
    M = max_delay + 1
    buf_AMPA = np.zeros((M, ns))
    buf_GABA = np.zeros((M, ns))
    buf_idx  = 0

    # Refractory timers (in timesteps)
    refrac_ecc = int(3.0 / deltat)  # 3 ms for excitatory
    refrac_inh = int(1.0 / deltat)  # 1 ms for inhibitory
    refrac_timer = np.zeros(ns, dtype=np.int32)

    for t in range(runtime):
        # 1) Passive decays (state at start of step)
        I_AMPA *= decay_AMPA
        I_GABA *= decay_GABA
        R = 1.0 + (R - 1.0) * decay_R
        # numerical safety
        for i in range(ns):
            if R[i] < 0.0:
                R[i] = 0.0
            elif R[i] > 1.0:
                R[i] = 1.0

        # 2) Apply arrivals due now (after decays, so new quanta are not instantly decayed)
        I_AMPA += buf_AMPA[buf_idx, :]
        I_GABA += buf_GABA[buf_idx, :]
        buf_AMPA[buf_idx, :].fill(0.0)
        buf_GABA[buf_idx, :].fill(0.0)

        # 3) Clamp v during absolute refractory (pre-integration)
        for i in range(ns):
            if refrac_timer[i] > 0:
                v[i] = c[i]

        # 4) Total current
        I = I_AMPA + I_GABA + I_noise_all[t, :]

        # 5) Membrane integration (two half-steps)
        dv = (0.04 * v * v + 5.0 * v + 140.0 - u + I) * (deltat * 0.5)
        v += dv
        for i in range(ns):
            if v[i] < -100.0:
                v[i] = -100.0
            elif v[i] > 100.0:
                v[i] = 100.0
        dv = (0.04 * v * v + 5.0 * v + 140.0 - u + I) * (deltat * 0.5)
        v += dv
        for i in range(ns):
            if v[i] < -100.0:
                v[i] = -100.0
            elif v[i] > 100.0:
                v[i] = 100.0

        # 6) Spike detection AFTER integration (respect refractory)
        fired_mask = (v >= v_peak) & (refrac_timer == 0)
        fired = np.where(fired_mask)[0]

        if fired.size > 0:
            # Mark spikes and reset
            for k in range(fired.size):
                j = fired[k]
                firings[j, t] = True
                v[j] = c[j]
                u[j] += d[j]
                # Set refractory per type
                if ntypes[j]:
                    refrac_timer[j] = refrac_ecc
                else:
                    refrac_timer[j] = refrac_inh

            # Schedule arrivals using Muñoz release: rel = R_pre; then R <- beta*R
            for k in range(fired.size):
                j = fired[k]
                beta = beta_E if ntypes[j] else beta_I
                rel = R[j]          # Muñoz: increment ∝ R_pre
                R[j] *= beta        # depression at spike

                start = indptr[j]
                end   = indptr[j+1]
                if ntypes[j]:          # excitatory -> AMPA channel
                    gain = j_AMPA[j]
                    for e in range(start, end):
                        i = indices[e]
                        w = weights[e]
                        dly = delays[e]
                        slot = (buf_idx + dly) % M
                        buf_AMPA[slot, i] += gain * rel * w
                else:                  # inhibitory -> GABA channel
                    gain = j_GABA[j]
                    for e in range(start, end):
                        i = indices[e]
                        w = weights[e]
                        dly = delays[e]
                        slot = (buf_idx + dly) % M
                        buf_GABA[slot, i] += gain * rel * w

        # 7) Recovery variable (vectorized)
        u += deltat * (a * (b * v - u))

        # 8) Decrement refractory timers
        for i in range(ns):
            if refrac_timer[i] > 0:
                refrac_timer[i] -= 1

        # 9) Advance ring buffer pointer
        buf_idx += 1
        if buf_idx == M:
            buf_idx = 0

    return firings

    
# ---------- wrapper ----------
def static_munoz_izhikevich_numba(runtime, deltat, ns, v_peak, a, b, c, d, S, D, ntypes, I_noise_all, tau_AMPA, tau_GABA, tau_R, beta_E, beta_I, j_AMPA, j_GABA):
    """
    Drop-in wrapper:
    - Builds CSR-by-presyn from dense S,D once.
    - Calls the numba core with ring buffers, correct STD+delays, and refractory.
    """
    # Ensure dtypes (Numba likes consistent types)
    S = np.asarray(S, dtype=np.float64)
    D = np.asarray(D, dtype=np.int32)
    indptr, indices, weights, delays, max_delay = _csr_by_presyn(S, D)

    # Forward to JIT-compiled core
    return _static_munoz_izhikevich_numba_core(runtime, deltat, ns, v_peak, a, b, c, d, indptr.astype(np.int32), indices.astype(np.int32),
                                               weights, delays.astype(np.int32), max_delay, ntypes, I_noise_all.astype(np.float64),
                                               tau_AMPA, tau_GABA, tau_R, beta_E, beta_I, j_AMPA.astype(np.float64), j_GABA.astype(np.float64))

#------------------------------------------------------------------------------------------------------------#
# generate perturbations in current noise matrix

#------------------------------------------------------------------------------------------------------------#
# generate perturbations in current noise matrix (parallel with joblib)
def generate_perturbations(net, I_noise_evoked, stim_chans, Ttrial_block,
                           dT_trial_sec=2, DeltaStim_ms=3, Tstim_ms=16,
                           stim_start_sec=1, amp_seed=1500,
                           nearest_neighbours=False, nearest_E=False,
                           r_mm=0.05, K=10, n_jobs=1, verobose=True):

    from joblib import Parallel, delayed

    # ---- constants / views ----
    rel_off = np.arange(0, Tstim_ms, DeltaStim_ms, dtype=np.int64)
    Emsk    = net['ntypes'].astype(bool)
    pos     = net['pos']
    fs      = int(net['fs'])
    runtime = int(I_noise_evoked.shape[0])
    amp_dtype = I_noise_evoked.dtype

    # ---- per-seed builder ----
    def _entries_for_seed(i, seed):
        """
        Ritorna tuple (t_idx, c_idx, amp) per il seed 'seed' e i suoi eventuali vicini.
        Gli indici temporali sono disgiunti tra seed diversi (offset a blocchi).
        """
        # tempi del blocco per questo seed: [stim_start_sec + i*Ttrial_block, stim_start_sec + (i+1)*Ttrial_block)
        starts  = np.arange(stim_start_sec + Ttrial_block * i,
                            stim_start_sec + Ttrial_block * (i + 1),
                            dT_trial_sec, dtype=np.int64)
        t_block = ((starts * fs)[:, None] + rel_off[None, :]).ravel()

        t_list, c_list, a_list = [], [], []

        # seed pulsato
        t_list.append(t_block)
        c_list.append(np.full(t_block.size, seed, dtype=np.int64))
        a_list.append(np.full(t_block.size, amp_seed, dtype=amp_dtype))

        if nearest_neighbours:
            # vicini entro r_mm (solo E se nearest_E=True), max K per seed
            d = np.linalg.norm(pos - pos[seed], axis=1)
            if nearest_E:
                idx = np.where((d <= r_mm) & Emsk)[0]
                idx = idx[idx != seed]
            else:
                idx = np.where((d <= r_mm) & (np.arange(d.size) != seed))[0]

            if idx.size:
                if idx.size > K:
                    idx = idx[np.argsort(d[idx])[:K]]
                # ampiezza dei vicini con decadimento gaussiano dalla distanza
                amp_neigh = (amp_seed * np.exp(-(d[idx] / (r_mm / 2.0))**2)).astype(amp_dtype)
                rep = t_block.size
                t_list.append(np.tile(t_block, idx.size))
                c_list.append(np.repeat(idx.astype(np.int64), rep))
                a_list.append(np.repeat(amp_neigh, rep))

        # concatena per questo seed
        t_idx = np.concatenate(t_list)
        c_idx = np.concatenate(c_list)
        amp   = np.concatenate(a_list)
        return t_idx, c_idx, amp

    # ---- parallel map sui seed ----
    results = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_entries_for_seed)(i, seed) for i, seed in enumerate(stim_chans))

    # ---- concatena tutti i seed ----
    if not results:
        return I_noise_evoked

    t_idx = np.concatenate([r[0] for r in results])
    c_idx = np.concatenate([r[1] for r in results])
    amp   = np.concatenate([r[2] for r in results])

    # ---- clip a runtime e scrivi una volta sola ----
    m = (t_idx >= 0) & (t_idx < runtime)
    t_idx = t_idx[m].astype(np.int64, copy=False)
    c_idx = c_idx[m].astype(np.int64, copy=False)
    amp   = amp[m].astype(amp_dtype,   copy=False)

    if nearest_neighbours:
        # somma (comportamento originale con "+=")
        np.add.at(I_noise_evoked, (t_idx, c_idx), amp)
    else:
        # assegnazione (comportamento originale con "=")
        I_noise_evoked[t_idx, c_idx] = amp

    return I_noise_evoked
#------------------------------------------------------------------------------------------------------------#


'''
def generate_perturbations(net, I_noise_evoked, stim_chans, Ttrial_block, dT_trial_sec=2, DeltaStim_ms=3, Tstim_ms=16, stim_start_sec=1, amp_seed=1500, nearest_neighbours=False, nearest_E=False, r_mm=0.05, K=10):

    # perturbation times per trial
    rel_off       = np.arange(0, Tstim_ms, DeltaStim_ms, dtype=np.int64)
    
    Emsk     = net['ntypes'].astype(bool)
    pos      = net['pos']
    
    fs      = net['fs']
    runtime = I_noise_evoked.shape[0]
    
    if nearest_neighbours:
        # -- if nearest_neighbours is True, stimulate also max K nearest neigbours in a radius r_mm --
        if nearest_E:
            # if nearest_E is True, stimulate only excitatory nearest neighbours
            t_all, c_all, a_all = [], [], []
            for i, seed in enumerate(stim_chans):
                # blocco tempi per questo seed: 1 + dT_trial_sec*k secondi, poi aggiungi offset in campioni
                starts  = np.arange(stim_start_sec + Ttrial_block*i, stim_start_sec + Ttrial_block*(i+1), dT_trial_sec, dtype=np.int64)
                t_block = ((starts * fs)[:, None] + rel_off[None, :]).ravel()
            
                # seed
                t_all.append(t_block)
                c_all.append(np.full(t_block.size, seed, dtype=np.int64))
                a_all.append(np.full(t_block.size, amp_seed, dtype=I_noise_evoked.dtype))
            
                # neigbours: only E inside r_mm, max K, sorted by distance
                d    = np.linalg.norm(pos - pos[seed], axis=1)
                idx  = np.where((d <= r_mm) & Emsk)[0]
                idx  = idx[idx != seed]                       # exclude seed if E
                if idx.size:
                    idx = idx[np.argsort(d[idx])[:K]]
                    amp_neigh = (amp_seed * np.exp(-(d[idx] / (r_mm/2))**2)).astype(I_noise_evoked.dtype)
                    rep = t_block.size
                    t_all.append(np.tile(t_block, idx.size))
                    c_all.append(np.repeat(idx, rep))
                    a_all.append(np.repeat(amp_neigh, rep))
            
            # scrittura in un colpo (somma, non sovrascrive)
            t_idx = np.concatenate(t_all); c_idx = np.concatenate(c_all); amp = np.concatenate(a_all)
            m = t_idx < runtime
            I_noise_evoked[t_idx[m], c_idx[m]] += amp[m]
    
        else:
            t_all, c_all, a_all = [], [], []
            for i, seed in enumerate(stim_chans):
                # times per seed: 1 + dT_trial_sec*k sec in block i
                starts  = np.arange(stim_start_sec + Ttrial_block*i, stim_start_sec + Ttrial_block*(i+1), dT_trial_sec, dtype=np.int64)
                t_block = ((starts * fs)[:, None] + rel_off[None, :]).ravel()
            
                # seed 
                t_all.append(t_block)
                c_all.append(np.full(t_block.size, seed, dtype=np.int64))
                a_all.append(np.full(t_block.size, amp_seed, dtype=I_noise_evoked.dtype))
            
                # neighbours: ALL (E e I) inside r_mm, excluded seed, until K nearest neig.
                d    = np.linalg.norm(pos - pos[seed], axis=1)
                idx  = np.where((d <= r_mm) & (np.arange(d.size) != seed))[0]
                if idx.size:
                    if idx.size > K:
                        idx = idx[np.argsort(d[idx])[:K]]     # keep K most close n.
                    amp_neigh = (amp_seed * np.exp(-(d[idx] / (r_mm/2))**2)).astype(I_noise_evoked.dtype)
                    rep = t_block.size
                    t_all.append(np.tile(t_block, idx.size))
                    c_all.append(np.repeat(idx, rep))
                    a_all.append(np.repeat(amp_neigh, rep))
            
            # scrivi in un colpo (somma al rumore esistente)
            t_idx = np.concatenate(t_all); c_idx = np.concatenate(c_all); amp = np.concatenate(a_all)
            m = t_idx < runtime
            I_noise_evoked[t_idx[m], c_idx[m]] += amp[m]
    else:

        starts_list = [np.arange(stim_start_sec + Ttrial_block*i, stim_start_sec + Ttrial_block*(i+1), dT_trial_sec, dtype=np.int64) for i in range(len(stim_chans))]
        t_idx       = np.concatenate([((s * fs)[:, None] + rel_off[None, :]).ravel() for s in starts_list])
        c_idx       = np.concatenate([np.full(s.size * rel_off.size, ch, dtype=np.int64) for ch, s in zip(stim_chans, starts_list)])
        m = t_idx < runtime
        I_noise_evoked[t_idx[m], c_idx[m]] = amp_seed
        
    return I_noise_evoked
'''
#------------------------------------------------------------------------------------------------------------#














#------------------------------------------------------------------------------------------------------------#
# Basic Izhikevic Model with refractory time

@njit
def static_ref_izhikevich_numba(runtime, deltat, ns, v_peak, a, b, c, d, S, D, ntypes, I_noise_all):
    v = -65.0 * np.ones(ns)
    u = b * v
    firings = np.zeros((ns, runtime), dtype=np.bool_)

    max_delay = np.max(D)
    D = D.astype(np.int32)

    refrac_ecc = int(3.0 / deltat)  # 3 ms refractory period for excitatory neurons
    refrac_inh = int(1.0 / deltat)  # 1 ms refractory period for inhibitory neurons
    refrac_timer = np.zeros(ns, dtype=np.int32)

    for t in range(runtime):
        # Find neurons that can fire: v >= v_peak and refractory timer is zero
        can_fire = np.zeros(ns, dtype=np.bool_)
        for i in range(ns):
            if v[i] >= v_peak and refrac_timer[i] == 0:
                can_fire[i] = True
        fired = np.where(can_fire)[0]

        for i in fired:
            # Register firing
            firings[i, t] = True
            # Reset v and update u for fired neurons
            v[i] = c[i]
            u[i] += d[i]
            # Assign differentiated refractory period
            if ntypes[i]:
                refrac_timer[i] = refrac_ecc
            else:
                refrac_timer[i] = refrac_inh

        # Decrement refractory timers
        for i in range(ns):
            if refrac_timer[i] > 0:
                refrac_timer[i] -= 1

        # Build input current with noise + synaptic inputs with delay
        I = I_noise_all[t, :].astype(np.float64)
        '''
        for i in range(ns):
            for j in range(ns):
                delay_idx = t - D[i, j]
                if delay_idx >= 0 and firings[j, delay_idx]:
                    I[i] += S[i, j]
        '''
        for j in fired:
            for i in range(ns):
                delay = D[i, j]
                if t - delay >= 0 and firings[j, t - delay]:
                    I[i] += S[i, j]
             
        # Update v
        dv = deltat / 2 * (0.04 * v * v + 5 * v + 140 - u + I)
        v += dv

        # Clip v between -100 and 100
        for i in range(ns):
            if v[i] < -100.0:
                v[i] = -100.0
            elif v[i] > 100.0:
                v[i] = 100.0
        v += dv
        for i in range(ns):
            if v[i] < -100.0:
                v[i] = -100.0
            elif v[i] > 100.0:
                v[i] = 100.0

        # Update u
        for i in range(ns):
            u[i] += deltat * (a[i] * (b[i] * v[i] - u[i]))

    return firings


#------------------------------------------------------------------------------------------------------------#
# Basic Izhijevic Model

@njit
def static_izhikevich_numba(runtime, deltat, ns, v_peak, a, b, c, d, S, D, ntypes, I_noise_all):

    v = -65.0 * np.ones(ns)
    u = b * v
    firings = np.zeros((ns, runtime), dtype=np.bool_)

    max_delay = np.max(D)
    D = D.astype(np.int32)

    for t in range(max_delay):
        fired = np.where(v >= v_peak)[0]
        firings[fired, t] = True

        v[fired] = c[fired]
        u[fired] += d[fired]

        I = I_noise_all[t, :].copy()
        if fired.size > 0:
            for i in range(ns):
                for j in fired:
                    I[i] += S[i, j]

        # update v with clipping inline
        dv = deltat / 2 * (0.04 * v * v + 5 * v + 140 - u + I)
        v += dv
        for i in range(ns):
            if v[i] < -100.0:
                v[i] = -100.0
            elif v[i] > 100.0:
                v[i] = 100.0

        v += dv
        for i in range(ns):
            if v[i] < -100.0:
                v[i] = -100.0
            elif v[i] > 100.0:
                v[i] = 100.0

        u += deltat * (a * (b * v - u))

    for t in range(max_delay, runtime):
        fired = np.where(v >= v_peak)[0]
        firings[fired, t] = True

        v[fired] = c[fired]
        u[fired] += d[fired]

        I = I_noise_all[t, :].copy()
        for i in range(ns):
            for j in range(ns):
                delay_idx = t - D[i, j]
                if delay_idx >= 0 and firings[j, delay_idx]:
                    I[i] += S[i, j]

        dv = deltat / 2 * (0.04 * v * v + 5 * v + 140 - u + I)
        v += dv
        for i in range(ns):
            if v[i] < -100.0:
                v[i] = -100.0
            elif v[i] > 100.0:
                v[i] = 100.0

        v += dv
        for i in range(ns):
            if v[i] < -100.0:
                v[i] = -100.0
            elif v[i] > 100.0:
                v[i] = 100.0

        u += deltat * (a * (b * v - u))

    return firings