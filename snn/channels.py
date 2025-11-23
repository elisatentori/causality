import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgba

import numpy as np

from utils import colormaps as maps
from utils import plot as pl

#=================================================================================================#
from matplotlib import font_manager, rcParams
font_file = "/home/tentori/.local/avenir_ff/AvenirLTStd-Roman.otf"
font_file_b = "/home/tentori/.local/avenir_ff/AvenirLTStd-Black.otf"
font_file_c = "/home/tentori/.local/avenir_ff/AvenirLTStd-Book.otf"
font_manager.fontManager.addfont(font_file)
font_manager.fontManager.addfont(font_file_b)
font_manager.fontManager.addfont(font_file_c)

# Imposta il font predefinito su Avenir
rcParams['font.family'] = "Avenir LT Std"

DIM = 22

plt.rcParams.update({
    'font.size': DIM,
    'axes.labelsize': DIM,
    'axes.titlesize': DIM,
    'xtick.labelsize': DIM,
    'ytick.labelsize': DIM
})
#=================================================================================================#

import os
def Set_Dir_Plots(path):
    if not os.path.exists(path):
        os.mkdir(path)

coldhot_cmap   = maps.create_cmaphot()
coldhot_cmap_r = maps.create_cmaphot_r()

colors = ['#2F7FC3','#E62A08','#464646','#FFD700','#32CD32','#8A2BE2']


'''
def stack_trials(channel, def_units, def_spikes, T_spont = 1800, DeltaT_sec = 3.8, Ntrials = 20, fs = 1000, 
                 n_zero_artifacts = 8, n_artifacts = 24*5, ipi_sec = 0.001, path_out='./', one = False):

    import time
    
    DeltaT = int(DeltaT_sec * fs)
    
    # for artefacts
    ipi    = int(ipi_sec * fs)
    offset = int(1 * fs)

    n_units = len(def_units)  # n. stimulation channels
    n_channels = len(channel) # n. recording  channels

    #-------------------------- global start times ----------------------------#
    total_blocks = n_units * Ntrials
    T_starts     = int(T_spont * fs) + np.arange(total_blocks) * DeltaT
    #--------------------------------------------------------------------------#
    
    #---------------------- stimulation artifacts (>1s) -----------------------#
    artifact_times     = np.arange(offset, offset + n_artifacts * ipi, ipi)
    artifact_mat       = T_starts[:, None] + artifact_times[None, :]
    artifact_all_times = artifact_mat.flatten()
    #--------------------------------------------------------------------------#
    
    if one:
        ###only one artifact at the beginning of each trial
        #---------------------- stimulation artifacts @ t=0 -----------------------#
        zero_artifact_times    = np.repeat(T_starts, len(channel))
        zero_artifact_channels = np.tile(channel, total_blocks)
        #--------------------------------------------------------------------------#
    else:
        #---------------------- stimulation artifacts @ t=0 -----------------------#
        zero_artifact_times_base = np.arange(0, n_zero_artifacts * ipi, ipi)
        zero_artifact_mat = T_starts[:, None] + zero_artifact_times_base[None, :]
        zero_artifact_all_times = zero_artifact_mat.flatten()
    
        zero_artifact_times = np.tile(zero_artifact_all_times, len(channel))
        zero_artifact_channels = np.repeat(channel, total_blocks * n_zero_artifacts)
        #--------------------------------------------------------------------------#
    
    spikeTimes_list    = []
    spikeChannels_list = []
    
    time_start = time.time()
    
    block_idx = 0
    for trial in range(Ntrials):
        for unitID in def_units:
            spike_time = []
            spike_chan = []
    
            for ch in range(1024):
                t_ = def_spikes[unitID][trial][ch]
                t  = t_[(t_ > 0) & (t_ <= DeltaT)]
                if len(t) == 0:
                    continue
                c = np.full(len(t), ch)
    
                spike_time.append(t)
                spike_chan.append(c)
    
            if spike_time:
                spike_time = np.concatenate(spike_time)
                spike_chan = np.concatenate(spike_chan)
    
                spikeTimes_list.append(spike_time + T_starts[block_idx])
                spikeChannels_list.append(spike_chan)
    
            block_idx += 1
    
    #-------------------- concatenate all spike data -------------------------#
    _spikeTimes    = np.concatenate(spikeTimes_list)
    _spikeChannels = np.concatenate(spikeChannels_list)
    
    #--------------------- filter by good channels ---------------------------#
    mask = np.isin(_spikeChannels, channel)
    __spikeTimes    = _spikeTimes[mask]
    __spikeChannels = _spikeChannels[mask]
    
    #--------------------- remove existing artifacts -------------------------#
    mask_clean          = ~np.isin(__spikeTimes, artifact_all_times)
    spikeTimes_clean    = __spikeTimes[mask_clean]
    spikeChannels_clean = __spikeChannels[mask_clean]
    
    #---------------------reinsert systematic artifacts ---------------------#
    n_artifacts_total = len(artifact_all_times)
    
    artifactTimes_repeated    = np.tile(artifact_all_times, n_channels)
    artifactChannels_repeated = np.repeat(channel, n_artifacts_total)
    
    if not one:
        #--------------------- systematic zer artifacts ---------------------#
        zeroArtifactTimes_repeated    = np.tile(zero_artifact_all_times, n_channels)
        zeroArtifactChannels_repeated = np.repeat(channel, total_blocks * n_zero_artifacts)
    
    #--------------------- combine all spike data ----------------------------#
    if one:
        spikeTimes = np.concatenate([spikeTimes_clean,artifactTimes_repeated,zero_artifact_times])
        spikeChannels = np.concatenate([spikeChannels_clean,artifactChannels_repeated,zero_artifact_channels])
    else:
        spikeTimes = np.concatenate([spikeTimes_clean,artifactTimes_repeated,zeroArtifactTimes_repeated])
        spikeChannels = np.concatenate([spikeChannels_clean,artifactChannels_repeated,zeroArtifactChannels_repeated])
    
    
    #--------------------- final sort ----------------------------------------#
    sort_idx      = np.argsort(spikeTimes)
    spikeTimes    = spikeTimes[sort_idx]
    spikeChannels = spikeChannels[sort_idx]
    time_stop = time.time()
    print(f"Elapsed: {time_stop - time_start:.3f} s")

    #--------------------- save output ----------------------------------------#
    time_start = time.time()
    data = np.column_stack((spikeChannels, spikeTimes.astype(int)))
    #np.savetxt("spikeTimes_stimulation.txt", data, fmt='%d', delimiter='\t')
    np.save(path_out+"spikeTimes_stimulation.npy", data)
    time_stop = time.time()
    print(f"Elapsed: {time_stop - time_start:.3f} s")

    return spikeTimes, spikeChannels
'''


#----------------------------------------------------------------------------------------------------------------------------#
def insert_artifacts(channel, def_units, def_spikes, Tstim_start_s=1, Tstim_duration_s=0.12, Tonset_duration=0.12,
                     erase_min=1.00-0.001, erase_max=1.12+0.001, erase_first=0.120, ipi_sec=0.001, Ntrials=200, fs=1000):

    erase_min_samp   = int(round(erase_min   * fs))
    erase_max_samp   = int(round(erase_max   * fs))
    erase_first_samp = int(round(erase_first * fs))
    
    Tstim_start_s_samp    = int(round(Tstim_start_s    * fs))
    Tstim_duration_s_samp = int(round(Tstim_duration_s * fs))
    Tonset_duration_samp  = int(round(Tonset_duration  * fs))
    ipi_samp              = int(round(ipi_sec          * fs))
    
    # ---- erase spikes during stimulation and the trial onset ----
    modified_spikes = {}
    for n_stim in def_units:
        modified_spikes[n_stim] = {}
        for n_trial in range(Ntrials):
            modified_spikes[n_stim][n_trial] = {}
            for n_channel in range(1000):
                s                 = def_spikes[n_stim][n_trial][n_channel]
                cond_outside_stim = (s < erase_min_samp) | (s >= erase_max_samp)
                cond_after_first  = s > erase_first_samp
                modified_spikes[n_stim][n_trial][n_channel] = s[cond_outside_stim & cond_after_first]
    
    # ---- insert artifacts ----
    artifact_spikes = {}
    for n_stim in def_units:
        artifact_spikes[n_stim] = {}
        for n_trial in range(Ntrials):
            artifact_spikes[n_stim][n_trial] = {}
            for n_channel in range(1000):
                s   = modified_spikes[n_stim][n_trial][n_channel]
                s_a = np.arange(0,                  Tonset_duration_samp,                     ipi_samp)
                s_b = np.arange(Tstim_start_s_samp, Tstim_start_s_samp+Tstim_duration_s_samp, ipi_samp)
                artifact_spikes[n_stim][n_trial][n_channel] = np.concatenate([s_a,s,s_b])

    return artifact_spikes
#----------------------------------------------------------------------------------------------------------------------------#


#----------------------------------------------------------------------------------------------------------------------------#
def stack_trials(channel, def_units, spikes, T_spont=1800., DeltaT_sec=3., Ntrials=200, fs=1000, path_out=None, one=False):

    DeltaT  = int(DeltaT_sec * fs)
    n_units = len(def_units)  # n. stimulation channels
    
    # Global start times
    total_blocks = n_units * Ntrials
    T_starts     = int(T_spont * fs) + np.arange(total_blocks) * DeltaT
    
    spikeTimes_list = []
    spikeChannels_list = []

    block_idx = 0
    for trial in range(Ntrials):
        for unitID in def_units:
            block_spike_times = []
            block_spike_chans = []
            
            for ch in channel:
                t_ = spikes[unitID][trial][ch]
                t_ = t_[t_ < DeltaT]
                if t_.size == 0:
                    continue
                block_spike_times.append(t_)
                block_spike_chans.append(np.repeat(ch, t_.size))
    
            if block_spike_times:
                block_spike_times = np.concatenate(block_spike_times) + T_starts[block_idx]
                block_spike_chans = np.concatenate(block_spike_chans)
    
                spikeTimes_list.append(block_spike_times)
                spikeChannels_list.append(block_spike_chans)
    
            block_idx += 1
            
    # Concatenate all spike data
    if spikeTimes_list:
        spikeTimes    = np.concatenate(spikeTimes_list)
        spikeChannels = np.concatenate(spikeChannels_list)
    else:
        spikeTimes    = np.array([])
        spikeChannels = np.array([])
    
    # Final sort
    sort_idx      = np.argsort(spikeTimes)
    spikeTimes    = spikeTimes[sort_idx]
    spikeChannels = spikeChannels[sort_idx]
    
    if path_out is not None:
        data = np.column_stack((spikeChannels, spikeTimes.astype(int)))
        np.save(path_out + "spikeTimes_stimulation.npy", data)
    
    return spikeTimes, spikeChannels
#----------------------------------------------------------------------------------------------------------------------------#



#----------------------------------------------------------------------------------------------------------------------------#
def stack_trials_compact(channel, def_units, def_spikes, T_spont=1800., DeltaT_sec=3.8, Ntrials=20, fs=1000,
                         cut_pre_ms=5.0, stim_offset_sec=1.0, stim_dur_ms = 120, path_out='./'):
    """
    Build evoked spikes by concatenating trials, but REMOVE (time-compaction):
      - the first 'cut_pre_ms' milliseconds of each trial
      - the whole stimulation window: [stim_offset_sec, stim_offset_sec + n_artifacts*ipi_sec)
    No artificial artifacts are added. Time is compacted, not silenced.
    Returns:
      spikeTimes (int samples, absolute timeline starting at T_spont*fs),
      spikeChannels (int),
      kept_per_trial_sec (float),
      kept_total_sec (float)
    """
    import time

    DeltaT = int(round(DeltaT_sec * fs))
    cut_pre = int(round((cut_pre_ms / 1000.0) * fs))
    stim_off = int(round(stim_offset_sec * fs))
    stim_dur = int(round((stim_dur_ms / 1000.0) * fs))
    stim_end = stim_off + stim_dur

    if cut_pre < 0 or stim_off < 0 or stim_end > DeltaT:
        raise ValueError("Check cut_pre_ms, stim_offset_sec, ipi_sec, n_artifacts vs DeltaT_sec.")

    # Effective kept length per trial (in samples) after removing the two intervals
    kept_len = DeltaT - cut_pre - stim_dur
    kept_per_trial_sec = kept_len / float(fs)

    n_units = len(def_units)
    n_channels = len(channel)
    total_blocks = n_units * Ntrials

    # Lists for compacted spikes
    spikeTimes_list = []
    spikeChannels_list = []

    t0_all = int(T_spont * fs)  # evoked starts after spontaneous
    trial_offsets = np.arange(total_blocks, dtype=np.int64) * kept_len  # compacted offsets per block

    time_start = time.time()
    block_idx = 0
    for trial in range(Ntrials):
        for unitID in def_units:
            st_list = []
            sc_list = []

            # Collect spikes per (recording) channel from original def_spikes
            # def_spikes[unitID][trial][ch] gives local times (samples) in [0, DeltaT]
            for ch in range(1000):
                t_ = def_spikes[unitID][trial][ch]
                if t_.size == 0:
                    continue
                # Keep only inside trial window
                t = t_[(t_ >= 0) & (t_ < DeltaT)]
                if t.size == 0:
                    continue

                # Remove intervals: [0, cut_pre) and [stim_off, stim_end)
                keep_mask = (t >= cut_pre) & (t < stim_off) | (t >= stim_end)
                t_keep = t[keep_mask]
                if t_keep.size == 0:
                    continue

                # Compact time within trial:
                # - times in [cut_pre, stim_off) shift by -cut_pre
                # - times in [stim_end, DeltaT) shift by -(cut_pre + stim_dur)
                shift = np.zeros_like(t_keep)
                mask_1 = (t_keep >= cut_pre) & (t_keep < stim_off)
                shift[mask_1] = cut_pre
                mask_2 = (t_keep >= stim_end)
                shift[mask_2] = (cut_pre + stim_dur)
                t_compact = t_keep - shift  # now in [0, kept_len)

                c = np.full(t_compact.shape[0], ch, dtype=int)
                st_list.append(t_compact)
                sc_list.append(c)

            if st_list:
                st_trial = np.concatenate(st_list)
                sc_trial = np.concatenate(sc_list)

                # Absolute compacted time: start at T_spont*fs + block_idx * kept_len
                st_abs = st_trial + (t0_all + trial_offsets[block_idx])
                spikeTimes_list.append(st_abs.astype(np.int64))
                spikeChannels_list.append(sc_trial.astype(int))

            block_idx += 1

    if len(spikeTimes_list) == 0:
        # No spikes survived
        spikeTimes    = np.array([], dtype=np.int64)
        spikeChannels = np.array([], dtype=int)
    else:
        spikeTimes = np.concatenate(spikeTimes_list)
        spikeChannels = np.concatenate(spikeChannels_list)

        # Keep only "good" channels
        mask = np.isin(spikeChannels, channel)
        spikeTimes = spikeTimes[mask]
        spikeChannels = spikeChannels[mask]

        # Sort by time
        idx = np.argsort(spikeTimes)
        spikeTimes = spikeTimes[idx]
        spikeChannels = spikeChannels[idx]

    kept_total_sec = kept_per_trial_sec * total_blocks

    # Save (optional)
    data = np.column_stack((spikeChannels.astype(int), spikeTimes.astype(int)))
    np.save(path_out + "spikeTimes_stimulation_compact.npy", data)

    print(f"[stack_trials_compact] Kept per trial: {kept_per_trial_sec:.3f} s | total: {kept_total_sec/60:.2f} min")
    
    return spikeTimes, spikeChannels, kept_per_trial_sec, kept_total_sec
#----------------------------------------------------------------------------------------------------------------------------#


# ------------------------------------------------------------------------------------ #
# --------------------         c h a n n e l    m a p s        ----------------------- #
# ------------------------------------------------------------------------------------ #

def alternating_colormap(n_clusters, color1='black', color2='crimson'):
    """Crea una ListedColormap alternata con due colori contrastanti."""
    colors = [color1 if i % 2 == 0 else color2 for i in range(n_clusters)]
    return mcolors.ListedColormap(colors)

# --------------------------------------------------------------------------------------------#

def neuronID_to_cluster(neuron_ids, channel, cluster):
    max_chan = np.max(channel) + 1
    lookup = np.full(max_chan, -1, dtype=int)
    lookup[channel] = cluster
    return lookup[neuron_ids]

# --------------------------------------------------------------------------------------------#

def plot_map(pos, car_array, cbar_label='cluster ID', title='recording channels map', cmap='viridis', outf : str = None, show_plot = True):

    # Colormap
    n_clusters = len(np.unique(car_array))
    if cmap is None:
        cmap = alternating_colormap(n_clusters)
    else:
        cmap = cmap

    # channels map
    plt.subplots(figsize=(13,5.5))
    plt.scatter(pos[:,0],pos[:,1],c=car_array,s=1,marker='s',cmap=cmap)
    xt=plt.xlabel('x (mm)'); plt.ylabel('y (mm)')

    if title:
        plt.title(title)
    plt.colorbar(label=cbar_label)
    
    if outf:
        plt.savefig(outf, bbox_inches='tight')
    
    if show_plot == False:
        plt.close()
    else:
        plt.show()

# ------------------------------------------------------------------------------------ #
# --------------------         r a s t e r    p l o t s        ----------------------- #
# ------------------------------------------------------------------------------------ #

def rasterplot(st, channel, cluster, title=None, dotsize=0.1, tmin=0, tmax=120, cmap='viridis',figsize=(15,5), outf : str = None, show_plot = True):
    spike_times = st[0]
    neuron_ids  = st[1].astype(int)
    cluster_ids = neuronID_to_cluster(neuron_ids, channel, cluster)
    condition   = np.logical_and(spike_times>=tmin, spike_times<=tmax)

    # Colormap
    n_clusters = len(np.unique(cluster))
    if cmap is None:
        cmap = alternating_colormap(n_clusters)
    else:
        cmap = cmap
        
    plt.subplots(figsize=figsize)
    plt.scatter(spike_times[condition], neuron_ids[condition], s=dotsize, c=cluster_ids[condition], cmap=cmap)
    plt.xlabel("time (s)")
    plt.ylabel("channel ID")
    #plt.xlim(tmin,tmax)
    if title:
        plt.title(title)
    plt.colorbar(label="cluster ID")

    if outf:
        plt.savefig(outf, bbox_inches='tight')
    
    if show_plot == False:
        plt.close()
    else:
        plt.show()

# ------------------------------------------------------------------------------------ #
# firing rate computed only channels present in rasterplot

def compute_rate(spikes, t_stop=None):
    spike_times = spikes[0]
    neuron_ids = spikes[1].astype(int)
    nNeu = np.max(neuron_ids) + 1
    spike_counts = np.bincount(neuron_ids, minlength=nNeu)
    if t_stop is None:
        t_stop = spike_times.max()
    firing_rates = spike_counts / t_stop
    return firing_rates

# ------------------------------------------------------------------------------------ #
# firing rate computed for all channels in channel array

def _compute_firing_rate(spikes, channel, t_stop=None):
    spike_times = spikes[0]
    neuron_ids  = spikes[1].astype(int)

    if t_stop is None:
        t_stop = spike_times.max()

    channel_id_to_index = {ch: i for i, ch in enumerate(channel)}

    spike_counts = np.zeros(len(channel), dtype=int)

    for nid in neuron_ids:
        if nid in channel_id_to_index:
            idx = channel_id_to_index[nid]
            spike_counts[idx] += 1

    firing_rates = spike_counts / t_stop
    return firing_rates

# ------------------------------------------------------------------------------------ #

def plot_mat_aspect(mat, cmap='viridis', title=None, xlabel='time (s)', ylabel='channel ID', cbarlabel='spike couunt', 
                    inevert_y : bool = True, figsize=(15,5), outf : str = None, show_plot = True):
    mat_max = np.max(mat.flatten())
    #vmin=0.6; vmax=0.1
    vmin,vmax = np.percentile(mat.flatten()/mat_max,[2.5,97.5])
    fig,ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat/mat_max, aspect='auto', vmin=vmin,vmax=vmax, cmap=cmap)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        plt.title(title)
    cbar = fig.colorbar(im, ax=ax, label=cbarlabel)
    if inevert_y:
        ax.invert_yaxis()
    
    if outf:
        plt.savefig(outf, bbox_inches='tight')
    
    if show_plot == False:
        plt.close()
    else:
        plt.show()
        
# ------------------------------------------------------------------------------------ #

def plot_mat(mat, title='', cmap='viridis', outf : str = None, show_plot = True):

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    fig, ax = plt.subplots(figsize=(20, 5))
    im = ax.imshow(mat, aspect='auto', cmap=cmap)
    plt.title(title)
    ax.invert_yaxis()
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    plt.colorbar(im, cax=cax)
    
    if outf:
        plt.savefig(outf, bbox_inches='tight')
    
    if show_plot == False:
        plt.close()
    else:
        plt.show()
        
# ------------------------------------------------------------------------------------ #
# --------------------      N  P A R A M S   H M M   B C I     ----------------------- #
# ------------------------------------------------------------------------------------ #

def num_params_poisson_hmm(K, D):
    """
    - K: n.- states
    - D: n. neurons
    """
    init_probs  = K - 1
    transitions = K * (K - 1)
    rates       = K * D
    return init_probs + transitions + rates

# ------------------------------------------------------------------------------------ #

def num_params_gaussian_hmm(K, D):
    init_probs  = K - 1
    transitions = K * (K - 1)
    means       = K * D
    covs        = K * (D * (D + 1)) // 2
    return init_probs + transitions + means + covs

# ------------------------------------------------------------------------------------ #

def num_params_categorical_hmm(K, D, M):
    """
    - K states
    - D categorical vars
    - M possible categories per each var
    Each state has a categorial distribution on M values per each dim.
    """
    init_probs  = K - 1
    transitions = K * (K - 1)
    emissions   = K * D * (M - 1)  # ogni distribuzione categoriale ha M-1 parametri (somma a 1)
    return init_probs + transitions + emissions

# ------------------------------------------------------------------------------------ #

def num_params_hmm(K, D, emission='poisson', M=None):
    if emission   == 'poisson':
        return num_params_poisson_hmm(K, D)
    elif emission == 'gaussian':
        return num_params_gaussian_hmm(K, D)
    elif emission == 'categorical':
        if M is None:
            raise ValueError("Specify number of categories M for categorical emissions.")
        return num_params_categorical_hmm(K, D, M)
    else:
        raise ValueError(f"Unknown emission type: {emission}")

        
# ------------------------------------------------------------------------------------ #
# --------------------               H M M   B C I             ----------------------- #
# ------------------------------------------------------------------------------------ #

def fit_hmm_and_compute_criteria(spike_counts, k, run, T, N, N_iters=100, TOL=1e-4):

    from ssm import HMM
    
    hmm = HMM(k, N, observations="poisson", transitions="standard")
    try:
        hmm.initialize(spike_counts, init_method="kmeans")
        hmm.fit(spike_counts, method="em", num_iters=N_iters, tolerance=TOL, initialize=False)
        loglik = hmm.log_likelihood(spike_counts)

        num_params = num_params_poisson_hmm(k, N)
        bic = num_params * np.log(T) - 2 * loglik
        aic = 2 * num_params - 2 * loglik

        return (k, run, bic, aic, loglik)
    except Exception as e:
        print(f"[k={k}, run={run}] Initialization failed: {e}")
        return (k, run, np.nan, np.nan, np.nan)

# ------------------------------------------------------------------------------------ #

def compute_bic_poisson_parallel(spike_counts, min_states=2, max_states=10, nRunEM=5, N_iters=200, TOL=1e-4, n_jobs=-1):
    
    from joblib import Parallel, delayed
    
    T, N = spike_counts.shape
    BIC = np.full((max_states-min_states, nRunEM), np.nan)
    AIC = np.full((max_states-min_states, nRunEM), np.nan)
    LLs = np.full((max_states-min_states, nRunEM), np.nan)

    # Tutte le combinazioni di (k, run)
    tasks = [(spike_counts, k, r, T, N, N_iters, TOL) 
             for k in range(min_states, max_states + 1) 
             for r in range(nRunEM)]

    results = Parallel(n_jobs=n_jobs)(delayed(fit_hmm_and_compute_criteria)(*args) for args in tasks)

    for k, r, bic, aic, ll in results:
        BIC[k - min_states, r] = bic
        AIC[k - min_states, r] = aic
        LLs[k - min_states, r] = ll

    return BIC, AIC, LLs

# ------------------------------------------------------------------------------------ #

def plot_model_selection(BIC, AIC, outf : str = None, show_plot = True):
    mean_bic = BIC.mean(axis=1)
    std_bic  = BIC.std(axis=1)
    mean_aic = AIC.mean(axis=1)
    std_aic  = AIC.std(axis=1)

    x = np.arange(1, BIC.shape[0] + 1)

    fig,ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, mean_bic, lw=4, color = 'tab:red', label="BIC")
    ax.fill_between(x, mean_bic - std_bic, mean_bic + std_bic, color = 'tab:red', alpha=0.2)
    ax.plot(x, mean_aic, '--', lw=4, color = 'grey', label="AIC")
    ax.fill_between(x, mean_aic - std_aic, mean_aic + std_aic,  color = 'grey', alpha=0.2)
    ax.set_xlabel("Number of states")
    ax.set_ylabel("Criterion value")
    pl.set_format(ax=ax,pwr_x_max=3,DIM=DIM)
    ax.legend()
    ax.set_title("Model selection \n HMM with Poisson emissions ")

    if outf:
        plt.savefig(outf, bbox_inches='tight')
    
    if show_plot == False:
        plt.close()
    else:
        plt.show()

# ------------------------------------------------------------------------------------ #

def select_best_k(BIC):
    mean_bic = BIC.mean(axis=1)
    best_k   = np.argmin(mean_bic) + 1
    print(f"Best number of states (by BIC): {best_k}")
    return best_k


# ------------------------------------------------------------------------------------ #

def plot_raster_with_state(time_axis, z, spikes, t_min, t_max, num_units=None, cmap='tab10', alpha=0.3, raster=True,
                           figsize=(20, 4), ncols=3, plot_posterior=False, posterior=None, dt=None, title=None, 
                           outf : str = None, show_plot = True):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    import numpy as np

    st, ni = spikes
    condition_spikes = np.logical_and(st >= t_min, st < t_max)
    condition_time   = np.logical_and(time_axis >= t_min, time_axis < t_max)

    time_axis_sel = time_axis[condition_time]
    z_sel         = z[condition_time]

    if dt is None:
        if len(time_axis_sel) > 1:
            dt = np.min(np.diff(time_axis_sel))
        else:
            raise ValueError("time_axis_sel must contain at least two time points to compute dt or pass dt manually.")

    if num_units is None:
        N = int(np.max(ni)) + 1
    else:
        N = num_units

    unique_states = np.unique(z_sel)
    num_states = len(unique_states)
    state_to_index = {state: i for i, state in enumerate(unique_states)}

    # Handle colormap
    if isinstance(cmap, str):
        base_cmap = plt.get_cmap(cmap)
        cmap_ = ListedColormap(base_cmap(np.linspace(0, 1, num_states)))
    else:
        cmap_ = cmap  # already a ListedColormap or discrete cmap

    # Start plotting
    fig, ax = plt.subplots(figsize=figsize)

    # State-colored rectangles
    for i in range(len(z_sel)):
        center_x = time_axis_sel[i]
        left = center_x - dt / 2
        color_idx = state_to_index[z_sel[i]]
        rect = Rectangle((left, 0), dt, N, color=cmap_(color_idx), alpha=alpha, linewidth=0)
        ax.add_patch(rect)

    # Raster plot
    if raster:
        ax.scatter(st[condition_spikes], ni[condition_spikes], s=0.1, c='k')

    # Posterior plot
    if plot_posterior and posterior is not None:
        posterior_sel = posterior[condition_time]
        for i in range(posterior_sel.shape[1]):
            ax.plot(time_axis_sel, posterior_sel[:, i] * N, lw=0.5)

    # Axis and labels
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(0, N)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Unit")
    if title:
        ax.set_title(title)

    # Legend (with correct state IDs)
    legend_elements = [Line2D([0], [0], color=cmap_(i), lw=10, label=f"{state}", alpha=alpha)
                       for i, state in enumerate(unique_states)]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
              frameon=False, ncol=ncols, title="States")

    #plt.tight_layout()
    if outf:
        plt.savefig(outf, bbox_inches='tight')
    
    if show_plot == False:
        plt.close()
    else:
        plt.show()



# ------------------------------------------------------------------------------------ #
# --------------------               k - m e a n s             ----------------------- #
# ------------------------------------------------------------------------------------ #



colorz    = ['#255D93','#5FA6D6','#B02106','#F24D33','#2C2C2C','#787878']

# ------------------------------------------------------------------------------------ #

def plot_elbow(K_range,inertia, outf : str = None, show_plot = True):
    
    fig,ax = plt.subplots(figsize=(5,4))
    ax.plot(K_range, inertia, 'o-', color=colorz[5])
    ax.set_xlabel('number of clusters')
    ax.set_ylabel('inertia')# (sum of squared distances)
    ax.set_title('elbow method for K-means\n')
    pl.set_format(ax=ax,pwr_x_max=2)
    
    if outf:
        plt.savefig(outf, bbox_inches='tight')
    
    if show_plot == False:
        plt.close()
    else:
        plt.show()
        
# ------------------------------------------------------------------------------------ #

def compute_k_elbow(data, verbose=False, outf : str = None, show_plot = True):

    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.manifold import TSNE
    from kneed import KneeLocator

    st = data['spikes'];  ch = data['channel'];    cl = data['cluster'];   
    sc = data['counts'];  pos = data['position'];  rt = data['rate'];
    matrix = sc.T
    
    # K-means (inertia method)
    inertia = []
    K_range = range(2, 6)
    for k in K_range:
        if verbose:
            print(k)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(matrix)
        inertia.append(kmeans.inertia_)
    
    # finding optimal-k
    knee_kmeans  = KneeLocator(K_range, inertia, curve="convex", direction="decreasing")
    optimal_K    = knee_kmeans.elbow if knee_kmeans.elbow else 3
    print('optimal K: ',optimal_K)
    
    # K-means with optimal-k
    kmeans          = KMeans(n_clusters=optimal_K, random_state=42, n_init=10)
    clusters_kmeans = kmeans.fit_predict(matrix)

    
    data['inertia']   = inertia
    data['k_cluster'] = clusters_kmeans
    data['kmeans']    = kmeans
    data['optimal_K'] = optimal_K

    if verbose:
        plot_elbow(K_range,inertia, outf=outf , show_plot = show_plot)

    return data

# ------------------------------------------------------------------------------------ #



# ------------------------------------------------------------------------------------ #
# --------------------                  P C A                  ----------------------- #
# ------------------------------------------------------------------------------------ #


def plot_expl_var(explained_variance, var_th = 0.95, figsize=(8,4), outf : str = None, show_plot = True):
    # Plot cumulative explained variance
    fig,ax = plt.subplots(figsize=figsize)
    ax.plot([1,len(explained_variance)],[var_th,var_th],'--k')
    ax.plot(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), markersize=5,
            marker='o', linestyle='-', color=colorz[5], lw=3)
    ax.set_xlabel("number of PC")
    ax.set_ylabel("cumulative \nexplained variance")
    ax.set_title("PCA: explained variance \nby number of components")
    pl.set_format(ax=ax,pwr_x_max=3,pwr_y_max=2)
    ax.set_ylim(-0.1,1.1)

    if outf:
        plt.savefig(outf, bbox_inches='tight')
    
    if show_plot == False:
        plt.close()
    else:
        plt.show()
        
#-----------------------------

def plot_components(mat_pca,clusters_list,kmeans=None,cmap='tab20b', outf : str = None, show_plot = True):
    
    fig,axs = plt.subplots(1,3,figsize=(18, 4))
    ax = axs[0]
    ax.scatter(mat_pca[:, 0], mat_pca[:, 1], c=clusters_list, cmap=cmap, alpha=0.8)
    if kmeans is not None:
        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=200)
    pl.set_format(ax=ax,pwr_x_max=3,pwr_y_max=2)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    ax = axs[1]
    ax.scatter(mat_pca[:, 1], mat_pca[:, 2], c=clusters_list, cmap=cmap, alpha=0.8)
    if kmeans is not None:
        ax.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], c='red', marker='x', s=200)
    pl.set_format(ax=ax,pwr_x_max=3,pwr_y_max=2)
    ax.set_xlabel('PC2')
    ax.set_ylabel('PC3')

    ax = axs[2]
    scatter = ax.scatter(mat_pca[:, 0], mat_pca[:, 2], c=clusters_list, cmap=cmap, alpha=0.8)
    if kmeans is not None:
        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 2], c='red', marker='x', s=200)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC3')
    pl.set_format(ax=ax,pwr_x_max=3,pwr_y_max=2)

    fig.suptitle('K-Means\n',y=1.1)
    fig.subplots_adjust(wspace=0.55)

    if outf:
        plt.savefig(outf, bbox_inches='tight')
    
    if show_plot == False:
        plt.close()
    else:
        plt.show()
        
#---------------------------------------------------------------------------------------------#

def plot_3D(mat_pca, clusters_list, cmap = 'tab20b', change_persp = False,gap=10, outf : str = None, show_plot = True):
    
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    
    fig  = plt.figure(figsize=(12, 10))
    ax   = fig.add_subplot(111, projection='3d')

    norm       = mcolors.Normalize(vmin=min(clusters_list), vmax=max(clusters_list))
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
    colors     = scalar_map.to_rgba(clusters_list)

    sc = ax.scatter(mat_pca[:, 0], mat_pca[:, 1], mat_pca[:, 2], c=colors, depthshade=True,
                    edgecolor='white',alpha=1, s=150)
    
    x_min = np.min(mat_pca[:,0])-gap; x_max = np.max(mat_pca[:,0])+gap
    y_min = np.min(mat_pca[:,1])-gap; y_max = np.max(mat_pca[:,1])+gap
    z_min = np.min(mat_pca[:,2])-gap; z_max = np.max(mat_pca[:,2])+gap

    proj_alpha = 0.15
    for i in range(len(mat_pca)):
        x, y, z = mat_pca[i, 0], mat_pca[i, 1], mat_pca[i, 2]
        ax.scatter(x, y, z_min, color=colors[i], edgecolor='white', linewidths=0.5, alpha=proj_alpha, s=150)
        if change_persp:
            ax.scatter(x, y_min, z, color=colors[i], edgecolor='white', linewidths=0.5, alpha=proj_alpha, s=150)
        else:
            ax.scatter(x, y_max, z, color=colors[i], edgecolor='white', linewidths=0.5, alpha=proj_alpha, s=150)
        ax.scatter(x_min, y, z, color=colors[i], edgecolor='white', linewidths=0.5, alpha=proj_alpha, s=150)

    ax.set_xlabel('PC1', labelpad=35)
    ax.set_ylabel('PC2', labelpad=35)
    ax.set_zlabel('PC3', labelpad=35)
    #ax.set_title('3D PC Projection with K-Means Clusters')

    ax.set_xlim(x_min,x_max)
    ax.set_ylim(y_min,y_max)
    ax.set_zlim(z_min,z_max)
    
    if change_persp:
        ax.view_init(10, 30)

    ax.set_box_aspect([1, 1, 0.8])
    
    
    # - - -  FORMATTER x axis
    formatter_x = ScalarFormatter(useMathText=True)   
    formatter_x.set_scientific(True)
    formatter_x.set_powerlimits((1, 2))
    ax.xaxis.set_major_formatter(formatter_x)
    ax.xaxis.offsetText.set_fontsize(20)
    
    # - - -  FORMATTER y axis
    formatter_y = ScalarFormatter(useMathText=True)    
    formatter_y.set_scientific(True) 
    formatter_y.set_powerlimits((1, 2))
    ax.yaxis.set_major_formatter(formatter_y);
    ax.yaxis.offsetText.set_fontsize(20)
    
    # - - -  FORMATTER x axis
    formatter_z = ScalarFormatter(useMathText=True)    
    formatter_z.set_scientific(True) 
    formatter_z.set_powerlimits((1, 2))
    ax.zaxis.set_major_formatter(formatter_z);
    ax.zaxis.offsetText.set_fontsize(20)

    if outf:
        plt.savefig(outf, bbox_inches='tight')
    
    if show_plot == False:
        plt.close()
    else:
        plt.show()
        


