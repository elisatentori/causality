import numpy as np
import scipy.io

# ================================================================================================================ #
# compute spike counts
def compute_spike_counts(spikes, channel=None, binsize=0.02):
    spike_times = np.asarray(spikes[0])
    neuron_ids  = np.asarray(spikes[1], dtype=int)

    if channel is not None:
        channel = np.asarray(channel, dtype=int)
        mask = np.isin(neuron_ids, channel)
        spike_times = spike_times[mask]
        neuron_ids  = neuron_ids[mask]
        neuron_id_map = {n_id: i for i, n_id in enumerate(channel)}
        num_neurons = len(channel)
    else:
        unique_neurons = np.unique(neuron_ids)
        neuron_id_map  = {n_id: i for i, n_id in enumerate(unique_neurons)}
        num_neurons    = len(unique_neurons)

    if spike_times.size == 0:
        return np.zeros((0, num_neurons), dtype=int)

    mapped_neuron_ids = np.fromiter((neuron_id_map[nid] for nid in neuron_ids), dtype=int, count=len(neuron_ids))

    spike_bins = (spike_times / binsize).astype(int)
    num_bins   = spike_bins.max() + 1

    spike_counts = np.zeros((num_bins, num_neurons), dtype=int)
    np.add.at(spike_counts, (spike_bins, mapped_neuron_ids), 1)
    return spike_counts



# ================================================================================================================ #
# reorder an array based on cluster labels

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

# ================================================================================================================ #
# Loads the features of the recording: number of neurons/channels, list of recording channels and electrods with
# related spatial coordinates,  list of stimulating channels and electrods with related spatial coordinates, rate. 
#
    
def load_original_data(main_original,sim_folder, file='Cult.mat'):

    path_or_data    = main_original+sim_folder
    data            = scipy.io.loadmat(path_or_data+file)

    # data
    nNeurons     = data['nNeurons'][0][0]
    channel      = data['channel'].flatten()
    electrode    = data['mapping']['electrode'].flatten()
    pos          = data['pos']
    cluster_orig = data['clusters'].flatten()
    cluster      = _reorder_clusters_by_x(pos, cluster_orig)

    stim_channel = data['stimMap']['channel'][0, 0].flatten()
    stim_x       = data['stimMap']['x'][0, 0].flatten()
    stim_y       = data['stimMap']['y'][0, 0].flatten()
    stim_pos     = np.column_stack((stim_x, stim_y))

    rate         = data['rate'].flatten()
    
    return nNeurons, channel, electrode[0].flatten(), pos, cluster, stim_channel, stim_pos, rate

# ================================================================================================================ #
# Erase spikes during stimulation and the trial onset 

def erase_artifacts(channel, def_units, def_spikes, Tstim_start_s=1, Tstim_duration_s=0.12, Tonset_duration=0.12,
                     erase_min=1.00-0.001, erase_max=1.12, erase_first=0.0, ipi_sec=0.001, Ntrials=200, fs=20000):

    erase_min_samp   = int(round(erase_min   * fs))
    erase_max_samp   = int(round(erase_max   * fs))
    erase_first_samp = int(round(erase_first * fs))
    
    Tstim_start_s_samp    = int(round(Tstim_start_s    * fs))
    Tstim_duration_s_samp = int(round(Tstim_duration_s * fs))
    Tonset_duration_samp  = int(round(Tonset_duration  * fs))
    ipi_samp              = int(round(ipi_sec          * fs))
    
    modified_spikes = {}
    for n_stim in def_units:
        modified_spikes[n_stim] = {}
        for n_trial in range(Ntrials):
            modified_spikes[n_stim][n_trial] = {}
            for n_channel in range(1024):
                s                 = def_spikes[n_stim][n_trial][n_channel]
                cond_outside_stim = (s < erase_min_samp) | (s >= erase_max_samp)
                cond_after_first  = s > erase_first_samp
                modified_spikes[n_stim][n_trial][n_channel] = s[cond_outside_stim & cond_after_first]
                
    return modified_spikes

# ================================================================================================================ #
# stack all trials : trial 1 –> all stim chans; trial 2 –> all stim stim chans; an so on...

def stack_trials(channel, def_units, spikes, T_spont=1800., DeltaT_sec=3., Ntrials=200, fs=20000, path_out=None, one=False):

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

# ================================================================================================================ #
# do the average of the spike counts (bin per bin) across trials

def trial_average_bin(spikes, def_units, channel=None, DeltaT_sec=4, Ntrials=200, bin_sz=0.05, t_start_sec=0.0, win_sec=3.8, out_dtype=np.float32, order="units_time_channels"):

    s_counts      = compute_spike_counts(spikes, channel=channel, binsize=bin_sz)   # (n_timebins, n_channels)
    n_units, n_ch = len(def_units), s_counts.shape[1]

    T_len_bin   = int(win_sec / bin_sz)       # time window length in n. bins
    t_start_bin = int(t_start_sec / bin_sz)   # start time in n. bins
    trial_block = n_units * float(DeltaT_sec) # trial block length in seconds

    # start bins for each (trial, unit), then shift by offset t_start_bin
    starts = ((trial_block * np.arange(Ntrials)[:, None] + float(DeltaT_sec) * np.arange(n_units)[None, :]) / bin_sz).astype(int) + t_start_bin

    if starts.min() < 0 or (starts.max() + T_len_bin) > s_counts.shape[0]:
        raise ValueError("Requested window exceeds s_counts time range (check t_start_sec/win_sec/bin_sz).")

    out  = np.empty((n_units, T_len_bin, n_ch), dtype=out_dtype)
    bins = np.arange(T_len_bin)[None, :]  # (1, T_len_bin)

    for i_unit in range(n_units):
        idx = starts[:, i_unit][:, None] + bins     # (Ntrials, T_len_bin)
        out[i_unit] = s_counts[idx].mean(axis=0)    # (T_len_bin, n_ch)

    # time axis (seconds) relative to unit start, shifted by t_start_sec
    t = t_start_sec + np.arange(T_len_bin) * bin_sz

    if order == "units_channels_time":
        out = out.transpose(0, 2, 1)  # (n_units, n_ch, T_len_bin)
    return out, t

# ================================================================================================================ #
# gives all trials first bin from a certain starting point t_start_sec

def firstbin_alltrials(spikes, channel, def_units, bin_sz=0.05, DeltaT_sec=4, Ntrials=200, t_start_sec=0.0):
    spike_times, spike_chans = spikes

    if spike_times.size and np.any(spike_times[1:] < spike_times[:-1]):
        order = np.argsort(spike_times)
        spike_times, spike_chans = spike_times[order], spike_chans[order]

    n_units = len(def_units)
    n_ch    = len(channel)

    max_chan_id = int(max(spike_chans.max(initial=0), channel.max(initial=0)))
    chan_to_col = np.full(max_chan_id + 1, -1, dtype=np.int64)
    chan_to_col[channel] = np.arange(n_ch, dtype=np.int64)

    trial_block_sec = n_units * float(DeltaT_sec)
    win_start = (trial_block_sec * np.arange(Ntrials)[:, None] +
                 float(DeltaT_sec) * np.arange(n_units)[None, :] +
                 float(t_start_sec)).ravel()
    win_end = win_start + float(bin_sz)
    n_win = win_start.size

    col_idx = chan_to_col[spike_chans]
    keep = col_idx >= 0
    if not np.any(keep):
        return np.zeros((n_units, n_ch, Ntrials), dtype=np.int32)

    spike_times = spike_times[keep]
    col_idx     = col_idx[keep]

    win_idx = np.searchsorted(win_start, spike_times, side="right") - 1
    in_win  = (win_idx >= 0) & (spike_times < win_end[win_idx])
    if not np.any(in_win):
        return np.zeros((n_units, n_ch, Ntrials), dtype=np.int32)

    win_idx = win_idx[in_win]
    col_idx = col_idx[in_win]

    flat_idx = win_idx * n_ch + col_idx
    counts_win_by_ch = np.bincount(flat_idx, minlength=n_win * n_ch).reshape(n_win, n_ch)

    return counts_win_by_ch.reshape(Ntrials, n_units, n_ch).transpose(1, 2, 0).astype(np.int32, copy=False)
