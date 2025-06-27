import numpy as np
import scipy.io
import matplotlib.pyplot as plt

class SpikeDataProcessor:

    def __init__(self, main_path, sim_folder, T_cutoff = 1800, stim_ST=None, stim_CH=None, spike_file='spikeTimes_binsize0.05.txt', fs=20000):
        self.main_path  = main_path
        self.sim_folder = sim_folder
        self.spike_file = spike_file
        self.fs         = fs
        self.stim_ST    = stim_ST
        self.stim_CH    = stim_CH
        self.T_cutoff   = T_cutoff

        self._load_data()
        self._load_spike_data()
        self._sort_by_clusters()

        self.views = {}
        self._prepare_views()

    # ------- original data ----------

    def _compute_firing_rate(self, spikes, t_stop=None):
        spike_times = spikes[0]
        neuron_ids  = spikes[1].astype(int)
    
        if t_stop is None:
            t_stop = spike_times.max()
    
        channel_id_to_index = {ch: i for i, ch in enumerate(self.channel)}
    
        spike_counts = np.zeros(len(self.channel), dtype=int)
    
        for nid in neuron_ids:
            if nid in channel_id_to_index:
                idx = channel_id_to_index[nid]
                spike_counts[idx] += 1
    
        firing_rates = spike_counts / t_stop
        return firing_rates
        

    def _reorder_clusters_by_x(self, pos, cluster):
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
        
    def _load_data(self):
        path = self.main_path + self.sim_folder
        data = scipy.io.loadmat(path + 'Cult.mat')
        
        self.nNeurons     = data['nNeurons'][0][0]
        self.channel      = data['channel'].flatten() - 1  # start from 0
        self.elec         = data['mapping']['electrode'][0][0].flatten()
        self.pos          = data['pos'] / 1000
        cluster_orig      = data['clusters'].flatten()
        self.cluster      = self._reorder_clusters_by_x(self.pos, cluster_orig)
        
        self.stim_channel = data['stimMap']['channel'][0, 0].flatten() -1
        stim_x            = data['stimMap']['x'][0, 0].flatten()
        stim_y            = data['stimMap']['y'][0, 0].flatten()
        self.stim_pos     = np.column_stack((stim_x, stim_y))
    
    def _load_spike_data(self):
        
        def load_spikeTimes(main_path, T_cutoff=10000, stim_ST=None, stim_CH=None):
            rows, cols = np.loadtxt(main_path, unpack=True,usecols=(0,1))
            if stim_ST is not None and stim_CH is not None:
                return (np.concatenate([cols[cols<T_cutoff*self.fs],stim_ST])/self.fs).astype(float), np.concatenate([(rows-1)[cols<T_cutoff*self.fs],stim_CH]).astype(int)
            else:
                return ((cols[cols<T_cutoff*self.fs])/self.fs).astype(float),((rows-1)[cols<T_cutoff*self.fs]).astype(int)
        
        path         = self.main_path + self.sim_folder + self.spike_file

        if self.stim_ST is not None and self.stim_CH is not None:
            self.spikes  = load_spikeTimes(path, self.T_cutoff, self.stim_ST, self.stim_CH)
        else:
            self.spikes  = load_spikeTimes(path, self.T_cutoff)
        self.T       = np.ceil(np.max(self.spikes[0]))
        self.rate    = self._compute_firing_rate(self.spikes)
        #spike_times  = self.spikes[0]

    # ------- sording data by clusters ----------
    
    def _sort_by_clusters(self):
        sorted_indices   = np.argsort(self.cluster)
        
        sorted_channels  = self.channel[sorted_indices]
        new_idxs         = np.full(1024, -1, dtype=int)
        for new_pos, ch in enumerate(sorted_channels):
            new_idxs[ch] = new_pos
        
        self.channel_map    = new_idxs[sorted_channels]
        self.cluster_map    = self.cluster[sorted_indices]
        self.electrodes_map = self.elec[sorted_indices]
        self.positions_map  = self.pos[sorted_indices]
        self.rate_map       = self.rate[sorted_indices]
        
        self.stim_channel_map     = new_idxs[self.stim_channel]
        self.stim_positions_map   = np.copy(self.stim_pos)
        
        # Apply to spike data
        spike_channels             = self.spikes[1].astype(int)
        sorted_spike_channels      = new_idxs[spike_channels]
        self.sorted_spikes         = np.stack([self.spikes[0], sorted_spike_channels], axis=0)

    # ------- subsampling data ----------
    
    def _subsample_indices(self, view):
        # Extract info from view (original or sorted)
        channel = view['channel']
        cluster = view['cluster']
        rate    = view['rate']
        
        cluster = np.asarray(cluster)
        n_clusters = cluster.max() + 1
        best_indices = []
        
        for c in range(n_clusters):
            indices = np.where(cluster == c)[0]
            if len(indices) == 0:
                continue
            cluster_rates = rate[indices]
            max_rate = np.max(cluster_rates)
            max_pos = np.where(cluster_rates == max_rate)[0]
            chosen = indices[np.random.choice(max_pos)]
            best_indices.append(chosen)
        
        return np.array(best_indices)
        
    def _subsample_spikes(self, spikes, keep_ids):
        spike_times = spikes[0]
        neuron_ids = spikes[1].astype(int)
        mask = np.isin(neuron_ids, keep_ids)
        return np.stack([spike_times[mask], neuron_ids[mask]], axis=0)

    
    # ------- aggregate data ----------
    
    def _aggregate_spikes(self, spikes, channel, cluster, sub_channel):
        spike_times = spikes[0]
        neuron_ids  = spikes[1].astype(int)
    
        max_chan = np.max(channel) + 1
        channel_to_cluster = np.full(max_chan, -1, dtype=int)
        channel_to_cluster[channel] = cluster
    
        sub_cluster = channel_to_cluster[sub_channel]
        valid_mask = sub_cluster != -1
        sub_channel = sub_channel[valid_mask]
        sub_cluster = sub_cluster[valid_mask]
    
        max_cluster = np.max(cluster) + 1
        cluster_to_dominant = np.full(max_cluster, -1, dtype=int)
        cluster_to_dominant[sub_cluster] = sub_channel
    
        chan_cluster = channel_to_cluster[channel]
        channel_to_dominant = cluster_to_dominant[chan_cluster]
    
        lookup = np.full(max_chan, -1, dtype=int)
        lookup[channel] = channel_to_dominant
    
        new_ids = lookup[neuron_ids]
        return np.stack([spike_times, new_ids], axis=0)

    
    # ------------------------------------------------------------------------------------------------- #
    # give data types : original, sorted by channel-cluster-unit, subsampled, aggregated by cluster unit
    
    def _prepare_views(self):

        # === ORIGINAL VIEW ===
        self.views['original'] = {
            'spikes'        : self.spikes,
            'channel'       : self.channel,
            'cluster'       : self.cluster,
            'electrode'     : self.elec,
            'position'      : self.pos,
            'stim_channel'  : self.stim_channel,
            'stim_position' : self.stim_pos,
            'rate'          : self.rate
        }
        
        # === SORTED BY CLUSTER VIEW ===
        self.views['sorted'] = {
            'spikes'        : self.sorted_spikes ,
            'channel'       : self.channel_map,
            'cluster'       : self.cluster_map,
            'electrode'     : self.electrodes_map,
            'position'      : self.positions_map,
            'stim_channel'  : self.stim_channel_map,
            'stim_position' : self.stim_positions_map,
            'rate'          : self.rate_map
        }
        
        # === SUBSAMPLED VIEW ===
        base = self.views['sorted']
        keep_idx = self._subsample_indices(base)
        
        subsampled_channel      = base['channel'][keep_idx]
        subsampled_cluster      = base['cluster'][keep_idx]
        subsampled_elec         = base['electrode'][keep_idx]
        subsampled_pos          = base['position'][keep_idx]
        subsampled_rate         = base['rate'][keep_idx]
        subsampled_stim_channel = base['stim_channel']
        subsampled_stim_pos     = base['stim_position']
        
        subsampled_spikes = self._subsample_spikes(base['spikes'], subsampled_channel)
        
        self.views['subsampled'] = {
            'spikes'        : subsampled_spikes,
            'channel'       : subsampled_channel,
            'cluster'       : subsampled_cluster,
            'electrode'     : subsampled_elec,
            'position'      : subsampled_pos,
            'stim_channel'  : subsampled_stim_channel,
            'stim_position' : subsampled_stim_pos,
            'rate'          : subsampled_rate
        }
        
        # === AGGREGATED VIEW ===
        base = self.views['sorted']
        keep_idx = self._subsample_indices(base) # use the same mapping of the subsampling, because we are putting together the spikes o
        
        agg_channel        = base['channel'][keep_idx]
        agg_cluster        = base['cluster'][keep_idx]
        channel_to_cluster = dict(zip(base['channel'], base['cluster']))   
        agg_stim_channel   = np.array([channel_to_cluster.get(ch, -1) for ch in base['stim_channel']])
        agg_elec           = base['electrode'][keep_idx]
        agg_pos            = base['position'][keep_idx]
        agg_stim_pos       = base['stim_position']
        
        agg_spikes         = self._aggregate_spikes(base['spikes'], base['channel'], base['cluster'], agg_channel)
        agg_rate           = self._compute_firing_rate(agg_spikes)
        
        self.views['aggregated'] = {
            'spikes'        : agg_spikes,
            'channel'       : agg_channel,
            'cluster'       : agg_cluster,
            'electrode'     : agg_elec,
            'position'      : agg_pos,
            'stim_channel'  : agg_stim_channel,
            'stim_position' : agg_stim_pos,
            'rate'          : agg_rate
        }


    def get(self, mode='original',verbose=False):
        if verbose:
            print("Available views:", list(self.views.keys()))
        if mode in self.views:
            return self.views[mode]
        else:
            raise ValueError(f"View '{mode}' not available. Options: {list(self.views.keys())}")



def compute_spike_counts(spikes, binsize=0.02):
    """
    Estimate spike counts in bins of given size and return a T x N array.
    
    Parameters:
    spikes (numpy array): 2D array with first row as spike times in seconds and second row as neuron IDs.
    binsize (float):      Size of the bins in seconds. Default is 20ms.
    
    Returns:
    numpy array:          T x N array with spike counts, where T is the number of bins and N is the number of neurons.
    """
    # Extract spike times and neuron IDs
    spike_times = spikes[0]
    neuron_ids  = spikes[1].astype(int)
    
    # Determine the number of unique neurons
    unique_neurons = np.unique(neuron_ids)
    num_neurons    = len(unique_neurons)
    
    # Map neuron IDs to a continuous range starting from 0
    neuron_id_map     = {neuron_id: i for i, neuron_id in enumerate(unique_neurons)}
    mapped_neuron_ids = np.array([neuron_id_map[neuron_id] for neuron_id in neuron_ids])
    
    # Convert spike times to bin indices
    spike_bins   = (spike_times / binsize).astype(int)
    
    # Determine the number of bins
    num_bins     = spike_bins.max() + 1
    
    # Initialize the T x N array with zeros
    spike_counts = np.zeros((num_bins, num_neurons), dtype=int)
    
    # Use np.add.at to accumulate spike counts
    np.add.at(spike_counts, (spike_bins, mapped_neuron_ids), 1)
    
    return spike_counts