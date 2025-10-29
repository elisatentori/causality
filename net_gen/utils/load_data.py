import numpy as np
import scipy.io

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
