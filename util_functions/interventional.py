import numpy as np
from scipy.stats import ks_2samp
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed

#------------------------------------------------------------------------------------------------------#

def ic_save(_path_,ic,ic_pval,ic_pval_FDR,suffix=''):
    
    import pickle
    with open(_path_+'ks'+suffix+'.pkl', 'wb') as f:
        pickle.dump(ic, f)

    with open(_path_+'ks_pval'+suffix+'.pkl', 'wb') as f:
        pickle.dump(ic_pval, f)

    with open(_path_+'ks_pval_FDR'+suffix+'.pkl', 'wb') as f:
        pickle.dump(ic_pval_FDR, f)
        
def ic_load(_path_,suffix=''):

    import pickle
    with open(_path_+'ks'+suffix+'.pkl', 'rb') as f:
        ic = pickle.load(f)

    with open(_path_+'ks_pval'+suffix+'.pkl', 'rb') as f:
        ic_pval = pickle.load(f)
        
    with open(_path_+'ks_pval_FDR'+suffix+'.pkl', 'rb') as f:
        ic_pval_FDR = pickle.load(f)
        
    return ic, ic_pval, ic_pval_FDR
    
#------------------------------------------------------------------------------------------------------#

def compute_KS(Ntrials, stim_units, stim_chans, spikes_times, rec_channels,
               stim_start, Delta_pre, stim_stop, Delta_post, Tmax,
               alpha_th=0.05, id_trial_start=0, n_jobs=1, verbose=False):
    
    ''' 
    Compute KS statistics trials spike counts (NO concatenated bins over trials).
    KS test is applied to flattened vectors of spike-count for each trial.
    
    Args:
        Ntrials (int):                Number of trials per stimulation unit.
        stim_units (Sequence[int]):   IDs of stimulation units.
        stim_chans (Array[int]):      IDs of stimulation channels.
        spikes_times ( Dict[int, Dict[int, Dict[int, np.ndarray]]] ):
                                      Mapping: unit ID -> trial dicts -> rec channel ID -> spike-time arrays.
        rec_channels (Sequence[int]): IDs of recording channels.
        stim_start (float):           Stimulation onset time.
        stim_stop (float):            Stimulation offset time.
        Tmax (float):                 Duration of each analysis window.
        Delta_pre (float):            Offset before stim_start for pre-window.
        Delta_post (float):           Offset after stim_stop for post-window.
        alpha_th (float):             threshold for p-value
        id_trial_start (int):         if you want to exclude some initial trials change from id_trial_start=0
        n_jobs (int) [default 1] :    N. jobs for joblib
        verbose (bool):               If True, print progress messages.
    Returns:
        KS         : raw KS statistics
        KS_sign    : significant KS values after thresholding
        KS_pval_FDR: p-values after FDR correction
        KS_pval    : raw p-values
    '''
    
    if verbose:
        print('\nComputing time-indep. KOLMOGOROV-SMIRNOV....')

    n_units = len(stim_units)
    n_chans = len(rec_channels)
    ntrials_used = Ntrials - id_trial_start

    # Pre-allocate spike count arrays
    count_pre  = np.zeros((n_units, n_chans, ntrials_used))
    count_post = np.zeros((n_units, n_chans, ntrials_used))

    # === Fill pre and post spike counts ===
    for i_unit, unitID in enumerate(stim_units):
        for j_chan, ch in enumerate(rec_channels):
            for k_trial, i_trial in enumerate(range(id_trial_start, Ntrials)):
                spikes = spikes_times[unitID].get(i_trial, {}).get(ch, np.array([]))

                pre_mask  = (spikes >= stim_start - Delta_pre - Tmax) & (spikes < stim_start - Delta_pre)
                post_mask = (spikes >  stim_stop + Delta_post) & (spikes <= stim_stop + Delta_post + Tmax)

                count_pre[i_unit, j_chan, k_trial]  = np.sum(pre_mask)
                count_post[i_unit, j_chan, k_trial] = np.sum(post_mask)

    # === KS test ===
    def ks_for_unit(i_unit):
        ks_vals = np.zeros(n_chans)
        p_vals = np.ones(n_chans)
        for j_chan in range(n_chans):
            ks_stat, p_val = ks_2samp(count_pre[i_unit, j_chan], count_post[i_unit, j_chan])
            ks_vals[j_chan] = ks_stat
            p_vals[j_chan]  = p_val
        return ks_vals, p_vals

    if n_jobs!=1:
        results = Parallel(n_jobs=n_jobs)(delayed(ks_for_unit)(i) for i in range(n_units))
        KS      = np.array([r[0] for r in results])
        KS_pval = np.array([r[1] for r in results])
    else:
        KS      = np.zeros((n_units, n_chans))
        KS_pval = np.ones((n_units, n_chans))
        for i in range(n_units):
            KS[i], KS_pval[i] = ks_for_unit(i)

    # === FDR correction ===
    KS_pval_flat = KS_pval.flatten()
    mask = KS_pval_flat < 1
    KS_pval_FDR_flat = np.ones_like(KS_pval_flat)

    if np.any(mask):
        _, p_corr, _, _ = multipletests(KS_pval_flat[mask], alpha=alpha_th, method='fdr_bh')
        KS_pval_FDR_flat[mask] = p_corr

    KS_pval_FDR = KS_pval_FDR_flat.reshape(KS_pval.shape)


    # === Significant KS statistics ===
    KS_sign = np.where(KS_pval < alpha_th, KS, 0.0)

    if verbose:
        print('....done\n')
    return KS, KS_sign, KS_pval_FDR, KS_pval

#------------------------------------------------------------------------------------------------------#

def compute_KS_binned(Ntrials, stim_units, stim_chans, spikes_times, rec_channels,
                      stim_start, Delta_pre, stim_stop, Delta_post, Tmax,
                      Nbins=10, alpha_th=0.05, id_trial_start=0, n_jobs=1, verbose=False):
    """
    Compute KS statistics on binned spike counts (concatenated bins over trials).

    Each pre/post trial window is divided into Nbins bins of duration Tmax/Nbins.
    KS test is applied to flattened bin vectors.
    
    Args:
        Ntrials (int):                Number of trials per stimulation unit.
        stim_units (Sequence[int]):   IDs of stimulation units.
        stim_chans (Array[int]):      IDs of stimulation channels.
        spikes_times ( Dict[int, Dict[int, Dict[int, np.ndarray]]] ):
                                      Mapping: unit ID -> trial dicts -> rec channel ID -> spike-time arrays.
        rec_channels (Sequence[int]): IDs of recording channels.
        stim_start (float):           Stimulation onset time.
        stim_stop (float):            Stimulation offset time.
        Tmax (float):                 Duration of each analysis window.
        Delta_pre (float):            Offset before stim_start for pre-window.
        Delta_post (float):           Offset after stim_stop for post-window.
        Nbins (int):                  Each pre/post trial window (duration Tmax) is divided into Nbins bins
        alpha_th (float):             threshold for p-value
        id_trial_start (int):         if you want to exclude some initial trials change from id_trial_start=0
        n_jobs (int) [default 1] :    N. jobs for joblib
        verbose (bool):               If True, print progress messages.

    Returns:
        KS         : raw KS statistics
        KS_sign    : significant KS values after thresholding
        KS_pval_FDR: p-values after FDR correction
        KS_pval    : raw p-values
        
    """
    if verbose:
        print(f'\nComputing binned KS with Nbins = {Nbins}...')

    n_units = len(stim_units)
    n_chans = len(rec_channels)
    ntrials_used = Ntrials - id_trial_start
    bin_edges = np.linspace(0, Tmax, Nbins + 1)

    def bin_counts(spikes, start_time):
        counts = np.histogram(spikes - start_time, bins=bin_edges)[0]
        return counts

    # Store binned counts per trial
    count_pre  = np.zeros((n_units, n_chans, ntrials_used, Nbins))
    count_post = np.zeros((n_units, n_chans, ntrials_used, Nbins))

    for i_unit, unitID in enumerate(stim_units):
        for j_chan, ch in enumerate(rec_channels):
            for k_trial, i_trial in enumerate(range(id_trial_start, Ntrials)):
                spikes = spikes_times[unitID].get(i_trial, {}).get(ch, np.array([]))
                
                # Select and bin pre spikes
                pre_start = stim_start - Delta_pre - Tmax
                pre_end   = stim_start - Delta_pre
                pre_spk   = spikes[(spikes >= pre_start) & (spikes < pre_end)]
                count_pre[i_unit, j_chan, k_trial] = bin_counts(pre_spk, pre_start)

                # Select and bin post spikes
                post_start = stim_stop + Delta_post
                post_end   = stim_stop + Delta_post + Tmax
                post_spk   = spikes[(spikes >= post_start) & (spikes < post_end)]
                count_post[i_unit, j_chan, k_trial] = bin_counts(post_spk, post_start)

    # === KS test on binned vectors ===
    def ks_binned_unit(i_unit):
        ks_vals = np.zeros(n_chans)
        p_vals = np.ones(n_chans)
        for j_chan in range(n_chans):
            pre_vec  = count_pre[i_unit, j_chan].ravel()
            post_vec = count_post[i_unit, j_chan].ravel()
            ks_stat, p_val = ks_2samp(pre_vec, post_vec)
            ks_vals[j_chan] = ks_stat
            p_vals[j_chan]  = p_val
        return ks_vals, p_vals

    if n_jobs != 1:
        results = Parallel(n_jobs=n_jobs)(delayed(ks_binned_unit)(i) for i in range(n_units))
        KS      = np.array([r[0] for r in results])
        KS_pval = np.array([r[1] for r in results])
    else:
        KS      = np.zeros((n_units, n_chans))
        KS_pval = np.ones((n_units, n_chans))
        for i in range(n_units):
            KS[i], KS_pval[i] = ks_binned_unit(i)

    # === FDR correction ===
    KS_pval_flat = KS_pval.flatten()
    mask = KS_pval_flat < 1
    KS_pval_FDR_flat = np.ones_like(KS_pval_flat)
    if np.any(mask):
        _, p_corr, _, _ = multipletests(KS_pval_flat[mask], alpha=alpha_th, method='fdr_bh')
        KS_pval_FDR_flat[mask] = p_corr
    KS_pval_FDR = KS_pval_FDR_flat.reshape(KS_pval.shape)

    # === Threshold for significance ===
    KS_sign = np.where(KS_pval < alpha_th, KS, 0.0)

    if verbose:
        print('....done\n')

    return KS, KS_sign, KS_pval_FDR, KS_pval