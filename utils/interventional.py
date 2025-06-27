import numpy as np
from scipy.stats import ks_2samp
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed

#------------------------------------------------------------------------------------------------------#


def time_window( Ntrials: int, stim_units, spikes_times, channels,
                stim_start: float, stim_stop: float, Tmax: float, Delta_pre: float, Delta_post: float,
                verbose: bool = False ):
    '''
    Extract spike trains in fixed-duration windows before and after stimulation.

    Args:
        Ntrials (int): Number of trials per stimulation unit.
        stim_units (Sequence[int]): IDs of stimulation units.
        spikes_times (Dict[int, List[Dict[int, np.ndarray]]]):
            Mapping unit ID → list of trial dicts mapping channel ID → spike-time arrays.
        channels (Sequence[int]): IDs of recording channels.
        stim_start (float): Stimulation onset time.
        stim_stop (float):  Stimulation offset time.
        Tmax (float):       Duration of each analysis window.
        Delta_pre (float):  Offset before stim_start for pre-window.
        Delta_post (float): Offset after stim_stop for post-window.
        verbose (bool):     If True, print progress messages.

    Returns:
        spikes_pre (Dict[int, Dict[int, np.ndarray]]):
            Pre-stimulation  spike times rebased so 0 corresponds to window start.
        spikes_post (Dict[int, Dict[int, np.ndarray]]):
            Post-stimulation spike times rebased so 0 corresponds to window start.
    '''

    pre_start  = stim_start - Delta_pre - Tmax
    pre_end    = stim_start - Delta_pre
    post_start = stim_stop + Delta_post
    post_end   = stim_stop + Delta_post + Tmax

    spikes_pre  = {unit: {} for unit in stim_units}
    spikes_post = {unit: {} for unit in stim_units}

    if verbose:
        print(f'Window boundaries: pre=[{pre_start},{pre_end}), post=[{post_start},{post_end})')

    for unit in stim_units: # stimulation channels
        trials = spikes_times[unit]
        for ch in channels: # recording channels
            pre_segs = [trial[ch][(trial[ch] >= pre_start) & (trial[ch] < pre_end)]
                        for trial in trials]
            post_segs = [trial[ch][(trial[ch] >= post_start) & (trial[ch] < post_end)]
                         for trial in trials]
            if pre_segs:
                spikes_pre[unit][ch] = np.concatenate(pre_segs) - pre_start
            else:
                spikes_pre[unit][ch] = np.empty(0, float)
            if post_segs:
                spikes_post[unit][ch] = np.concatenate(post_segs) - post_start
            else:
                spikes_post[unit][ch] = np.empty(0, float)

    if verbose:
        print('time_window extraction complete.')

    return spikes_pre, spikes_post


#------------------------------------------------------------------------------------------------------#


def ks_on_binned_counts(pre_counts: np.ndarray, post_counts: np.ndarray):
    """
    KS test on flat binned counts vectors.
    """
    pre_flat = pre_counts.ravel()
    post_flat = post_counts.ravel()
    ks_stat, pval = ks_2samp(pre_flat, post_flat)
    return ks_stat, pval


def ks_on_raw_counts(pre_counts: np.ndarray, post_counts: np.ndarray):
    """
    KS test on raw counts per trial.
    pre_counts, post_counts: shape (Ntrials,)
    """
    ks_stat, pval = ks_2samp(pre_counts, post_counts)
    return ks_stat, pval


def diff_counts(pre_counts: np.ndarray, post_counts: np.ndarray):
    """
    Simple difference between total post and pre counts across trials.
    Returns difference and NaN for p-value.
    """
    diff = post_counts.sum() - pre_counts.sum()
    return diff, np.nan


def compute_KS(Ntrials, def_units, def_chans, def_spikes, channel,
               stim_start, Delta_pre, stim_stop, Delta_post, Tmax,
               alpha_th=0.05, id_trial_start=0, n_jobs=1, verbose=False):
    
    if verbose:
        print('\nComputing time-indep. KOLMOGOROV-SMIRNOV....')

    n_units = len(def_units)
    n_chans = len(channel)
    ntrials_used = Ntrials - id_trial_start

    # Pre-allocate spike count arrays
    count_pre  = np.zeros((n_units, n_chans, ntrials_used))
    count_post = np.zeros((n_units, n_chans, ntrials_used))

    # === Fill pre and post spike counts ===
    for i_unit, unitID in enumerate(def_units):
        for j_chan, ch in enumerate(channel):
            for k_trial, i_trial in enumerate(range(id_trial_start, Ntrials)):
                spikes = def_spikes[unitID].get(i_trial, {}).get(ch, np.array([]))

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

def compute_KS_binned(Ntrials, def_units, def_chans, def_spikes, channel,
                      stim_start, Delta_pre, stim_stop, Delta_post, Tmax,
                      Nbins=10, alpha_th=0.05, id_trial_start=0, n_jobs=1, verbose=False):
    """
    Compute KS statistics on binned spike counts (concatenated bins over trials).

    Each pre/post trial window is divided into Nbins bins of duration Tmax/Nbins.
    KS test is applied to flattened bin vectors.

    Returns:
        KS         : raw KS statistics
        KS_sign    : significant KS values after thresholding
        KS_pval_FDR: p-values after FDR correction
        KS_pval    : raw p-values
    """
    if verbose:
        print(f'\nComputing binned KS with Nbins = {Nbins}...')

    n_units = len(def_units)
    n_chans = len(channel)
    ntrials_used = Ntrials - id_trial_start
    bin_edges = np.linspace(0, Tmax, Nbins + 1)

    def bin_counts(spikes, start_time):
        counts = np.histogram(spikes - start_time, bins=bin_edges)[0]
        return counts

    # Store binned counts per trial
    count_pre  = np.zeros((n_units, n_chans, ntrials_used, Nbins))
    count_post = np.zeros((n_units, n_chans, ntrials_used, Nbins))

    for i_unit, unitID in enumerate(def_units):
        for j_chan, ch in enumerate(channel):
            for k_trial, i_trial in enumerate(range(id_trial_start, Ntrials)):
                spikes = def_spikes[unitID].get(i_trial, {}).get(ch, np.array([]))
                
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