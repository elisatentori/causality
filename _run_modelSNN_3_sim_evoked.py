import argparse
import time
#------------------------------------------------------------#
import numpy as np
import pickle
from matplotlib import pyplot as plt
#------------------------------------------------------------#
from utils import colormaps as maps
from utils import load_data as ld
from utils import plot as pl
from utils import distance as di
from utils import interventional as IC
#------------------------------------------------------------#
from utils_izhi import topology as tp
from utils_izhi import izhikevic as iz
#------------------------------------------------------------#
# snn
from snn import channels as ut
#------------------------------------------------------------#
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

#------------------------------------------------------------#

# ---- runtime parallel config (read from SLURM) ----
import os
try:
    N_JOBS = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
except Exception:
    N_JOBS = 1
print(f"[cfg] Using N_JOBS={N_JOBS} worker(s)")
#------------------------------------------------------------#


from joblib import Parallel, delayed
def build_def_spikes(pre_filtered_spikes, channels, Ntrials, dT_trial, n_units, include_right_boundary=True):
    """
    Restituisce def_spikes con la stessa struttura di process_unit:
      def_spikes[i_unit][i_trial][ch] = array tempi relativi (step)
    Pre-inizializza TUTTI i canali per ogni (unit,trial) con array vuoti,
    così le chiavi esistono sempre ed eviti KeyError.
    Se include_right_boundary=True, duplica gli spike esattamente al confine
    (t_rel==0) nel trial precedente, con t_rel=dT_trial.
    """
    channels = np.asarray(channels, dtype=int)
    unit_len = Ntrials * dT_trial  # passi per "blocco unità"

    # prefill: tutti i (unit, trial, ch) -> array vuoto
    EMPTY = np.empty(0, dtype=np.int32)
    def_spikes = {i: {j: {int(ch): EMPTY for ch in channels} for j in range(Ntrials)}for i in range(n_units)}

    for ch in channels:
        t = pre_filtered_spikes[ch]  # array ordinato di indici (steps)
        if t.size == 0:
            continue

        u = t // unit_len                 # unit index
        r = t - u * unit_len              # offset all'interno del blocco unità
        tr = r // dT_trial                # trial index [0..Ntrials-1]
        t_rel = r - tr * dT_trial         # tempo relativo nel trial

        # Assegnazione "pulita" [start, end)
        keys = u * Ntrials + tr
        uniq, idx, cnt = np.unique(keys, return_index=True, return_counts=True)
        for k, s, c in zip(uniq, idx, cnt):
            i_unit = int(k // Ntrials)
            i_trial = int(k % Ntrials)
            if 0 <= i_unit < n_units:
                def_spikes[i_unit][i_trial][ch] = t_rel[s:s+c]

        if include_right_boundary:
            # Spike esattamente al confine del trial (t_rel == 0) -> duplicali anche nel trial precedente
            # (tr > 0 per evitare di duplicare prima dell'inizio del blocco unità)
            mask = (t_rel == 0) & (tr > 0)
            if np.any(mask):
                keys_prev = u[mask] * Ntrials + (tr[mask] - 1)
                # nel trial precedente il tempo relativo è esattamente dT_trial (come t - time del tuo codice)
                t_rel_prev = np.full(mask.sum(), dT_trial, dtype=t_rel.dtype)
                uniq_p, idx_p, cnt_p = np.unique(keys_prev, return_index=True, return_counts=True)
                # raccogliamo in ordine: essendo t ordinato, questi tempi (t_rel=dT_trial) finiranno in coda
                # del trial precedente, come nel tuo slicing
                for k, s, c in zip(uniq_p, idx_p, cnt_p):
                    i_unit = int(k // Ntrials)
                    i_trial = int(k % Ntrials)
                    if 0 <= i_unit < n_units:
                        prev = def_spikes[i_unit][i_trial].get(ch)
                        block = t_rel_prev[s:s+c]
                        if prev is None:
                            def_spikes[i_unit][i_trial][ch] = block
                        else:
                            def_spikes[i_unit][i_trial][ch] = np.concatenate([prev, block])

    return def_spikes


#=================================================================================================#
# For Avenir font. To use it, you have to download the font-style files below.
# Comment the lines referred to fontManager if you don't need to set the font.
from matplotlib import font_manager, rcParams
font_file   = "/home/tentori/.local/avenir_ff/AvenirLTStd-Roman.otf"
font_file_b = "/home/tentori/.local/avenir_ff/AvenirLTStd-Black.otf"
font_file_c = "/home/tentori/.local/avenir_ff/AvenirLTStd-Book.otf"
font_manager.fontManager.addfont(font_file)
font_manager.fontManager.addfont(font_file_b)
font_manager.fontManager.addfont(font_file_c)
#------------------------------------------------------------#
rcParams['font.family']  = "Avenir LT Std"
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype']  = 42
#------------------------------------------------------------#
DIM = 25
plt.rcParams.update({'font.size': DIM,'axes.labelsize': DIM,'axes.titlesize': DIM,'xtick.labelsize': DIM,'ytick.labelsize': DIM})
#=================================================================================================#
# Colors
colorz    = ['#255D93','#5FA6D6','#B02106','#F24D33','#2C2C2C','#787878']
coldhot_cmap   = maps.create_cmaphot()
coldhot_cmap_r = maps.create_cmaphot_r()
#------------------------------------------------------------#
show_plot   = True
#------------------------------------------------------------#
from cmap import Colormap
def Cmap(cmap):
    return Colormap(cmap).to_mpl()
#------------------------------------------------------------#
def truncate_cmap(cmap, minval=0.0, maxval=0.8, n=256):
    from matplotlib.colors import Normalize, LinearSegmentedColormap
    return LinearSegmentedColormap.from_list( f'{cmap.name}_trunc', cmap(np.linspace(minval, maxval, n)) )
#------------------------------------------------------------#
def discrete_cmap(cmap_name='turbo', num_colors=25, minval=0., maxval=1):
    from matplotlib.colors import ListedColormap
    base = truncate_cmap(plt.get_cmap(cmap_name), minval=minval, maxval=maxval)
    return ListedColormap(base(np.linspace(0, 1, num_colors)))
#=================================================================================================#
# to create folder
def Set_Dir_Plots(path):
    import os
    if not os.path.exists(path):
        os.mkdir(path)
#=================================================================================================#

#=================================================================================================#
#                                       A R G P A R S E
#

my_parser = argparse.ArgumentParser(description='Arguments to pass')

my_parser.add_argument('path_results',      metavar='path_results_',   type=str,    help='Main path folder for results')
my_parser.add_argument('modules',           metavar='modules_',        type=str,    help='network modules number')

my_parser.add_argument('dist_rule',         metavar='dist_rule_',      type=str,    help='EDR or random')
my_parser.add_argument('nNeurons',          metavar='nNeurons_',       type=int,    help='number of network nodes')

#my_parser.add_argument('local_II',          metavar='local_II_',       type=bool,   help='local connections II')
my_parser.add_argument('local_II',          type=int,             choices=[0,1],    help='0 = no local II, 1 = local II')

# STD PARAMETERS
my_parser.add_argument('tau_AMPA',          metavar='tau_AMPA_',       type=float,  help='tau_AMPA')
my_parser.add_argument('tau_GABA',          metavar='tau_GABA_',       type=float,  help='tau_GABA')
my_parser.add_argument('tau_R',             metavar='tau_R_',          type=float,  help='tau_R')
my_parser.add_argument('beta_E',            metavar='beta_E_',         type=float,  help='beta_E')
my_parser.add_argument('beta_I',            metavar='beta_I_',         type=float,  help='beta_I')

# GAIN
my_parser.add_argument('gain_AMPA',         metavar='gain_AMPA_',      type=float,  help='gain_AMPA')
my_parser.add_argument('gain_GABA',         metavar='gain_GABA_',      type=float,  help='gain_GABA')

# COUPLING
my_parser.add_argument('g_E',               metavar='g_E_',            type=float,  help='g_E')
my_parser.add_argument('g_I',               metavar='g_I_',            type=float,  help='g_I')

# NOISE
my_parser.add_argument('I_intensity_exc',  metavar='I_intensity_exc_', type=float,  help='I_intensity_exc')
my_parser.add_argument('I_intensity_inh',  metavar='I_intensity_inh_', type=float,  help='I_intensity_inh')

my_parser.add_argument('rate_exc',          metavar='rate_exc_',       type=float,  help='rate_exc')
my_parser.add_argument('rate_inh',          metavar='rate_inh_',       type=float,  help='rate_inh')

# EVOKED ACTIVITY
my_parser.add_argument('Ntrials',           metavar='Ntrials_',        type=int,    help='Ntrials')
my_parser.add_argument('dT_trial_sec',      metavar='dT_trial_sec_',   type=float,  help='dT_trial_sec')

my_parser.add_argument('IPI_ms',            metavar='IPI_ms_',         type=int,    help='IPI_ms')
my_parser.add_argument('n_pulses',          metavar='n_pulses_',       type=int,    help='n_pulses')

# INTERVENTIONAL CONNECTIVITY 
my_parser.add_argument('alpha_th',          metavar='alpha_th_',       type=float,  help='alpha_th')
my_parser.add_argument('Tmax',              metavar='Tmax_',           type=float,  help='Tmax')

my_parser.add_argument('Folder',            metavar='folder_',         type=str,    help='foldername')

args = my_parser.parse_args()


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #
# recording spontaneous activity: features
#
path_results = args.path_results   #'./Data/'
modules      = args.modules        #'2'
dist_rule    = args.dist_rule      #'EDR'
nNeurons     = args.nNeurons       #1000
local_II     = bool(args.local_II) #True

#================================================================================================================#
#                                            1) PATHS
#================================================================================================================#

#------------------------------------------------------------#
# folder

if local_II==True:
    folder     = f'network{nNeurons}_{modules}mod_{dist_rule}_II__tauR_{args.tau_R}__g_I_{args.g_I}__betaE_{args.beta_E}__gain_AMPA_{args.gain_AMPA}/'
    folder_net = f'network{nNeurons}_{modules}mod_{dist_rule}_II/'
else:
    folder     = f'network{nNeurons}_{modules}mod_{dist_rule}_noII__tauR_{args.tau_R}__g_I_{args.g_I}__betaE_{args.beta_E}__gain_AMPA_{args.gain_AMPA}/'
    folder_net = f'network{nNeurons}_{modules}mod_{dist_rule}_noII/'

del folder
folder = args.Folder+'/'

#------------------------------------------------------------#
# data generation paths

path_out     = path_results+'Evoked_activity/'+folder
path_data    = path_out+'Data/'
path_params  = path_data+'Parameters/'
path_imgs    = path_data+'Plots/'

# ------------------------------------------------------------#
# create folders

Set_Dir_Plots(path_results)
Set_Dir_Plots(path_results+'Evoked_activity/')
Set_Dir_Plots(path_out)
Set_Dir_Plots(path_data)
Set_Dir_Plots(path_params)
Set_Dir_Plots(path_imgs)

#------------------------------------------------------------#
# interventional connectivity

path_IC_data     = path_out+'IC_Data/'
path_ICbin_data  = path_out+'IC_bin_Data/'

# set output directories
Set_Dir_Plots(path_IC_data)
Set_Dir_Plots(path_ICbin_data)

#------------------------------------------------------------#


#================================================================================================================#
#                                                 2.1) NETWORK 
#================================================================================================================#

# generated connectivity
path_network = path_results+'Generated_network/'+folder_net+'Data/'

#------------------------------------------------------------#
# load net dictionary

with open(path_network+'net.pkl', 'rb') as file:
    net = pickle.load(file)
deltat  = net['deltat']
net['fs'] = 1000/deltat

#================================================================================================================#
#                                         2.2) PARAMETERS FOR DYNAMICS
#================================================================================================================#

# synaptic resources
net['tau_AMPA'] = args.tau_AMPA       # 3            # [ms]
net['tau_GABA'] = args.tau_GABA       # 10           # [ms]
net['tau_R']    = args.tau_R          # 500          # [ms]
net['beta_E']   = args.beta_E         # 0.1          # 0 < beta < 1
net['beta_I']   = args.beta_I         # 0.8          # 0 < beta < 1

# gains
net['gain_AMPA'] = args.gain_AMPA
net['gain_GABA'] = args.gain_GABA

net['j_AMPA'][net['ntypes']==True]  *= net['gain_AMPA']
net['j_GABA'][net['ntypes']==False] *= net['gain_GABA']

tau_AMPA, tau_GABA, tau_R, beta_E, beta_I, j_AMPA, j_GABA = iz.get_STD_parameters(net)


#----------------------------------------------------------------------------------------------------------------#
# coupling
net['g_E']             = args.g_E                   # 1.            # coupling E –> all
net['g_I']             = args.g_I                   # 1.            # coupling I –> all

# noise intensity
net['I_intensity_exc'] = args.I_intensity_exc       # 13.5          # mV      # exc current value  -  for noise
net['I_intensity_inh'] = args.I_intensity_inh       # 10.           # mV      # inh current value  -  for noise

# noise rate
net['rate_exc']        = args.rate_exc              # 1.1           # [Hz]
net['rate_inh']        = args.rate_inh              # 1             # [Hz]

#------------------------------------------------------------------------------------------------------------#
# Neurons parameters
regime             = 'RS-IB'
izhi_exc, izhi_inh = iz.set_izhikevic_parameters(regime)

#================================================================================================================#
#                                       2.3) EVOKED ACTIVITY PARAMETERS
#================================================================================================================#

# trials features
Ntrials        = args.Ntrials            # number of trials
dT_trial_sec   = args.dT_trial_sec       # trial duration (sec)
dT_trial       = dT_trial_sec*net['fs']  # trial duration (simualation steps)
Ttrial_block   = dT_trial_sec*Ntrials    # one trial block duration in (sec), which comprises all trial per one stim channel

# stimulation features:
stim_start_sec = 1                       # stimulation window onset (sec) from the begginning of each trial
IPI_ms         = args.IPI_ms             # inter pulse interval (ms)
n_pulses       = args.n_pulses           # number of pulsis per stimulation windoe
Tstim_ms       = IPI_ms*n_pulses+1       # duration of stimulation window (ms)

#---------------------------------------------------------------------------------------------------------------#
# STIMULATION CHANNELS 

def_units  = np.load(path_network + 'def_units.npy')
def_chans  = np.load(path_network + 'def_chans.npy')
indices    = np.copy(def_chans)
stim_chans = np.copy(def_chans)
print('\n n. stim. channels: ',len(def_units))

#---------------------------------------------------------------------------------------------------------------#
# simulation duration – counting the cumulative time of Ntrials per each stimulation channel
evoked_sec    = Ttrial_block*len(stim_chans)          # runtime (sec)  
runtime       = evoked_sec*net['fs']                  # runtime (simulation steps)  
net = iz.net_runtime(net, time_sec=evoked_sec, deltat=deltat)

#================================================================================================================#
#                                          3.4) PERTURBATION CURRENTS
#================================================================================================================#

# Simulation parameters
net['v_peak']      = 30.
ns, v_peak, a, b, c, d, S, D, ntypes, I_noise_evoked = iz.prepare_numba_parameters(net, izhi_exc, izhi_inh)
net['W_effective'] = (S * (j_AMPA+j_GABA)).T

# gen perturbation currents
I_noise_evoked = iz.generate_perturbations(net, I_noise_evoked, stim_chans, Ttrial_block, dT_trial_sec=dT_trial_sec, DeltaStim_ms=IPI_ms, Tstim_ms=Tstim_ms, stim_start_sec=stim_start_sec, amp_seed=1500, nearest_neighbours=False, nearest_E=False, r_mm=0.05, K=10, n_jobs=N_JOBS)


#================================================================================================================#
#                                             3.5) DATA SAVE
#================================================================================================================#

# net
with open(path_data+'net.pkl', 'wb') as file:
    pickle.dump(net, file)
# izhi_exc
with open(path_params+'izhi_exc.pkl', 'wb') as file:
    pickle.dump(izhi_exc, file)
# izhi_inh
with open(path_params+'izhi_inh.pkl', 'wb') as file:
    pickle.dump(izhi_inh, file)

# ------------------------------------------------------------------------------------------------------------#
# save parameters

#samples
np.savetxt(path_data+'net_n_positions.txt',net['pos'])
np.savetxt(path_data+'net_n_channel.txt',net['channel'])

#parameters
parameters = np.column_stack((a, b, c, d))
np.savetxt(path_params+"model_parameters.txt", parameters, delimiter=' ')

# stim. chans
np.save(path_data + 'def_units.npy', def_units)
np.save(path_data + 'def_chans.npy', def_chans)


#================================================================================================================#
#                                   4) SIMULATE EVOKED ACTIVITY
#================================================================================================================#

import time
t_start  = time.time()
print('\nSimulating dynamics...')
firings  = iz.static_munoz_izhikevich_numba(net['runtime'], net['deltat'], ns, v_peak, a, b, c, d, S, D, ntypes, 
                                            I_noise_evoked, tau_AMPA, tau_GABA, tau_R, beta_E, beta_I, j_AMPA, j_GABA)
t_stop   = time.time()
print('elapsed time:\t',t_stop-t_start,'s')
#------------------------------------------------------------------------------------------------------------#
del ns, v_peak, a, b, c, d, S, D, ntypes, I_noise_evoked, tau_AMPA, tau_GABA, tau_R, beta_E, beta_I, j_AMPA, j_GABA

#------------------------------------------------------------------------------------------------------------#
# spike times
spikeChans, spikeTimes = np.nonzero(firings)
del firings
#np.savetxt(path_data+'spikeTimes.txt',list(zip(spikeChans, spikeTimes)))


#------------------------------------------------------------------------------------------------------------#
# rate
rates = ut._compute_firing_rate(np.vstack([ spikeTimes/net['fs'],spikeChans.astype(int) ]), np.arange(net['neurons']), t_stop=None)
np.savetxt(path_data+'rates.txt',rates)

#------------------------------------------------------------------------------------------------------------#
# PLOT: raster 
ut.rasterplot(np.vstack([spikeTimes/net['fs'],spikeChans.astype(int)]), np.arange(net['neurons']), np.arange(net['neurons']), dotsize=0.15, figsize=(25,5), tmin=0, tmax=10, cmap='viridis', outf = path_imgs + 'evoked_rasterplot.png', show_plot = show_plot)


#================================================================================================================#
#                                   5) INTERVENTIONAL CONNECTIVITY
#================================================================================================================#

#------------------------------------------------------------------------------------------------------------#
# 5.1) prepare trials

print('\nPreparing trials.....')
spikeChans = spikeChans.astype(np.int32, copy=False)
spikeTimes = spikeTimes.astype(np.int32, copy=False)
order   = np.argsort(spikeChans, kind='stable')
ch_s    = spikeChans[order]
t_s     = spikeTimes[order]

uniq_ch, counts = np.unique(ch_s, return_counts=True)
splits  = np.cumsum(counts[:-1])        #  split idxs
groups  = np.split(t_s, splits)         #  spike times array
pre_filtered_spikes = {int(ch): arr for ch, arr in zip(uniq_ch, groups)}
for ch in np.asarray(net['channel'], dtype=int):
    if ch not in pre_filtered_spikes:
        pre_filtered_spikes[ch] = np.empty(0, dtype=np.int32)

dT_trial = int(round(dT_trial_sec * net['fs']))
def_spikes = build_def_spikes(pre_filtered_spikes, net['channel'], Ntrials, dT_trial, n_units=len(def_units), include_right_boundary=True)
print('...done\n')

import pickle
with open(path_data+'def_spikes.pkl', 'wb') as file:
    pickle.dump(def_spikes, file)

#------------------------------------------------------------------------------------------------------------#
# 5.2) IC parameters

# pre– and post–stimulation time windows
Tmax_ms    = args.Tmax                           
Tmax       = int(round(Tmax_ms / net['deltat']))

Delta_post = -1
Delta_pre  = 5

# binning size for spike count
dT_ms   = 50
dT      = int(round(dT_ms / net['deltat']))
Nbins   = max(1, Tmax // dT)


# stim. window-length
stim_start = int(round(1.0 * net['fs']))                   # onset of the stimulus
Tstim_steps = int(round(Tstim_ms / net['deltat']))         # end of the stimulus
stim_stop  = stim_start + Tstim_steps

# P-value threshold for EC and IC
alpha_th   = args.alpha_th #0.05

# Kolmogorov-Smirnov on spike counts per trial
t_start  = time.time()
ks, ks_sign, ks_pval_FDR, ks_pval  = IC.compute_KS(Ntrials, def_units, def_chans, def_spikes, net['channel'], stim_start, Delta_pre, stim_stop, Delta_post, Tmax, alpha_th=alpha_th, id_trial_start=0, n_jobs=N_JOBS, verbose=True)
IC.ic_save(path_IC_data, ks, ks_pval, ks_pval_FDR)
t_stop   = time.time()
print('elapsed time:\t',t_stop-t_start,'s')

# Kolmogorov-Smirnov on BINNED spike counts per trial
t_start  = time.time()
ks_bin, ks_sign_bin, ks_pval_FDR_bin, ks_pval_bin  = IC.compute_KS_binned(Ntrials, def_units, def_chans, def_spikes, net['channel'], stim_start, Delta_pre, stim_stop, Delta_post, Tmax, Nbins=Nbins, alpha_th=alpha_th, id_trial_start=0, n_jobs=N_JOBS, verbose=True)
IC.ic_save(path_ICbin_data, ks_bin, ks_pval_bin, ks_pval_FDR_bin)
t_stop   = time.time()
print('elapsed time:\t',t_stop-t_start,'s\n')

#------------------------------------------------------------------------------------------------------------#
# PLOT: IC vs. Structure

print('PLOT: IC vs. Structure')

fig, axs = plt.subplots(1,2,figsize=(25,7))

mat = np.copy(ks_sign)
pl.plot_mat_aspect(mat, vmin=0,vmax=1, ylabel='target channel', xlabel='source channel', cbarlabel='IC', 
                   title='interventional connectivity', cmap='cool', ax=axs[0], yticklabels=indices)

W_effective = net['W_effective']
mat = (W_effective[indices,:]/np.max(np.abs(W_effective.flatten()))); extr = np.max(np.abs(mat).flatten())
pl.plot_mat_aspect(mat, vmin=-1,vmax=1, ylabel='ground truth', xlabel='source channel', cbarlabel='weight', 
                   title='ground truth', cmap=coldhot_cmap, ax=axs[1], yticklabels=indices)

fig.subplots_adjust(wspace=0.25, hspace=0.5)
plt.savefig(path_imgs + '2_IC_Weff_matrices.png', bbox_inches='tight')
#------------------------------------------------------------------------------------------------------------#

#------------------------------------------------------------------------------------------------------------#
# PLOT: IC vs. Structure

print('PLOT: IC vs. Structure')

fig, axs = plt.subplots(1,2,figsize=(25,7))

mat = np.copy(ks_sign_bin)
pl.plot_mat_aspect(mat, vmin=0,vmax=1, ylabel='target channel', xlabel='source channel', cbarlabel='IC', 
                   title='interventional connectivity', cmap='cool', ax=axs[0], yticklabels=indices)

W_effective = net['W_effective']
mat = (W_effective[indices,:]/np.max(np.abs(W_effective.flatten()))); extr = np.max(np.abs(mat).flatten())
pl.plot_mat_aspect(mat, vmin=-1,vmax=1, ylabel='ground truth', xlabel='source channel', cbarlabel='weight', 
                   title='ground truth', cmap=coldhot_cmap, ax=axs[1], yticklabels=indices)

fig.subplots_adjust(wspace=0.25, hspace=0.5)
plt.savefig(path_imgs + '3_ICbin_Weff_matrices.png', bbox_inches='tight')
#------------------------------------------------------------------------------------------------------------#


#------------------------------------------------------------------------------------------------------------#
# PLOT: perturbome vs. structure

print('PLOT: perturbome vs. structure')

list_idxs = stim_chans
for idx in list_idxs:
    i = np.where(indices==idx)[0][0]

    fig,axs=plt.subplots(1,2,figsize=(40,10))
    pl.plot_perturbome(ks_sign, net['pos'], net['channel'], indices, label='IC', stim_id=i, vmin=0, vmax=1, cmap='cool',
                   ax = axs[0], DIM=40, dotsize=82, starsize=2800)
    pl.plot_perturbome((W_effective[indices,:]/np.max(np.abs(W_effective.flatten()))), net['pos'], net['channel'], indices, 
                       label='ground truth', stim_id=i, vmin=-0.05, vmax=0.05, 
                       cmap=coldhot_cmap, ax = axs[1], DIM=40, dotsize=82, starsize=2800, log=False)
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(path_imgs + f'6_perturbomes_{indices[i]}.png', bbox_inches='tight')
    if show_plot==False:
        plt.close()
#------------------------------------------------------------------------------------------------------------#

#------------------------------------------------------------------------------------------------------------#
# PLOT: perturbome vs. structure

print('PLOT: perturbome vs. structure')

list_idxs = stim_chans
for idx in list_idxs:
    i = np.where(indices==idx)[0][0]

    fig,axs=plt.subplots(1,2,figsize=(40,10))
    pl.plot_perturbome(ks_sign_bin, net['pos'], net['channel'], indices, label='IC', stim_id=i, vmin=0, vmax=0.3, cmap='cool',
                   ax = axs[0], DIM=40, dotsize=82, starsize=2800)
    pl.plot_perturbome((W_effective[indices,:]/np.max(np.abs(W_effective.flatten()))), net['pos'], net['channel'], indices, 
                       label='ground truth', stim_id=i, vmin=-0.05, vmax=0.05, 
                       cmap=coldhot_cmap, ax = axs[1], DIM=40, dotsize=82, starsize=2800, log=False)
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(path_imgs + f'7_perturbomes_bin_{indices[i]}.png', bbox_inches='tight')
    if show_plot==False:
        plt.close()
#------------------------------------------------------------------------------------------------------------#



#================================================================================================================#
#                          7) Probability of connection as a function of distance
#================================================================================================================#

dist_mat = net['dist_matrix']
Dmat     = dist_mat[indices,:]
W        = net['W_effective'][indices,:]

#------------------------------------------------------------------------------------------------------------#
# CHARACTERISTIC LENGTH OF DECAY FOR P(conn|distance) – WHOLE MATS
print('\n\nCHARACTERISTIC LENGTH OF DECAY FOR P(conn|distance) – WHOLE MATS')

def func(x, a, b, c): return a * np.exp(-b * x) + c

fig,axs=plt.subplots(1,2,figsize=(14,4)); 
Pn_cond_d, dist_conn1 = di.distance_probabilities(ks_sign, Dmat, which=1, N_bins=25, edges=None, density=False)
cmap=truncate_cmap(plt.get_cmap('Greens'), minval=0.2, maxval=1)
lambda_mm, P0, popt = di.fit_decay_length(dist_conn1, Pn_cond_d, plot=True, color='tab:green', ylabel='P(IC$_{conn}$|dist)', label_meas = 'P', label_curve = 'exp. fit', xmax=4, cmap=cmap, ax=axs[0])
np.save(path_IC_data+'lambda_wholemat_ks_sign.npy',    lambda_mm)
axs[0].plot([lambda_mm,lambda_mm],[0,func(lambda_mm, *popt)],'--g',lw=3)

Pn_cond_d, dist_conn1 = di.distance_probabilities(W, Dmat, which=1, N_bins=25, edges=None, density=False)
cmap=truncate_cmap(plt.get_cmap('Blues'), minval=0.2, maxval=1)
lambda_mm, P0, popt = di.fit_decay_length(dist_conn1, Pn_cond_d, plot=True, color='tab:green', ylabel='P(W$_{conn}$|dist)', label_meas = 'P', label_curve = 'exp. fit', xmax=4, cmap=cmap, ax=axs[1])
np.save(path_IC_data+'lambda_wholemat_W.npy',          lambda_mm)
axs[1].plot([lambda_mm,lambda_mm],[0,func(lambda_mm, *popt)],'--b',lw=3)
fig.subplots_adjust(wspace=0.5, hspace=0.5)

plt.savefig(path_imgs + f'7.1_Pn_cond_d__and__decayLength_IC_W.png', bbox_inches='tight')
plt.close()

#------------------------------------------------------------------------------------------------------------#


#------------------------------------------------------------------------------------------------------------#
# CHARACTERISTIC LENGTH OF DECAY FOR P(conn|distance) – FIXED SOURCE
print('CHARACTERISTIC LENGTH OF DECAY FOR P(conn|distance) – FIXED SOURCE')

def func(x, a, b, c): return a * np.exp(-b * x) + c
cmap=truncate_cmap(plt.get_cmap('Greens'), minval=0.2, maxval=1)
cmap2=truncate_cmap(plt.get_cmap('Blues'), minval=0.2, maxval=1)

lambda_ks_sign_bin = np.zeros(len(indices))
lambda_ks_sign     = np.zeros(len(indices))
lambda_W           = np.zeros(len(indices))

P0_ks_sign_bin = np.zeros(len(indices))
P0_ks_sign     = np.zeros(len(indices))
P0_W           = np.zeros(len(indices))

sigOverBackg_ks_sign_bin = np.zeros(len(indices))
sigOverBackg_ks_sign     = np.zeros(len(indices))
sigOverBackg_W           = np.zeros(len(indices))


#------------------------------------------------------------------------------------------------------------#
N_bins = 30
edges_global = np.linspace(0.0, 3.5, N_bins + 1)

list_idxs = indices
for idx in list_idxs:
    i = np.where(indices==idx)[0][0]
    
    fig,axs=plt.subplots(1,2,figsize=(14,4)); 

    Pn_cond_d, dist_conn1 = di.distance_probabilities(ks_sign_bin[i,:], Dmat[i,:], which=1, N_bins=N_bins, edges=edges_global, density=False)
    lambda_mm, P0_ks_sign_bin[i], popt = di.fit_decay_length(dist_conn1, Pn_cond_d, plot=False)
    lambda_ks_sign_bin[i] = lambda_mm
    sigOverBackg_ks_sign_bin[i] = popt[0]

    Pn_cond_d, dist_conn1 = di.distance_probabilities(ks_sign[i,:], Dmat[i,:], which=1, N_bins=N_bins, edges=edges_global, density=False)
    lambda_mm, P0_ks_sign[i], popt = di.fit_decay_length(dist_conn1, Pn_cond_d, plot=True, color='tab:green', ylabel='P(IC$_{conn}$|dist)', label_meas = 'P', label_curve = 'exp. fit', xmax=4, cmap=cmap, ax=axs[0])
    lambda_ks_sign[i] = lambda_mm
    sigOverBackg_ks_sign[i] = popt[0]
    axs[0].plot([lambda_mm,lambda_mm],[0,func(lambda_mm, *popt)],'--g',lw=3)
    axs[0].set_title(f'stim. chan. {indices[i]}', y=1.6)
    
    Pn_cond_d, dist_conn1 = di.distance_probabilities(W[i,:], Dmat[i,:], which=1, N_bins=N_bins, edges=edges_global, density=False)
    lambda_mm, P0_W[i], popt = di.fit_decay_length(dist_conn1, Pn_cond_d, plot=True, color='tab:green', ylabel='P(W$_{conn}$|dist)', label_meas = 'P', label_curve = 'exp. fit', xmax=4, cmap=cmap2, ax=axs[1])
    lambda_W[i] = lambda_mm
    sigOverBackg_W[i] = popt[0]
    axs[1].plot([lambda_mm,lambda_mm],[0,func(lambda_mm, *popt)],'--b',lw=3)
    axs[1].set_title(f'stim. chan. {indices[i]}', y=1.6)
    
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.savefig(path_imgs + f'7.2_Pn_cond_d__and__decayLength_IC_W_{indices[i]}.png', bbox_inches='tight')
    plt.close()

#------------------------------------------------------------------------------------------------------------#

np.save(path_IC_data+'lambda_ks_sign_bin.npy',lambda_ks_sign_bin)
np.save(path_IC_data+'lambda_ks_sign.npy',    lambda_ks_sign)
np.save(path_IC_data+'lambda_W.npy',          lambda_W)

#------------------------------------------------------------------------------------------------------------#
# PLOT CHAR. LENGTHS
print('PLOT CHAR. LENGTHS')

cond = net['ntypes'][indices]
fig,axs=plt.subplots(1,3,figsize=(21,4));
ax = axs[0]
ax.scatter(lambda_ks_sign[cond], lambda_W[cond], s=100); ax.set_xlabel(r'$\lambda_{IC}$'); ax.set_ylabel(r'$\lambda_{W}$'); ax.set_title('characteristic length')
ax = axs[1]
ax.scatter(P0_ks_sign[cond], P0_W[cond], s=100); ax.set_xlabel(r'$P_0$$_{IC}$'); ax.set_ylabel(r'$P_0$$_{W}$'); 
ax = axs[2]
ax.scatter(sigOverBackg_ks_sign[cond], sigOverBackg_W[cond], s=100); ax.set_xlabel(r'$a_{IC}$'); ax.set_ylabel(r'$a_{W}$'); ax.set_title('signal-over-background')
for ax in axs:
    pl.set_format(ax=ax, DIM=DIM)
fig.subplots_adjust(wspace=0.5, hspace=0.5)
plt.savefig(path_imgs + f'7.2__decayLength_IC_W.png', bbox_inches='tight')
plt.close()

fig,axs=plt.subplots(1,3,figsize=(21,4));
ax = axs[0]
ax.scatter(lambda_ks_sign_bin[cond], lambda_W[cond], s=100); ax.set_xlabel(r'$\lambda_{IC_{bin}}$'); ax.set_ylabel(r'$\lambda_{W}$'); ax.set_title('characteristic length')
ax = axs[1]
ax.scatter(P0_ks_sign_bin[cond], P0_W[cond], s=100); ax.set_xlabel(r'$P_0$$_{IC_{bin}}$'); ax.set_ylabel(r'$P_0$$_{W}$'); 
ax = axs[2]
ax.scatter(sigOverBackg_ks_sign_bin[cond], sigOverBackg_W[cond], s=100); ax.set_xlabel(r'$a_{IC_{bin}}$'); ax.set_ylabel(r'$a_{W}$'); ax.set_title('signal-over-background')
for ax in axs:
    pl.set_format(ax=ax, DIM=DIM)
fig.subplots_adjust(wspace=0.5, hspace=0.5)
plt.savefig(path_imgs + f'7.2__decayLength_ICbin_W.png', bbox_inches='tight')
plt.close()

#================================================================================================================#
#                                        8) SPATIAL FOOTPRINT
#================================================================================================================#

dist_mat = net['dist_matrix']
Dmat     = dist_mat[indices,:]
W        = net['W_effective'][indices,:]

#------------------------------------------------------------------------------------------------------------#
# MEAN IC AND W AS A FUNCTION OF DISTANCE – WHOLE MATRICES
print('MEAN IC AND W AS A FUNCTION OF DISTANCE – WHOLE MATRICES')

fig,axs=plt.subplots(1,2,figsize=(9,3))
pl.plot_binned_mean(ks_sign,     Dmat, xlabel='eucl. dist.(mm)', ylabel='IC', color='limegreen', N_bins=25, xmax=4, ax=axs[0])
pl.plot_binned_mean(W, Dmat, xlabel='eucl. dist.(mm)', ylabel='structure',   color=colorz[0], N_bins=25, xmax=4, ax=axs[1])
fig.subplots_adjust(wspace=0.5, hspace=0.5)
plt.savefig(path_imgs + f'8.1_Cmean_distance_wholeMat.png', bbox_inches='tight')

#------------------------------------------------------------------------------------------------------------#
# MEAN IC AND W AS A FUNCTION OF DISTANCE – FOR FIXED STIM. CHANNEL
print('MEAN IC AND W AS A FUNCTION OF DISTANCE – FOR FIXED STIM. CHANNEL')

list_idxs = indices
for idx in list_idxs:
    i = np.where(indices==idx)[0][0] # 601 for culture 2

    fig,axs=plt.subplots(1,2,figsize=(9,3))
    pl.plot_binned_mean(ks[i],   Dmat[i],  N_bins=20, xlabel='eucl. dist.(mm)', ylabel='IC',   color='limegreen', xmax=4, ax=axs[0])
    pl.plot_binned_mean(W[i],   Dmat[i],  N_bins=20, xlabel='eucl. dist.(mm)', ylabel='structure',   color=colorz[0],   xmax=4, ax=axs[1])
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    fig.suptitle(f'stim. chan. {indices[i]}',y=1.1)
    plt.savefig(path_imgs + f'8.2_Cmean_distance_source_{indices[i]}.png', bbox_inches='tight')
#------------------------------------------------------------------------------------------------------------#

#================================================================================================================#
#                                        9) RASTER PERTURBATIONS
#================================================================================================================#

#------------------------------------------------------------------------------------------------------------#
# RASTER PLOT OF ALL TRIALS FOR SELECTED COUPLES SOURCE-TARGET
print('RASTER PLOT OF ALL TRIALS FOR SELECTED COUPLES SOURCE-TARGET')

t_min=0.; t_max=2  # start stop trial (sec)

# connectivity matrix
Weff=net['W_effective']

# stimulated neuron/channel
c_unit   = def_units[0]; c_source = def_chans[0]
nonzero=np.where(Weff[c_source,:]!=0)[0]

for i in (nonzero[:20]):
    # target neuron/channel
    c_target=net['channel'][i]
    
    plt.subplots(figsize=(15,4))
    for n_trial in range(Ntrials):
        # stimulation start and stop
        plt.plot([stim_start/net['fs'],stim_start/net['fs']],[0,Ntrials], color='tab:red', lw=0.5)
        plt.plot([stim_stop/net['fs'], stim_stop/net['fs']],  [0,Ntrials], color='tab:red', lw=0.5)
        # scatter
        time = def_spikes[c_unit][n_trial][c_target]/net['fs']
        x = time[(time>=t_min) & (time<t_max)]
        y = np.ones(len(x))*(n_trial+1)
        plt.scatter(x,y,s=0.5,c='k')
        plt.xlabel('time (s)')
        plt.ylabel('trial')
        plt.title(f'{np.round(Weff[c_unit,c_target],3)}')
    plt.savefig(path_imgs + f'19.rasterTrials__{c_source}_{c_target}.png', bbox_inches='tight')
    plt.close()
#------------------------------------------------------------------------------------------------------------#



#------------------------------------------------------------------------------------------------------------#
# SCRIPT ELAPSED TIME
print('\n\nDONE')
#------------------------------------------------------------------------------------------------------------#






















