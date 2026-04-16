import argparse
#------------------------------------------------------------#
import numpy as np
import pickle
from matplotlib import pyplot as plt
#------------------------------------------------------------#
from utils import colormaps as maps
from utils import load_data as ld
from utils import plot as pl
from utils import network as nt
#------------------------------------------------------------#
from utils_izhi import topology as tp
from utils_izhi import izhikevic as iz
#------------------------------------------------------------#
# snn
from utils_snn import channels as ut
#------------------------------------------------------------#
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
#------------------------------------------------------------#


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
DIM = 20
plt.rcParams.update({'font.size': DIM,'axes.labelsize': DIM,'axes.titlesize': DIM,'xtick.labelsize': DIM-6,'ytick.labelsize': DIM-6,
                   'legend.fontsize' : DIM-5})
print('fontsize:',plt.rcParams['font.size'],'\ntitlesize:', plt.rcParams['axes.titlesize'], '\nlabelsize:',plt.rcParams['xtick.labelsize'], plt.rcParams['ytick.labelsize'])
#=================================================================================================#
# Colors
colorz    = ['#255D93','#5FA6D6','#B02106','#F24D33','#2C2C2C','#787878']
coldhot_cmap   = maps.create_cmaphot()
coldhot_cmap_r = maps.create_cmaphot_r()
#------------------------------------------------------------#
show_plot   = False
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
# IBI stability
def compute_burst_stability(firings, deltat):
    """
    Compute stability of 'bursty' activity based on the regularity of IBIs.
    
    Args:
        firings (np.ndarray): boolean spikes [N_neurons, N_steps]
        deltat (float):       Simulation step (ms)
    
    Returns:
        tuple: (mean_ibi, cv_ibi)
    """
    from scipy.signal import find_peaks
    
    # 1. global spike rate
    pop_activity = np.sum(firings, axis=0)
    
    # 2. Individua i picchi (burst) nell'attività di popolazione
    # La soglia `prominence` identifica i picchi che si distinguono dal rumore di fondo.
    # Puoi regolare questo valore in base all'attività della tua rete.
    burst_indices, _ = find_peaks(pop_activity, prominence=np.mean(pop_activity) * 1.5)
    
    if len(burst_indices) < 3:
        # Non ci sono abbastanza burst per calcolare un CV significativo
        return np.nan, np.nan
        
    # 3. Calcola gli intervalli tra i burst (IBI)
    # L'IBI è la distanza (in passi) tra i picchi consecutivi.
    ibi_steps = np.diff(burst_indices)
    
    # 4. Converti gli intervalli in millisecondi
    ibi_ms = ibi_steps * deltat
    
    # 5. Calcola la media e il Coefficiente di Variazione (CV) dell'IBI
    mean_ibi = np.mean(ibi_ms)
    cv_ibi = np.std(ibi_ms) / mean_ibi
    
    return mean_ibi, cv_ibi
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

# DURATION 
my_parser.add_argument('time_sec',          metavar='time_sec_',       type=float,  help='time_sec')

# COUPLING
my_parser.add_argument('g_E',               metavar='g_E_',            type=float,  help='g_E')
my_parser.add_argument('g_I',               metavar='g_I_',            type=float,  help='g_I')

# NOISE
my_parser.add_argument('I_intensity_exc',  metavar='I_intensity_exc_', type=float,  help='I_intensity_exc')
my_parser.add_argument('I_intensity_inh',  metavar='I_intensity_inh_', type=float,  help='I_intensity_inh')

my_parser.add_argument('rate_exc',          metavar='rate_exc_',       type=float,  help='rate_exc')
my_parser.add_argument('rate_inh',          metavar='rate_inh_',       type=float,  help='rate_inh')

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
#                                            1.1) PATHS
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

path_out     = path_results+'Spontaneous_activity/'+folder
path_data    = path_out+'Data/'
path_params  = path_data+'Parameters/'
path_imgs    = path_data+'Plots/'

#------------------------------------------------------------#
# create folders

Set_Dir_Plots(path_results)
Set_Dir_Plots(path_results+'Spontaneous_activity/')
Set_Dir_Plots(path_out)
Set_Dir_Plots(path_data)
Set_Dir_Plots(path_params)
Set_Dir_Plots(path_imgs)

#================================================================================================================#
#                                            1.2) NETWORK PARAMETERS
#================================================================================================================#

# generated connectivity
path_network = path_results+'Generated_network/'+folder_net+'Data/'

#------------------------------------------------------------#
# load net dictionary

with open(path_network+'net.pkl', 'rb') as file:
    net = pickle.load(file)
deltat  = net['deltat']

#----------------------------------------------------------------------------------------------------------------#
# save additional features (optional)
np.savetxt(path_data+'net_n_positions.txt',net['pos'])
np.savetxt(path_data+'net_n_channel.txt',net['channel'])

print(path_data)
print('\nNetwork positions saved\n')
print('======================================================================================================')
print('                                           SPONTANEOUS ACTIVITY')
print('======================================================================================================\n\n')

    

#================================================================================================================#
#                                         1.2) PARAMETERS FOR DYNAMICS
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
# simulation duration
time_sec               = args.time_sec              # 1000          # sec     # runtime in seconds
net = iz.net_runtime(net, time_sec=time_sec, deltat=deltat)

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

# Simulation parameters
net['v_peak']      = 30.
ns, v_peak, a, b, c, d, S, D, ntypes, I_noise_all = iz.prepare_numba_parameters(net, izhi_exc, izhi_inh)
net['W_effective'] = (S * (j_AMPA+j_GABA)).T

#================================================================================================================#
#                                                DATA SAVE
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

#------------------------------------------------------------------------------------------------------------#
# save parameters

#samples
np.savetxt(path_data+'net_n_positions.txt',net['pos'])
np.savetxt(path_data+'net_n_channel.txt',net['channel'])

#parameters
parameters = np.column_stack((a, b, c, d))
np.savetxt(path_params+"model_parameters.txt", parameters, delimiter=' ')


#================================================================================================================#
#                                  2) SIMULATE SPONTANEOUS ACTIVITY
#================================================================================================================#

import time
t_start  = time.time()
firings  = iz.static_munoz_izhikevich_numba(net['runtime'], net['deltat'], ns, v_peak, a, b, c, d, S, D, ntypes, I_noise_all, tau_AMPA, tau_GABA, tau_R, beta_E, beta_I, j_AMPA, j_GABA)
t_stop   = time.time()
print('elapsed time:\t',t_stop-t_start,'s')
#------------------------------------------------------------------------------------------------------------#
del ns, v_peak, a, b, c, d, S, D, ntypes, I_noise_all, tau_AMPA, tau_GABA, tau_R, beta_E, beta_I, j_AMPA, j_GABA

#------------------------------------------------------------------------------------------------------------#
# spike times
spikeChans, spikeTimes = np.nonzero(firings)
np.savetxt(path_data+'spikeTimes.txt',list(zip(spikeChans, spikeTimes)))
spikes = np.vstack([spikeTimes/net['fs'],spikeChans.astype(int)])
del spikeChans, spikeTimes
#------------------------------------------------------------------------------------------------------------#
#bin_sz = 0.01
#s_counts = SDP.compute_spike_counts(spikes, binsize=bin_sz)
#------------------------------------------------------------------------------------------------------------#

#------------------------------------------------------------------------------------------------------------#
# rate
rates = ut._compute_firing_rate(spikes, np.arange(net['neurons']), t_stop=None)
np.savetxt(path_data+'rates.txt',rates)

#------------------------------------------------------------------------------------------------------------#
# PLOT: raster 
ut.rasterplot(spikes, np.arange(net['neurons']), np.arange(net['neurons']), dotsize=0.15, figsize=(25,5), tmin=0, tmax=10, cmap='viridis', outf = path_imgs + 'spontaneous_rasterplot.png', show_plot = show_plot)
del spikes

#------------------------------------------------------------------------------------------------------------#
# PLOT: rate 
fig,ax = plt.subplots(figsize=(4,3))
h=ax.hist(rates[net['ntypes']==0], 40, [0, np.max(rates)], histtype='step', color='tab:blue',alpha = 0.4,  lw=3, label='inh')
h=ax.hist(rates[net['ntypes']==1], 40, [0, np.max(rates)], histtype='step', color='tab:red', alpha = 0.4,  lw=3, label='exc')
ax.set_xlabel('rate (Hz)')
ax.set_ylabel('count')
ax.legend(ncol=1, loc='upper center', bbox_to_anchor=(0.5, 1.45), labelspacing=0.4, handletextpad=0.8, handlelength = 1., frameon=False) 
pl.set_format(ax=ax, pwr_x_max=3,pwr_x_min=-2, pwr_y_max=2, pwr_y_min=-2)
plt.savefig(path_imgs + 'spontaneous_rates.png', bbox_inches='tight')
if show_plot==False:
    plt.close()
#------------------------------------------------------------------------------------------------------------#


#================================================================================================================#
#                                       3) population activity
#================================================================================================================#

mean_ibi, cv_ibi = compute_burst_stability(firings, deltat)
np.savetxt(path_data+'mean_cv_IBI.txt',(mean_ibi, cv_ibi))

pop_activity = np.sum(firings, axis=0).astype(np.float64)
fig,ax=plt.subplots(figsize=(20,5))
t = np.arange(0,time_sec*net['fs'],1)/net['fs']
ax.plot(t[t<10], pop_activity[t<10], color='grey' )
ax.set_xlabel('time (s)')
ax.set_ylabel('count')
ax.set_title('population activity')
pl.set_format(ax=ax, pwr_x_max=3,pwr_x_min=-2, pwr_y_max=2, pwr_y_min=-2)
plt.savefig(path_imgs + 'spontaneous_pop_activity.png', bbox_inches='tight')
if show_plot==False:
    plt.close()


if modules=='2':
    # modules pop activity
    firings_mod1 = firings[net['modules'] == 0, :]
    firings_mod2 = firings[net['modules'] == 1, :]
    
    # modules CV IBI
    mean_ibi_mod1, cv_ibi_mod1 = compute_burst_stability(firings_mod1, deltat)
    np.savetxt(path_data+'mean_cv_IBI__module0.txt',(mean_ibi_mod1, cv_ibi_mod1))

    mean_ibi_mod2, cv_ibi_mod2 = compute_burst_stability(firings_mod2, deltat)
    np.savetxt(path_data+'mean_cv_IBI__module1.txt',(mean_ibi_mod2, cv_ibi_mod2 ))

    pop_activity_mod1 = np.sum(firings_mod1, axis=0).astype(np.float64)
    pop_activity_mod2 = np.sum(firings_mod2, axis=0).astype(np.float64)
    
    fig,ax=plt.subplots(figsize=(20,5))
    t = np.arange(0,time_sec*net['fs'],1)/net['fs']
    ax.plot(t[t<10], pop_activity_mod1[t<10], color='tab:purple', label='1' )
    ax.plot(t[t<10], pop_activity_mod2[t<10], color='gold', label='2' )
    ax.set_xlabel('time (s)')
    ax.set_ylabel('count')
    ax.set_title('modules population activity')
    pl.set_format(ax=ax, pwr_x_max=3,pwr_x_min=-2, pwr_y_max=2, pwr_y_min=-2)
    ax.legend(ncol=2)
    plt.savefig(path_imgs + 'spontaneous_pop_activity_modules.png', bbox_inches='tight')
    if show_plot==False:
        plt.close()
#------------------------------------------------------------------------------------------------------------#




#================================================================================================================#
#                                       4) COMPUTE SHORTEST PATHS
#================================================================================================================#

#----------------------------------------------------------------------------------------------------------------#
# STIMULATION CHANNELS 

def_units  = np.load(path_network + 'def_units.npy')
def_chans  = np.load(path_network + 'def_chans.npy')
indices    = np.copy(def_chans)
stim_chans = np.copy(def_chans)


'''#================================================================================================================#
#                               C O M P U T I N G   S H O R T E S T   P A T H S                                  #
#                                 for abs(W) matrix and W oly exc/inh matrices                                   #

W_eff = net['W_effective'] 
SP  = nt.find_SP(np.abs(W_eff),  indices, dist_mat=False, outf=path_data+'SP_Wabs.npy', verbose=True)

mat = np.copy(net['W_effective'])
mat[net['ntypes']==False,:] = 0
SP_E = nt.find_SP(mat,  indices, dist_mat=False, outf=path_data+'SP_Wexc.npy', verbose=True)

del mat; mat = np.copy(net['W_effective'])
mat[net['ntypes']==True,:] = 0
SP_I = nt.find_SP(np.abs(mat),  indices, dist_mat=False, outf=path_data+'SP_Winh.npy', verbose=True)
#================================================================================================================#
'''

#================================================================================================================#
#                               C O M P U T I N G   S H O R T E S T   P A T H S                                  #
#                                 for abs(W) matrix and W only exc/inh matrices                                  #

W_eff = net['W_effective']

# ABS(W): weighted SP + hops (unweighted)
SP,  H  = nt.find_SP(np.abs(W_eff), indices, dist_mat=False, return_hops=True,
                     outf=path_data+'SP_Wabs.npy', verbose=True)

# EXCITATORY: keep only rows from excitatory sources (ntypes==True), weights >= 0 by construction
mat = np.copy(W_eff)
mat[net['ntypes'] == False, :] = 0.0
SP_E, H_E = nt.find_SP(mat, indices, dist_mat=False, return_hops=True,
                       outf=path_data+'SP_Wexc.npy', verbose=True)

# INHIBITORY: keep only rows from inhibitory sources (ntypes==False), take abs for costs
mat = np.copy(W_eff)
mat[net['ntypes'] == True,  :] = 0.0
SP_I, H_I = nt.find_SP(np.abs(mat), indices, dist_mat=False, return_hops=True,
                       outf=path_data+'SP_Winh.npy', verbose=True)
#================================================================================================================#




#================================================================================================================#
#                                   5) PLOTS ON CONNECTIVITY STATISTICS
#================================================================================================================#

print('\n\n5) PLOTS ON CONNECTIVITY STATISTICS\n')
#---------------------------------------------------------------------------------------------------------------#
# PLOT: CONNECTIVITY

fig,axs=plt.subplots(1,2,figsize=(9.1,3))
#===  Weights histogram   ===#

W_max = np.max(np.abs(W_eff).flatten())
ax=axs[0]
ax.hist(W_eff[W_eff>0], len(W_eff)//4, [-W_max,W_max], color='tab:red', alpha=0.4, density=False, label='exc')
ax.hist(W_eff[W_eff<0], len(W_eff)//4, [-W_max,W_max], color='tab:blue', density=False, alpha=0.4,  label='inh')
ax.set_xlabel('weight (mV)')
ax.set_ylabel('count')
ax.legend(ncol=1, loc='upper center', bbox_to_anchor=(0.5, 1.45), labelspacing=0.4, handletextpad=0.8, handlelength = 1., frameon=False) 
pl.set_format(ax=ax, pwr_x_max=2, pwr_y_max=2, pwr_y_min=-2)
#===   Adjacency matrix   ===#
ax=axs[1]
im = ax.imshow(W_eff,cmap=coldhot_cmap_r, vmin=-np.max(np.abs(W_eff.flatten()))/3, vmax=np.max(np.abs(W_eff.flatten()))/3 )
ax.invert_yaxis()
ax.set_title('connectivity')
cbar = fig.colorbar(im, ax=ax, shrink=0.5, label='weight (mV)')
plt.savefig(path_imgs + 'connectivityEFF_weights.png', bbox_inches='tight')
if show_plot==False:
    plt.close()
#---------------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------------#
# how many links
N = W_eff.shape[0]
N_possible = N*(N-1)
N_links = np.sum(W_eff!=0)
print('links fraction:\t',N_links/N_possible)

#---------------------------------------------------------------------------------------------------------------#
# in/out strength/degree
W     = net['weights']
inDeg = np.sum(W_eff!=0, axis=0); outDeg = np.sum(W_eff!=0,axis=-1)
inStr = np.sum(W_eff, axis=0);    outStr = np.sum(W_eff,axis=-1)
print('in str < 0  N:\t',np.sum(inStr<0))

#----------------------------------------------------------------------------------------------------------------#
# PLOT: degree and strength

fig,axs=plt.subplots(1,2,figsize=(8,3))
#=== degree ===
ax=axs[0]
hmax=np.max([np.max(inDeg),np.max(outDeg)])
ax.hist(inDeg,  26, [-0.1,hmax], color='lightgrey', density=True, label='k$_{in}$')
h = ax.hist(outDeg, 26, [-0.1,hmax], color='tab:blue', density=True, histtype='step', lw=3, label='k$_{out}$')
ax.set_xlabel('degree'); ax.set_ylabel('pdf')
pl.set_format(ax=ax,  pwr_x_max=3, pwr_y_max=2, pwr_y_min=-2)
ax.legend(ncol=1, loc='upper center', bbox_to_anchor=(0.5, 1.65), labelspacing=0.4, handletextpad=0.8, handlelength = 1., frameon=False) 
#=== strengths ===
ax=axs[1]
hmax=np.max([np.max(inStr),np.max(outStr)])
ax.hist(inStr,  26, [-hmax,hmax], color='lightgrey', density=True, label='in-strength')
ax.hist(outStr, 26, [-hmax,hmax], color='tab:blue', density=True, histtype='step', lw=3, label='out-strength')
ax.set_xlabel('strength'); ax.set_ylabel('pdf')
pl.set_format(ax=ax, pwr_x_max=3, pwr_y_max=2, pwr_y_min=-2)
ax.legend(ncol=1, loc='upper center', bbox_to_anchor=(0.5, 1.65), labelspacing=0.4, handletextpad=0.8, handlelength = 1., frameon=False) 
fig.subplots_adjust(hspace=0.6, wspace=0.7)
plt.savefig(path_imgs + 'connectivityEFF_strength.png', bbox_inches='tight')
if show_plot==False:
    plt.close()
#---------------------------------------------------------------------------------------------------------------#


#---------------------------------------------------------------------------------------------------------------#
# compute probability of connection

cmap = truncate_cmap(plt.get_cmap('Blues'), minval=0.2, maxval=1)
mats = [W_eff]; labs = 'W'; colz = 'tab:red'
P_conn,dist_conn,params_conn = pl.fit_measure(W_eff, net['dist_matrix'], N_bins=50, bounds=(-np.inf,np.inf))
pl.plot_fit_meas(P_conn, dist_conn, params_conn, colz, labs, xmax=4, cmap=cmap)

#------------------------------------------------------------------------------------------------------------#
# PLOT: probability of connection
fig,ax=plt.subplots()
cmap = truncate_cmap(plt.get_cmap('Blues'), minval=0.2, maxval=1)
im   = ax.imshow(net['dist_matrix'],cmap=coldhot_cmap_r, vmin=-np.max(np.abs(net['dist_matrix'].flatten())), vmax=np.max(np.abs(net['dist_matrix'].flatten())))
ax.invert_yaxis()
ax.set_title('distance')
cbar = fig.colorbar(im, ax=ax, shrink=0.5, label='distance (mm)')
plt.savefig(path_imgs + 'connectivityEFF_connection_prob.png', bbox_inches='tight')
if show_plot==False:
    plt.close()
#---------------------------------------------------------------------------------------------------#


#---------------------------------------------------------------------------------------------------#
# PLOT: weights as a function of distance

fig,axs=plt.subplots(1,2,figsize=(9,3))

#=== excitatory ===
ax=axs[0]
y = W_eff[net['ntypes']==True]; x = net['dist_matrix'][net['ntypes']==True]
pl.plot_binned_mean(y[y!=0],  x[y!=0],  xlabel='eucl. dist.(mm)', 
                    ylabel='w$_{excitatory}$', color='tab:red',  N_bins=25, xmax=3.8, ax=axs[0])
#=== inhibitory ===
ax=axs[1]
y = W_eff[net['ntypes']==False]; x = net['dist_matrix'][net['ntypes']==False]
pl.plot_binned_mean(y[y!=0],  x[y!=0], xlabel='eucl. dist.(mm)', 
                    ylabel='w$_{inhibitory}$', color='tab:blue', N_bins=25, xmax=3.8, ax=axs[1])
fig.subplots_adjust(hspace=0.6, wspace=0.8)
plt.savefig(path_imgs + 'connectivityEFF_weights_distance.png', bbox_inches='tight')
if show_plot==False:
    plt.close()
#---------------------------------------------------------------------------------------------------#


# =================================================================================================
#         6) GLOBAL DYNAMICS METRICS (global, E/I, modules)  — store in a dictionary
# =================================================================================================
print('\n\n6) GLOBAL DYNAMICS METRICS (global, E/I, modules)  — store in a dictionary\n')

from scipy.signal import find_peaks, peak_widths

dyn_metrics = {}

Emsk = net['ntypes'].astype(bool)  # True = excitatory, False = inhibitory

# --- basic rates ---
dyn_metrics['rate_all_Hz'] = float(np.mean(rates))
dyn_metrics['rate_E_Hz']   = float(np.mean(rates[Emsk]))    if np.any(Emsk)    else np.nan
dyn_metrics['rate_I_Hz']   = float(np.mean(rates[~Emsk]))   if np.any(~Emsk)   else np.nan
dyn_metrics['frac_silent_all'] = float(np.mean(rates == 0))
dyn_metrics['frac_silent_E']   = float(np.mean(rates[Emsk] == 0))   if np.any(Emsk)  else np.nan
dyn_metrics['frac_silent_I']   = float(np.mean(rates[~Emsk] == 0))  if np.any(~Emsk) else np.nan

# --- population activity and burst stats (reuse your compute_burst_stability) ---
pop_activity = np.sum(firings, axis=0).astype(np.float64)
mean_ibi_ms, cv_ibi = compute_burst_stability(firings, deltat)
dyn_metrics['IBI_mean_ms'] = float(mean_ibi_ms) if not np.isnan(mean_ibi_ms) else np.nan
dyn_metrics['IBI_CV']      = float(cv_ibi)      if not np.isnan(cv_ibi)      else np.nan

# burst duration (FWHM of peaks) and burst size (area under pop. curve around peaks)
prom = np.mean(pop_activity) * 1.5
pk_idx, _ = find_peaks(pop_activity, prominence=prom)

#---------------------------------------------------------------------------------------------------#

from scipy.signal import peak_widths

prom = np.mean(pop_activity) * 1.5
pk_idx, _ = find_peaks(pop_activity, prominence=prom)

if pk_idx.size > 0:
    # peak_widths -> widths, h_eval, left_ips, right_ips (4 valori)
    widths, _h_eval, left_ips, right_ips = peak_widths(pop_activity, pk_idx, rel_height=0.5)
    # 'widths' è in campioni → ms moltiplicando per deltat (ms/step)
    burst_dur_ms = float(np.mean(widths) * deltat)

    sizes = []
    # left_ips/right_ips sono float (interpolati): usa floor/ceil per includere tutto il picco
    for L, R in zip(left_ips, right_ips):
        Lc = max(0, int(np.floor(L)))
        Rc = min(pop_activity.size - 1, int(np.ceil(R)))
        sizes.append(float(np.sum(pop_activity[Lc:Rc+1])))

    sizes = np.asarray(sizes, dtype=np.float64)
    burst_size_mean = float(np.mean(sizes)) if sizes.size else np.nan
    burst_size_cv   = float(np.std(sizes) / np.mean(sizes)) if sizes.size and np.mean(sizes) > 0 else np.nan
else:
    burst_dur_ms     = np.nan
    burst_size_mean  = np.nan
    burst_size_cv    = np.nan

dyn_metrics['burst_duration_mean_ms'] = burst_dur_ms
dyn_metrics['burst_size_mean']        = burst_size_mean
dyn_metrics['burst_size_CV']          = burst_size_cv

# synchrony index (coefficient of variation of population activity)
dyn_metrics['synchrony_index_CVpop'] = float(np.std(pop_activity) / max(np.mean(pop_activity), 1e-12))

# dimensionality: participation ratio (10 ms bins, SVD of neuron×time matrix)
bin_w = max(1, int(round(10.0 / net['deltat'])))  # 10 ms / dt(ms)
T     = firings.shape[1]
Tb    = T // bin_w
if Tb >= 2:
    X = firings[:, :Tb*bin_w].reshape(firings.shape[0], Tb, bin_w).sum(axis=2).astype(np.float64)  # (N, Tb)
    X -= X.mean(axis=1, keepdims=True)
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    lam = (s**2) / max(Tb - 1, 1)  # eigenvalues of covariance
    pr_dim  = (lam.sum()**2) / (np.sum(lam**2) + 1e-12)
    evr_top3 = np.sum(lam[:3]) / max(lam.sum(), 1e-12)
else:
    pr_dim, evr_top3 = np.nan, np.nan

dyn_metrics['participation_ratio_dim'] = float(pr_dim)
dyn_metrics['EVR_top3']                = float(evr_top3)

# per-module dynamics (only if 2 modules)
if modules == '2':
    m0 = (net['modules'] == 0)
    m1 = (net['modules'] == 1)
    dyn_metrics['rate_mod0_Hz'] = float(np.mean(rates[m0])) if np.any(m0) else np.nan
    dyn_metrics['rate_mod1_Hz'] = float(np.mean(rates[m1])) if np.any(m1) else np.nan

    mean_ibi_m0, cv_ibi_m0 = compute_burst_stability(firings[m0, :], deltat) if np.any(m0) else (np.nan, np.nan)
    mean_ibi_m1, cv_ibi_m1 = compute_burst_stability(firings[m1, :], deltat) if np.any(m1) else (np.nan, np.nan)
    dyn_metrics['IBI_mean_mod0_ms'] = float(mean_ibi_m0) if not np.isnan(mean_ibi_m0) else np.nan
    dyn_metrics['IBI_mean_mod1_ms'] = float(mean_ibi_m1) if not np.isnan(mean_ibi_m1) else np.nan
    dyn_metrics['IBI_CV_mod0']      = float(cv_ibi_m0)  if not np.isnan(cv_ibi_m0)  else np.nan
    dyn_metrics['IBI_CV_mod1']      = float(cv_ibi_m1)  if not np.isnan(cv_ibi_m1)  else np.nan

# save dynamics metrics
with open(path_data + 'metrics_dynamics.pkl', 'wb') as f:
    pickle.dump(dyn_metrics, f)


# =================================================================================================
#        7) STRUCTURAL INDICATORS (from W_effective and geometry) — store in a dictionary
# =================================================================================================

print('7) STRUCTURAL INDICATORS (from W_effective and geometry) — store in a dictionary')

str_metrics = {}

W_eff   = net['W_effective']
edgeEff = (W_eff != 0.0)
types   = net['ntypes'].astype(bool)  # True=E, False=I

# spectral radius (|largest eigenvalue|)
rho = np.nan
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
try:
    val = eigs(csr_matrix(W_eff.astype(np.float64)), k=1, which='LM', return_eigenvectors=False)
    rho = float(np.abs(val[0]))
except Exception:
    vals = np.linalg.eigvals(W_eff.astype(np.float64))
    rho  = float(np.max(np.abs(vals)))
str_metrics['spectral_radius_abs'] = rho

# degree/strength statistics
kin   = edgeEff.sum(axis=0)
kout  = edgeEff.sum(axis=1)
sin_  = W_eff.sum(axis=0)
sout_ = W_eff.sum(axis=1)
str_metrics['deg_mean_in']   = float(np.mean(kin))
str_metrics['deg_mean_out']  = float(np.mean(kout))
str_metrics['str_mean_in']   = float(np.mean(sin_))
str_metrics['str_mean_out']  = float(np.mean(sout_))

# E/I edge category counts (TOTAL)
E_pre  = types[:, None]
I_pre  = ~E_pre
E_post = types[None, :]
I_post = ~E_post
str_metrics['edges_EtoE'] = int(np.sum(edgeEff & E_pre & E_post))
str_metrics['edges_EtoI'] = int(np.sum(edgeEff & E_pre & I_post))
str_metrics['edges_ItoE'] = int(np.sum(edgeEff & I_pre & E_post))
str_metrics['edges_ItoI'] = int(np.sum(edgeEff & I_pre & I_post))

#---------------------------------------------------------------------------------------------------#
# within/between modules (if available)
if 'modules' in net:
    M = net['modules']
    within  = (M[:, None] == M[None, :])
    between = ~within
    np.fill_diagonal(within, False)
    np.fill_diagonal(between, False)

    Wabs = np.abs(W_eff)
    # connection probability
    str_metrics['p_within']  = float(np.mean(edgeEff[within]))  if np.any(within)  else np.nan
    str_metrics['p_between'] = float(np.mean(edgeEff[between])) if np.any(between) else np.nan
    # mean absolute weight on existing edges
    w_within_vals  = Wabs[within & edgeEff]
    w_between_vals = Wabs[between & edgeEff]
    str_metrics['wabs_within']  = float(np.mean(w_within_vals))  if w_within_vals.size  else np.nan
    str_metrics['wabs_between'] = float(np.mean(w_between_vals)) if w_between_vals.size else np.nan

#---------------------------------------------------------------------------------------------------#
# short/long range split by distance
try:
    Dmat = np.asarray(net['dist_matrix'])
except KeyError:
    P = net['pos']
    Dmat = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
np.fill_diagonal(Dmat, 0.0)

r_short = 0.60  # mm
is_short = (Dmat <= r_short)
is_long  = ~is_short
str_metrics['conn_prob_short(<=0.60mm)'] = float(np.mean(edgeEff[is_short]))
str_metrics['conn_prob_long(>0.60mm)']   = float(np.mean(edgeEff[is_long]))

# save structural metrics
with open(path_data + 'metrics_structure.pkl', 'wb') as f:
    pickle.dump(str_metrics, f)

print("\n\n[metrics] saved:", path_data + "metrics_dynamics.pkl")
print("\n[metrics] saved:", path_data + "metrics_structure.pkl")

#---------------------------------------------------------------------------------------------------#
