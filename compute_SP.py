import argparse
import numpy as np

import os

from utils import load_EC as lec
from utils import load_data as ld
from utils import network as nt



#=================================================================================================#
#                                      F U N C T I O N S

def Set_Dir_Plots(path):
    if not os.path.exists(path):
        os.mkdir(path)

#=================================================================================================#
#                                       A R G P A R S E
#

my_parser = argparse.ArgumentParser(description='Arguments to pass')

my_parser.add_argument('Main_path',  metavar='main_path_',  type=str,   help='Main path folder containing Results folder')
my_parser.add_argument('Sim_folder', metavar='sim_folder_', type=str,   help='Results storage folder')

my_parser.add_argument('Meas',       metavar='Meas_',       type=str,   help='Which EC measure or dist')
my_parser.add_argument('Type_meas',  metavar='Type_meas_',  type=str,   help='CI, pk')

my_parser.add_argument('Tlen',       metavar='Tlen_',       type=int,   help='TW duration (minutes)')
my_parser.add_argument('Bin_size',   metavar='bin_size_',   type=float, help='Bin size data (ms)')

my_parser.add_argument('Alpha_th',   metavar='alpha_th_',   type=float, help='p-value alpha for EC')

args = my_parser.parse_args()


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #
# recording spontaneous activity: features
#

DeltaT    = args.Tlen       #minutes
binsize   = args.Bin_size  #ms

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #
# EC features
#

meas      = args.Meas      # if TE, SC, XCov or dist
type_meas = args.Type_meas # pk or CI (when meas is not dist)
alpha_th  = args.Alpha_th  # p-value alpha for EC


#=================================================================================================#
#                                         P A T H S                                               #

main_path  = args.Main_path
sim_folder = args.Sim_folder+'/'

path_IC_data = main_path+sim_folder+'IC_Data/'
Set_Dir_Plots(path_IC_data)


#=================================================================================================#
#                 R E C O R D I N G   M A P   F E A T U R E S   L O A D I N G                     #

# loading channel IDs
_,channel,_,_,_,_,_,_ = ld.load_original_data(main_path,sim_folder+'Data/')
channel = channel - 1     # channels now start from 0 (handy to read arrays from file)


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #
# Each connectivity measure is a squared matrix (Num_all_channels X Num_all_channels), 
# We compute the shortest paths using just the stimulating channels as source.

path_IC = main_path+sim_folder+'electrodes/'
def_chans = np.load(path_IC+'def_chans.npy')

indices = np.zeros(len(def_chans),dtype=int)
for i in range(len(def_chans)):
    indices[i] = int(np.where(channel==def_chans[i])[0][0])

# EC tresholding:
# excluding the left cue of the weights distribution for each EC measure, as they seem to be related 
# to too low probability to be real EC links (and seems that sign. test didn't detected them).
#TE_th   = 10**-7
#SC_th   =(10**-0.35)
#XCov_th =(10**-7.1)

#=================================================================================================#
#                   L O A D I N G   E F F E T I V E   C O N N E C T I V I T Y                     #

print('\nLoading EC (non-significant connections are set to zero) ....')

alpha=0.001

if type_meas == 'pk':

    if meas == 'TE':
        _, mat, _, z_mat, _ = lec.load_complete_measures(main_path, sim_folder, folder_prefix='ECdata_',
                                                         file_prefix='Cult_'+str(DeltaT)+'min_', 
                                                         ECmeas = 'TE',  tp = 'pk', alpha=alpha_th)
        #idxs = np.where(mat<TE_th)
        #mat[idxs]=0; z_mat[idxs]=0;
        meas_label=['TE_sign','zTE_sign']
        
    elif meas == 'SC':
        _, mat, _, z_mat, _  = lec.load_complete_measures(main_path, sim_folder, folder_prefix='ECdata_',
                                                          file_prefix='Cult_'+str(DeltaT)+'min_', 
                                                          ECmeas = 'SC',  tp = 'pk', alpha=alpha_th)
        #idxs = np.where(mat<SC_th)
        #mat[idxs]=0; z_mat[idxs]=0;
        meas_label=['SC_sign','zSC_sign']
        
    else:
        _, mat, _, z_mat, _ = lec.load_complete_measures(main_path, sim_folder, folder_prefix='ECdata_',
                                                         file_prefix='Cult_'+str(DeltaT)+'min_', 
                                                         ECmeas = 'XCov',tp = 'pk', alpha=alpha_th)
        #idxs = np.where(mat<XCov_th)
        #mat[idxs]=0; z_mat[idxs]=0;
        meas_label=['XCov_sign','zXCov_sign']
        
        
elif type_meas == 'ci':
    
    if meas == 'TE':
        _, mat, _, z_mat, _ = lec.load_complete_measures(main_path, sim_folder, folder_prefix='ECdata_',
                                                         file_prefix='Cult_'+str(DeltaT)+'min_', 
                                                         ECmeas = 'TE',  tp = '_CI', alpha=alpha_th)
        meas_label = ['TEci_sign','zTEci_sign']
        
    elif meas == 'SC':
        _, mat, _, z_mat, _ = lec.load_complete_measures(main_path, sim_folder, folder_prefix='ECdata_',
                                                         file_prefix='Cult_'+str(DeltaT)+'min_', 
                                                         ECmeas = 'SC',  tp = '_CI', alpha=alpha_th)
        meas_label = ['SCci_sign','zSCci_sign']
        
    else:
        _, mat, _, z_mat, _ = lec.load_complete_measures(main_path, sim_folder, folder_prefix='ECdata_',
                                                         file_prefix='Cult_'+str(DeltaT)+'min_', 
                                                         ECmeas = 'XCov',tp = '_CI', alpha=alpha_th)
        meas_label = ['XCovci_sign','zXCovci_sign']
        
        
else: 
    print(f"[ERROR] Measurement type '{type_meas}' not recognized. Exiting.")
    sys.exit(1)

print('\n.......loading done')


#=================================================================================================#
#                        C O M P U T I N G   S H O R T E S T   P A T H S                          #
#                               for EC matrix and z-scored EC matrix                              #

SP = nt.find_SP(np.abs(mat),  indices, dist_mat=False, outf=path_IC_data+'SP_'+meas_label[0]+'.npy', 
                n_jobs=-1, verbose=True)
SP = nt.find_SP(np.abs(z_mat),indices, dist_mat=False, outf=path_IC_data+'SP_'+meas_label[1]+'.npy', 
                n_jobs=-1, verbose=True)

#=================================================================================================#









