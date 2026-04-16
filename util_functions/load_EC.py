import numpy as np
import scipy.io

# functions to load EC matrices

# ================================================================================================================ #
# Sets to zero the diagonal of the matrix

def dediag(mat):
    return mat-np.diag(np.diag(mat))


# ================================================================================================================ #
# Loads matrix related to EC (pk or CI estimate): it can be the EC matrix, the matrix of p-values, of counts (rela-
#                                                 ted to p-value), of delays, etc. (see the list below)
#

def load_mat(main_path, sim_folder, binsize=0.3, file_prefix='Cult_', folder_prefix='TECN_TE', ECmeas='TE', suffix='pk', mat_type='peakEC'):
    
    '''
         suffix EC       suffix CI       mat_type
      –––––––––––––––––––––––––––––––––––––––––––––––
         -              _CI              ci
        Pk                -              peakEC

        pk_Pcount       _CI_Pcount       count
        pk_Pval         _CI_Pval         P
        pk_sum          _CI_sum          sum
        pk_sum_sq       _CI_sum_sq       sum_sq
        pk_Zscored      _CI_Zscored      Zscored_EC
      ––––––––––––––––––––––––––––––––––––––––––––––– 
      
          suffix                         mat_type
      –––––––––––––––––––––––––––––––––––––––––––––––
        _comm_delays                     delays
        _delay$num                       mat_del
        _DistanceMat                     mat_d          # only in path_results_TE
      –––––––––––––––––––––––––––––––––––––––––––––––
    
    '''
    path  = main_path+sim_folder+folder_prefix+'_binsize'+str(binsize)+'/'
    mat   = scipy.io.loadmat(path+file_prefix+ECmeas+suffix+'.mat')
    
    return dediag(mat[mat_type]);

# ================================================================================================================ #
# Load EC, EC significative (jittering test), EC zscored,  EC significative zscored
#

def load_complete_measures(main_path, sim_folder, binsize=0.3, alpha=0.001, alpha_max=0.998, file_prefix='Cult_', folder_prefix='TECN_', ECmeas = 'TE', tp = 'pk'):
    
    if tp=='pk':
        M     = load_mat(main_path, sim_folder, binsize=binsize, file_prefix=file_prefix, folder_prefix=folder_prefix+ECmeas, ECmeas=ECmeas, suffix='Pk',           mat_type='peakEC')
    else:
        M     = load_mat(main_path, sim_folder, binsize=binsize, file_prefix=file_prefix, folder_prefix=folder_prefix+ECmeas, ECmeas=ECmeas, suffix=tp,             mat_type='ci')    
    M[np.isnan(M)] = 0
    M[np.isinf(M)] = 0
    
    
    M_Pval    = load_mat(main_path, sim_folder, binsize=binsize, file_prefix=file_prefix, folder_prefix=folder_prefix+ECmeas, ECmeas=ECmeas, suffix=tp+'_Pval',     mat_type='P')   
    M_Pval[np.isnan(M_Pval)] = 0
    M_Pval[np.isinf(M_Pval)] = 0
    
    M_Zscored = load_mat(main_path, sim_folder, binsize=binsize, file_prefix=file_prefix, folder_prefix=folder_prefix+ECmeas, ECmeas=ECmeas, suffix=tp+'_Zscored',  mat_type='Zscored_EC')
    M_Zscored[np.isnan(M_Zscored)] = 0
    M_Zscored[np.isinf(M_Zscored)] = 0
    
    M_sign                = np.zeros_like(M)
    M_sign[M_Pval<=alpha] = M[M_Pval<=alpha]
    if ECmeas != 'TE' and tp=='pk':
        M_sign[M_Pval>=alpha_max] = M[M_Pval>=alpha_max]
    
    M_Zscored_sign                = np.zeros_like(M)
    M_Zscored_sign[M_Pval<=alpha] = M_Zscored[M_Pval<=alpha]
    if ECmeas != 'TE' and tp=='pk':
        M_Zscored_sign[M_Pval>=alpha_max] = M_Zscored[M_Pval>=alpha_max]
    
    M_Zscored_sign[np.isnan(M_Zscored_sign)] = 0
    M_Zscored_sign[np.isinf(M_Zscored_sign)] = 0
    
    return M, M_sign, M_Zscored, M_Zscored_sign, M_Pval

# ================================================================================================================ #

def load_matrices(main_path, sim_folder, binsize=0.3, folder_prefix='ECdata_', file_prefix='Cult_'+str(30)+'min_'):

    TE      = load_mat(main_path, sim_folder, binsize=binsize, file_prefix=file_prefix, 
                     folder_prefix=folder_prefix+'TE', ECmeas = 'TE', suffix='Pk', mat_type='peakEC')
    TE[np.isnan(TE)] = 0; TE[np.isinf(TE)] = 0
    TEci    = load_mat(main_path, sim_folder, binsize=binsize, file_prefix=file_prefix, 
                     folder_prefix=folder_prefix+'TE', ECmeas = 'TE', suffix='_CI', mat_type='ci')
    TEci[np.isnan(TEci)] = 0; TEci[np.isinf(TEci)] = 0
    SC      = load_mat(main_path, sim_folder, binsize=binsize, file_prefix=file_prefix, 
                     folder_prefix=folder_prefix+'SC', ECmeas = 'SC', suffix='Pk', mat_type='peakEC')
    SC[np.isnan(SC)] = 0; SC[np.isinf(SC)] = 0
    SCci    = load_mat(main_path, sim_folder, binsize=binsize, file_prefix=file_prefix, 
                     folder_prefix=folder_prefix+'SC', ECmeas = 'SC', suffix='_CI', mat_type='ci')
    SCci[np.isnan(SCci)] = 0; SCci[np.isinf(SCci)] = 0

    XCov    = load_mat(main_path, sim_folder, binsize=binsize, file_prefix=file_prefix, 
                     folder_prefix=folder_prefix+'XCov', ECmeas = 'XCov', suffix='Pk', mat_type='peakEC')
    XCov[np.isnan(XCov)] = 0; XCov[np.isinf(XCov)] = 0
    XCovci  = load_mat(main_path, sim_folder, binsize=binsize, file_prefix=file_prefix, 
                     folder_prefix=folder_prefix+'XCov', ECmeas = 'XCov', suffix='_CI', mat_type='ci')
    XCovci[np.isnan(XCovci)] = 0; XCovci[np.isinf(XCovci)] = 0

    # No Significant
    EC_mat  = [TE, TEci, np.abs(SC), np.abs(SCci), np.abs(XCov), np.abs(XCovci)]
    EC_lab  = ['TE$_{pk}$','TE$_{ci}$','SC$_{pk}$','SC$_{ci}$',
                  'XCov$_{pk}$','XCov$_{ci}$']
    EC_save = ['TEpk','TEci','SCpk','SCci','XCovpk','XCovco']
    
    return EC_mat, EC_lab, EC_save

# ================================================================================================================ #
