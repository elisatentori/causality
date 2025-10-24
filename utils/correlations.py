import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgba

from scipy.stats import spearmanr, pearsonr

from . import colormaps as maps
from . import plot as pl

#=================================================================================================#
from matplotlib import font_manager, rcParams
font_file = "/home/tentori/.local/avenir_ff/AvenirLTStd-Roman.otf"
font_file_b = "/home/tentori/.local/avenir_ff/AvenirLTStd-Black.otf"
font_file_c = "/home/tentori/.local/avenir_ff/AvenirLTStd-Book.otf"
font_manager.fontManager.addfont(font_file)
font_manager.fontManager.addfont(font_file_b)
font_manager.fontManager.addfont(font_file_c)

# predef font: Avenir
rcParams['font.family'] = "Avenir LT Std"

DIM = 25

plt.rcParams.update({
    'font.size': DIM,
    'axes.labelsize': DIM,
    'axes.titlesize': DIM,
    'xtick.labelsize': DIM,
    'ytick.labelsize': DIM
})
#=================================================================================================#

coldhot_cmap   = maps.create_cmaphot()
coldhot_cmap_r = maps.create_cmaphot_r()
        
#=================================================================================================#
# Accuracy of EC in reconstructing the perturbome (IC) [for fixed stimulation/source channel] :
# correlation (pearson/spearman) betw the rows of EC matrices (stored in a list) 
# and the correspondent rows of IC

def correlate_source_fixed(meas_mat, IC_mat, exclude_zeros=False, pearson=False):
    
    correlation = np.zeros((len(meas_mat),IC_mat.shape[0]))
    p_value     = np.zeros((len(meas_mat),IC_mat.shape[0]))

    for idx_meas in range(len(meas_mat)):
        for i in range(IC_mat.shape[0]):
            
            EC_v = meas_mat[idx_meas]
            
            if pearson==False:
                if exclude_zeros:
                    condition = np.logical_and(IC_mat[i,:]!=0, EC_v[i,:]!=0)
                    correlation[idx_meas,i], p_value[idx_meas,i] = spearmanr(IC_mat[i,:][condition], EC_v[i,:][condition])            
                else:
                    correlation[idx_meas,i], p_value[idx_meas,i] = spearmanr(IC_mat[i,:], EC_v[i,:])
            else:
                if exclude_zeros:
                    condition = np.logical_and(IC_mat[i,:]!=0, EC_v[i,:]!=0)
                    correlation[idx_meas,i], p_value[idx_meas,i] = pearsonr(IC_mat[i,:][condition], EC_v[i,:][condition])            
                else:
                    correlation[idx_meas,i], p_value[idx_meas,i] = pearsonr(IC_mat[i,:], EC_v[i,:])
    return correlation, p_value


#=====================================#
# Accuracy of EC in reconstructing the whole perturbome (IC) or its hubs:
# correlation (pearson/spearman) betw EC matrices (stored in a list) and IC + correlation betw their in-strength

def corrMeasures(meas_mat, IC_measure, indices, compute_strength=False):
    
    def in_strength(mat,pos=False):
        return np.sum(mat,axis=0)

    Scorr = np.zeros(len(meas_mat))
    Pcorr = np.zeros(len(meas_mat))
    
    for i in range(len(meas_mat)):
        if compute_strength:
            v1 = in_strength(IC_measure)
            v2 = in_strength((meas_mat[i]))
        else:
            v1 = IC_measure
            v2 = (meas_mat[i])
        csp, _ = np.round(spearmanr(v1.flatten(), v2.flatten()), 2)
        Scorr[i] = csp
        cp, _ = np.round(pearsonr(v1.flatten(), v2.flatten()), 2)
        Pcorr[i] = cp  
            
    return Scorr, Pcorr


#=====================================#
# Accuracy of EC in reconstructing the out strength of every node in the perturbome:
# correlation (pearson/spearman) betw EC out-strengths (stored in a list) and IC out-strengths

def corrMeasures_outStr(meas_mat, IC_measure, indices):
    
    Scorr = np.zeros(len(meas_mat))
    Pcorr = np.zeros(len(meas_mat))

    for i in range(len(meas_mat)):
        v1 = np.sum(IC_measure,axis=-1);    
        v2 = np.sum((meas_mat[i]),axis=-1)
        csp,_ = np.round(spearmanr(v1.flatten(),v2.flatten()),2); 
        Scorr[i] = csp
        cp,_  = np.round(pearsonr(v1.flatten(),v2.flatten()),2);  
        Pcorr[i] = cp  

    return Scorr, Pcorr


#=================================================================================================#
# PLOTS

def plot_correlations(corr_mat, label_list, indices, ylabel=r'$\rho_{sp}(EC,IC)$', sort=False, alpha=0.8, markersize=20,lw=2,ymin=0,ymax=1, figsize=(8, 6), ax=None, outf: str = None, show_plot: bool = False):
    #colors_dl = ['#255D93', '#5FA6D6', '#B02106', '#F24D33', '#2C2C2C', '#787878']
    colors_dl = ['#255D93', '#B02106', '#2C2C2C']
    
    if ax is None:
        fig,ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure() 
        outf=None
        
    if sort:
        L = np.argsort(corr_mat[0])
    else:
        L = np.arange(len(indices))
        
    for idx_plot in range(len(label_list)):
        ax.plot(corr_mat[idx_plot][L], '.-', markersize=markersize,
                color=colors_dl[idx_plot], label=label_list[idx_plot], lw=lw, alpha=alpha)

    ax.set_xlabel(r'source channel ID', fontsize=DIM)
    ax.set_ylabel(ylabel, fontsize=DIM)
    ax.set_ylim(ymin, ymax)

    pl.set_format(ax, pwr_x_max=2,  DIM = DIM)
    
    ax.set_xticks(np.arange(len(indices)))
    ax.set_xticklabels([ indices[i] for i in range(len(indices))])
    ax.tick_params(axis='x', labelsize=DIM-12, rotation=80)
    ax.tick_params(axis='y', which='major', rotation=0, labelsize=DIM)
    ax.legend(fontsize=DIM-5, ncol=1, handlelength=1, frameon=False)

    fig.set_facecolor('none')

    if outf is not None:
        plt.savefig(outf, bbox_inches='tight')
        if not show_plot:
            plt.close()

            
            
#=========================================================================================================#
# accuracy of EC in reconstructing the perturbome [for every fixed stim. channel.]

def plot_corr_elements(vec,mat_labels,indices,title,label=r'$\rho_{{\,spearman}}$',vmin=-1, vmax=1, 
                       cmap = coldhot_cmap, pearson=False, ax=None, outf : str = None, show_plot = True):

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    if ax is None:
        fig,ax = plt.subplots()
    else:
        fig = ax.get_figure() 
        outf=None
        
    myplot = ax.imshow(vec,vmin=vmin, vmax=vmax, cmap = cmap)
    
    divider = make_axes_locatable(ax); 
    cax = divider.append_axes("right", size="10%", pad=0.05)
    cbar = fig.colorbar(myplot, cax=cax); cbar.ax.tick_params(labelsize=DIM)
    cbar.set_label(label, fontsize=DIM)
    
    ax.set_xlabel('source channel ID',fontsize=DIM)
    ax.xaxis.set_label_position('top') 

    ax.set_yticks(np.arange(len(mat_labels)))
    if pearson==False:
        ax.set_yticklabels([r'$\rho_{sp}$('+mat_labels[idx_meas]+',IC)' for idx_meas in range(len(mat_labels))])
    else:
        ax.set_yticklabels([r'$\rho_{p}$('+mat_labels[idx_meas]+',IC)' for idx_meas in range(len(mat_labels))])
    #ax.set_yticklabels(mat_labels)
    ax.set_xticks(np.arange(len(indices)))
    dm = len(indices)-1
    ax.set_xticklabels([ indices[i] for i in range(len(indices))])

    ax.tick_params(axis='x', labelsize=20, rotation=45)
    ax.tick_params(axis='y', which='major', rotation=0, labelsize=DIM-2)
    ax.xaxis.tick_top()

    ax.set_title(title, fontsize=DIM)

    fig.set_facecolor('none')

    if outf is not None:
        plt.savefig(outf, bbox_inches='tight')
        if not show_plot:
            plt.close()

        
#=========================================================================================================#
#  accuracy of EC in reconstructing the perturbome IC, IC in-strengths and IC out-strengths

def plot_corr_matrix(vec,meas_lab,title,vmin=-1, vmax=1, cmap = coldhot_cmap, pearson=False, 
                         ax=None, outf : str = None, show_plot=True):
    
    if ax is None:
        fig,ax = plt.subplots()
    else:
        fig = ax.get_figure() 
        outf=None
        
    myplot = ax.imshow(vec,vmin=vmin, vmax=vmax, cmap = cmap)
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax); 
    cax = divider.append_axes("right", size="10%", pad=0.05)
    cbar = fig.colorbar(myplot, cax=cax); cbar.ax.tick_params(labelsize=DIM)
    cbar.set_label(title, fontsize=DIM)

    ax.set_yticks(np.arange(len(meas_lab)))
    if pearson==False:
        ax.set_yticklabels([r'$\rho_{sp}$('+meas_lab[idx_meas]+',IC)' for idx_meas in range(len(meas_lab))])
    else:
        ax.set_yticklabels([r'$\rho_{p}$('+meas_lab[idx_meas]+',IC)' for idx_meas in range(len(meas_lab))])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['global', 'in-strength', 'out-strenght',])
    
    ax.tick_params(axis='x', labelsize=DIM, rotation=80)
    ax.tick_params(axis='y', which='major', labelsize=DIM)
    ax.xaxis.tick_top()
    
    ax.set_facecolor('none');
    from matplotlib.ticker import ScalarFormatter
    formatter = ScalarFormatter(useMathText=True);    
    formatter.set_scientific(True); formatter.set_powerlimits((-2, 2))
    cbar.ax.yaxis.set_major_formatter(formatter); cbar.ax.yaxis.offsetText.set_fontsize(20)
    cbar.ax.xaxis.set_major_formatter(formatter); cbar.ax.xaxis.offsetText.set_fontsize(20)

    fig.set_facecolor('none')

    if outf is not None:
        plt.savefig(outf, bbox_inches='tight')
        if not show_plot:
            plt.close()


#=========================================================================================================#
# data-split validation

def plot_correlations_cv(corr_mat, corr_mat2, corr_mat3, label, color='blue', pearson=False, marker='X-', 
                         markersize=12, figsize=(5, 4), ax=None, outf : str = None, show_plot = False):

    if ax is None:
        fig,ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure() 
        outf=None
        
    L = np.argsort(corr_mat)
    line, = ax.plot(corr_mat[L], marker, markersize=markersize, color=color, label=label, lw=3, alpha=1)
    
    L = np.argsort(corr_mat2)
    line, = ax.plot(corr_mat2[L], '-', color='crimson', label='validation set 1', lw=2, alpha=1)
    
    L = np.argsort(corr_mat3)
    line, = ax.plot(corr_mat3[L], '--.', color='orange', label='validation set 2', lw=2, alpha=1)
    
    ax.set_xlabel('source channel ID\n'+r'$_{sorted\,\,by\,\,\rho}$', fontsize=DIM)
    if pearson==False:
        ax.set_ylabel(r'$\rho_{sp}(EC,IC)\,|_{source\, fixed}$', fontsize=DIM)
    else:
        ax.set_ylabel(r'$\rho_{p}(EC,IC)\,|_{source\, fixed}$', fontsize=DIM)
    ax.set_ylim(-1, 1)
    
    ax.legend(fontsize=DIM-5, ncol=1, handlelength=1)
    ax.set_xticks([])
    pl.set_format(ax, axis_ticks = 'both', DIM = DIM)

    fig.set_facecolor('none')
    
    if outf is not None:
        plt.savefig(outf, bbox_inches='tight')
        if not show_plot:
            plt.close()

#=========================================================================================================#

def plotcorrs(corr_mat, corr_mat_2, label_list, label_list2, indices, sort=False, ymin=-1, ymax=1, figsize=(4,3), outf: str = None, show_plot: bool = False):
    DIM = 25
    #colors_dl = ['#255D93', '#5FA6D6', '#B02106', '#F24D33', '#2C2C2C', '#787878']
    colors_dl = ['#255D93', '#B02106', '#2C2C2C']
    
    fig, ax = plt.subplots(figsize=figsize)

    if sort:
        L = np.argsort(corr_mat[idx_plot])
    else:
        L = np.arange(len(indices))
    #--------------------------------------------------------------------------------------------#
    # CORRECTED WITH DISTANCE
    #--------------------------------------------------------------------------------------------#
    idx_plot=0
    line, = ax.plot(corr_mat[idx_plot][L], 'X-', markersize=12/3*2, markeredgecolor='white', markeredgewidth=0.2,
                    color=colors_dl[idx_plot], label=label_list[idx_plot], lw=2, alpha=1)
    idx_plot=1
    line, = ax.plot(corr_mat[idx_plot][L], '.-', markersize=20/3*2, markeredgecolor='white', markeredgewidth=0.2,
                    color=colors_dl[idx_plot], label=label_list[idx_plot], lw=2, alpha=1)
    idx_plot = 2
    line, = ax.plot(corr_mat[idx_plot][L], '.-', markersize=15/3*2, markeredgecolor='white', markeredgewidth=0.2,
                    color=colors_dl[idx_plot], label=label_list[idx_plot], lw=2, alpha=1)

    #--------------------------------------------------------------------------------------------#
    # NON CORRECTED WITH DISTANCE
    #--------------------------------------------------------------------------------------------#
    idx_plot=0
    line, = ax.plot(corr_mat_2[idx_plot][L], 'X-', markersize=12/3*2, markeredgecolor='white', markeredgewidth=0.2,
        color=colors_dl[idx_plot], label=label_list2[idx_plot], 
                    lw=1.5, alpha=0.5)
    idx_plot=1
    line, = ax.plot(corr_mat_2[idx_plot][L], '.-', markersize=20/3*2, markeredgecolor='white', markeredgewidth=0.2,
        color=colors_dl[idx_plot], label=label_list2[idx_plot], 
                    lw=1.5, alpha=0.5)
    line.set_dashes([2, 2])
    idx_plot=2
    line, = ax.plot(corr_mat_2[idx_plot][L], '.-', markersize=15/3*2, markeredgecolor='white',markeredgewidth=0.2,
            color=colors_dl[idx_plot], 
                    label=label_list2[idx_plot], 
                    lw=1.5, alpha=0.5)
    #--------------------------------------------------------------------------------------------#
    ax.set_xlabel(r'source channel ID', fontsize=DIM)
    ax.set_ylabel(r'$\rho_{sp}(\epsilon_{EC},\epsilon_{IC})\,|_{source\, fixed}$', fontsize=DIM)
    ax.set_ylim(ymin, ymax)

    pl.set_format(ax, pwr_x_max=2,  DIM = DIM)
    
    ax.set_xticks(np.arange(len(indices)))
    ax.set_xticklabels([ indices[i] for i in range(len(indices))])
    ax.tick_params(axis='x', labelsize=10, rotation=45)
    ax.tick_params(axis='y', which='major', rotation=0, labelsize=DIM-2)
    ax.legend(fontsize=DIM-5, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.7), labelspacing=0.4, 
              handletextpad=0.8, handlelength = 1., frameon=False) 
    #-------------------------------------

    fig.set_facecolor('none')
    
    if outf is not None:
        plt.savefig(outf, bbox_inches='tight')
        if not show_plot:
            plt.close()
