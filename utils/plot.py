import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgba
from scipy.stats import linregress, gaussian_kde
from scipy.stats import spearmanr, pearsonr
from scipy.optimize import curve_fit

import seaborn as sns 
import numpy as np


from . import load_EC as lec
from . import correlations as cc

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

import os
def Set_Dir_Plots(path):
    if not os.path.exists(path):
        os.mkdir(path)

#=================================================================================================#


colors = ['#2F7FC3','#E62A08','#464646','#FFD700','#32CD32','#8A2BE2']


#-------------------------------------------------------------------------------------------------#

def interpolate_light_to_dark(color, steps=10):
    """
    Crea una sfumatura di colori dal bianco fino al colore specificato.
    
    Parameters:
    - color (str): Codice colore di base in formato hex (es. '#255D93').
    - steps (int): Numero di colori nella sfumatura.
    
    Returns:
    - list: Lista di codici colore hex dalla sfumatura chiara a quella scura.
    """
    color_rgb = np.array(mcolors.to_rgb(color))
    white_rgb = np.array([1, 1, 1])  # RGB per il bianco
    gradient = [(1 - t) * white_rgb + t * color_rgb for t in np.linspace(0, 1, steps)]
    return [mcolors.to_hex(c) for c in gradient]


def get_color_gradient(idx_meas, num_colors, base_color):
    """
    Generates a gradient of colors from light to dark based on the base color for the specified idx_meas.
    
    Parameters:
    - idx_meas (int): Index of the base color in colorz.
    - num_colors (int): Number of colors required in the gradient.
    
    Returns:
    - list of str: List of hex color codes from light to dark.
    """
    return interpolate_light_to_dark(base_color, num_colors)            

#=================================================================================================#
# Axes formatter for plots 

def set_format(ax, axis_ticks = 'both', pwr_x_min=-1, pwr_x_max=1, pwr_y_min=-1, pwr_y_max=1,  cbar = None, pwr_cbar_min=-1, pwr_cbar_max=1,  DIM = 30):
    
    import seaborn as sns
    
    sns.despine(ax=ax, trim=False)
    ax.set_facecolor('none')
    
    # - - -  TICKS
    ax.tick_params(axis=axis_ticks, which='major', labelsize=DIM)
    
    # - - -  FORMATTER x axis
    formatter_x = ScalarFormatter(useMathText=True)   
    formatter_x.set_scientific(True)
    formatter_x.set_powerlimits((pwr_x_min, pwr_x_max))
    ax.xaxis.set_major_formatter(formatter_x)
    ax.xaxis.offsetText.set_fontsize(DIM-10)
    
    from matplotlib.transforms import ScaledTranslation
    dx, dy = 15/72, 15/72
    offset = ScaledTranslation(dx, dy, ax.figure.dpi_scale_trans)
    ax.xaxis.offsetText.set_transform(ax.xaxis.offsetText.get_transform() + offset)

    # - - -  FORMATTER y axis
    formatter_y = ScalarFormatter(useMathText=True)    
    formatter_y.set_scientific(True) 
    formatter_y.set_powerlimits((pwr_y_min, pwr_y_max))
    ax.yaxis.set_major_formatter(formatter_y);
    ax.yaxis.offsetText.set_fontsize(DIM-10)
    
    if cbar:
        # - - -  FORMATTER cbar
        formatter_cbar = ScalarFormatter(useMathText=True)   
        formatter_cbar.set_scientific(True)
        formatter_cbar.set_powerlimits((pwr_cbar_min, pwr_cbar_max))
        cbar.ax.yaxis.set_major_formatter(formatter_cbar); 
        cbar.ax.yaxis.offsetText.set_fontsize(DIM-10)
        cbar.ax.xaxis.set_major_formatter(formatter_cbar); 
        cbar.ax.xaxis.offsetText.set_fontsize(DIM-10)
        
        # Move the offset text to the top of the colorbar
        dx, dy = 0.8, 0.3  # Adjust dy for vertical and dx for horizontal shifts
        cbar_offset = ScaledTranslation(dx, dy, cbar.ax.figure.dpi_scale_trans)
        cbar.ax.yaxis.offsetText.set_transform(cbar.ax.yaxis.offsetText.get_transform() + cbar_offset)
        

#=================================================================================================#
# Plot the MEA channel map [with colorbar]

def plot_map(pos, car_array, cbar_label='cluster ID', title='recording channels map', cmap='viridis', 
             figsize=(15, 7), ax=None, outf : str = None, show_plot = True):

    # Colormap
    n_clusters = len(np.unique(car_array))
    if cmap is None:
        cmap = alternating_colormap(n_clusters)
    else:
        cmap = cmap
        
    if ax is None:
        fig,ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure() 
        
    # channels map
    im = ax.scatter(pos[:,0],pos[:,1],c=car_array,s=1,marker='s',cmap=cmap)
    xt=ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')

    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)
    
    if ax is None:
        if outf:
            ax.savefig(outf, bbox_inches='tight')
            if not show_plot:
                plt.close()
        else:
            plt.show()

#=================================================================================================#
# Plot the perturbome : MEA channel map [with colorbar] - colors corresponding to the response 
#                       (or predicted response) of each channel to the stimulation of a fixed channel

def plot_perturbome(mat, map_coords, channel, indices, label='IC', stim_id=0, vmin=0, vmax=1, cmap='cool', log=False, 
             DIM=40, dotsize=22, starsize=1800, figsize=(20,11), ax=None, outf: str = None, show_plot=False):

    ch  = channel[indices[stim_id]]

    if ax is None:
        fig,ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # map of recording channels
    ax.scatter(map_coords[:,0], map_coords[:,1], s=dotsize, marker='s', c='lightgrey', edgecolor='white', zorder=1) #label = 'recording channels',
    
    dx, dy = 10.5/1000, 40/1000
    
    # stim. channel
    ax.scatter(map_coords[channel==ch,0], map_coords[channel==ch,1], c='tab:red', edgecolor='white', s=starsize, marker='*', label = f'stim. channel {indices[stim_id]}', zorder=0)
    ax.text(map_coords[channel==ch,0]+dx, map_coords[channel==ch,1]+dy, str(indices[stim_id]), fontdict=dict(color='black', alpha=1, size=DIM), zorder=4)
    
    idxs = np.where(mat[stim_id,:]!=0)
    if log:
        scatter = ax.scatter(map_coords[idxs,0], map_coords[idxs,1], s=dotsize, marker='s', c=np.log10(mat[stim_id,idxs]), edgecolor='white', zorder=2, cmap=cmap, vmin=vmin, vmax=vmax) # label = 'recording channels', 
    else:
        scatter = ax.scatter(map_coords[idxs,0], map_coords[idxs,1], s=dotsize, marker='s', c=mat[stim_id,idxs], edgecolor='white', zorder=2, cmap=cmap, vmin=vmin, vmax=vmax) # label = 'recording channels', 
    
    ax.set_xlabel('x  (mm)',fontsize=DIM)
    ax.set_ylabel('y  (mm)',fontsize=DIM)
    
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(label=label, fontsize=DIM) 
    cbar.ax.tick_params(labelsize=DIM)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1, fontsize=DIM, frameon=False)
    
    set_format(ax=ax, pwr_x_min=-3, pwr_x_max=3, pwr_y_min=-2, pwr_y_max=2, axis_ticks = 'both', cbar = cbar, DIM = DIM)
    
    if ax is None:
        if outf:
            ax.savefig(outf, bbox_inches='tight')
            if not show_plot:
                plt.close()
        else:
            plt.show()
            
#=================================================================================================#
# plot matrix – aspect='auto'

def plot_mat_aspect(mat, vmin=None, vmax=None, cmap='viridis', title=None, xlabel='target', ylabel='source', 
                    cbarlabel='spike count', invert_y: bool = True, xticklabels: list = None, yticklabels: list = None, 
                    figsize=(15, 10), ax = None, outf: str = None, show_plot: bool = True):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure() 
        
    if vmin is None and vmax is None:
        vmin, vmax = np.percentile(mat, [5, 97.5])
        
    im = ax.imshow(mat, aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    if xticklabels is not None:
        ax.set_xticks(np.arange(len(xticklabels)))
        ax.set_xticklabels(xticklabels, rotation=45)

    if yticklabels is not None:
        ax.set_yticks(np.arange(len(yticklabels)))
        ax.set_yticklabels(yticklabels, fontsize=21)

    if invert_y:
        ax.invert_yaxis()

    fig.colorbar(im, ax=ax, label=cbarlabel)

    if ax is None:
        if outf:
            plt.savefig(outf, bbox_inches='tight')
            
            if not show_plot:
                plt.close()
        else:
            plt.show()
        

#=================================================================================================#
# plot matrix

def plot_mat(mat, title=None, xlabel='target', ylabel='source', cmap='viridis', cbarlabel='spike count', invert_y: bool = True, figsize=(12, 10), ax = None, outf : str = None, show_plot = True):

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure() 
        
    if vmin is None and vmax is None:
        vmin, vmax = np.percentile(mat, [5, 97.5])
                
    im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)
        
    if invert_y:
        ax.invert_yaxis()
    
    fig.colorbar(im, ax=ax, label=cbarlabel)
    
    if outf:
        plt.savefig(outf, bbox_inches='tight')
    
    if ax is None:
        if outf:
            plt.savefig(outf, bbox_inches='tight')
            
            if not show_plot:
                plt.close()
        else:
            plt.show()


#=================================================================================================#
# scatter plot
            
def plot_scatter(ic_mat, ec_mat, zeroExp=-11, log=False, xlabel='IC', ylabel='EC', title=None, dotsize=0.1, 
                 cmap=None, edgecolor=None, linewidths=0.2, regcolor='tab:red', dotcolor='tab:blue', reg_line=True, show_corr=True,
                 ymin=None, ymax=None, ax=None, outf=None, show_plot=False):
    import seaborn as sns
    from scipy.stats import pearsonr, spearmanr, gaussian_kde
    ic_vec = ic_mat.flatten()
    if log:
        ec_vec = np.log10(ec_mat.flatten()+10**zeroExp)
        x=ic_vec[ec_mat.flatten()!=0]
        y=ec_vec[ec_mat.flatten()!=0]
    else:
        ec_vec = ec_mat.flatten()
        x=ic_vec
        y=ec_vec
    
    if ax is None:
        fig,ax = plt.subplots()
    else:
        fig = ax.get_figure() 
        
    if cmap:
        xy = np.vstack([ic_vec, ec_vec])
        z = gaussian_kde(xy)(xy)
        vmax = np.max(z); vmin = vmax; 
        # Plot with density-based color
        scatter = ax.scatter(ic_vec, ec_vec, c=z, s=dotsize, alpha=0.7, edgecolor=edgecolor, linewidths=linewidths, cmap=cmap)
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
        cbar.set_label(r'density', fontsize=DIM)
        cbar.ax.tick_params(labelsize=DIM)
    else:
        ax.scatter(ic_vec, ec_vec, s=dotsize, c=dotcolor, edgecolor=edgecolor, linewidths=linewidths, alpha=0.7)

    if reg_line:
        sns.regplot(x=x, y=y, ax=ax, scatter=False, color=regcolor, line_kws={'linewidth': 1})
    
    if show_corr:
        cp = np.round(pearsonr(x,y)[0],2)
        cs = np.round(spearmanr(x,y)[0],2)
        if title:
            ax.set_title(title+'\n'+rf'$\rho_p={cp}$'+'\n'+rf'$\rho_{{sp}}={cs}$')
        else:
            ax.set_title(rf'$\rho_p={cp}$'+'\n'+rf'$\rho_{{sp}}={cs}$')
    else:
        if title:
            ax.set_title(title)
        
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin,ymax)
        
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel);

    if cmap:
        set_format(ax, pwr_x_min=-3, pwr_x_max=3, pwr_y_min=-2, pwr_y_max=2, axis_ticks = 'both', cbar = cbar, DIM = DIM)
    else:
        set_format(ax, pwr_x_min=-3, pwr_x_max=3, pwr_y_min=-2, pwr_y_max=2, axis_ticks = 'both', cbar = None, DIM = DIM)
    
    if ax is None:
        if outf:
            ax.savefig(outf, bbox_inches='tight')
            if not show_plot:
                plt.close()
        else:
            plt.show()


        
#=================================================================================================#
#                                      B A R P L O T S
#=================================================================================================#

#-------------------------------------------------------------------------------------------------#
# Permutation test – referred to plot_bars function (see below)

def permutation_test(data1, data2, num_permutations=10000, alternative='two-sided'):
    
    # mean(data1) vs. mean(data2) 
    
    observed_diff = np.mean(data2) - np.mean(data1)
    combined = np.concatenate([data1, data2])
    perm_diffs = []
    
    n1 = len(data1)
    n2 = len(data2)
    
    for _ in range(num_permutations):
        np.random.shuffle(combined)
        
        # Preserving data1 and data2 sizes2
        perm_data1 = combined[:n1]
        perm_data2 = combined[n1:n1 + n2]
        
        # mean(data1_permuted) - mean(data2_permuted)
        perm_diff = np.mean(perm_data2) - np.mean(perm_data1)
        perm_diffs.append(perm_diff)

    perm_diffs = np.array(perm_diffs)
    
    if alternative == 'less':
        p_val = np.sum(perm_diffs >= observed_diff) / num_permutations
    elif alternative == 'greater':
        p_val = np.sum(perm_diffs <= observed_diff) / num_permutations
    elif alternative == 'two-sided':
        p_val = np.sum(np.abs(perm_diffs) >= np.abs(observed_diff)) / num_permutations
    else:
        raise ValueError("Alternative must be 'less', 'greater', or 'two-sided'")

    return p_val

#-------------------------------------------------------------------------------------------------#
# Labels for significance – referred to plot_bars function (see below)

def significance_label(p_value):
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'n.s.'

#-------------------------------------------------------------------------------------------------#
# barplot of the mean of vec1 elements corresponding to null or non-null elements of vec2

def plot_bars(mat1, mat2, label1, label2, mat2_pval, color, colBar='black',doubleP=True, 
              alternative='two-sided', alpha_th = 0.001, title=None, ax=None, outf : str = None, show_plot=False):
    
    if ax is None:
        fig,ax = plt.subplots()
    else:
        fig = ax.get_figure() 
        
    vec1 = mat1.flatten()      # you do the average of vec1 for links corresponding to significant vec2 links
                               # and the average of vec1 for links corresponding to non-significance vec2 links
    vec2 = mat2.flatten()
    pval2 = mat2_pval.flatten()
    
    x = vec1
    y = vec2
        
    # Filter data based on significance
    if doubleP:
        t_sign    = vec1[np.logical_or(pval2 <= alpha_th, pval2 >= 1-alpha_th)]
        t_nonSign = vec1[np.logical_and(pval2 > alpha_th, pval2 < 1-alpha_th)]
    else:
        t_sign    = vec1[pval <= alpha_th]
        t_nonSign = vec1[pval > alpha_th]
        
    # means of vec1 corresponding to non-sign vec2 links (left bar) and to sig. links (right bar)
    means = [np.mean(t_nonSign), np.mean(t_sign)]
    stds  = [np.std(t_nonSign)/np.sqrt(len(t_nonSign)), np.std(t_sign)/np.sqrt(len(t_sign))]

    # Calculate significance (t-test)
    if len(t_nonSign) > 1 and len(t_sign) > 1:
        p_val = permutation_test(t_nonSign, t_sign, num_permutations=10000, alternative=alternative)
    else:
        p_val = None  # Too few data points for p-value calculation

    # Plot bars
    bars = ax.bar([label2 + '\n$^{no\,\,sig}$', 
                   label2 + '\n$^{sig}$'], means, color='white', 
                  edgecolor=[color, color], linewidth=7)

    ax.errorbar(0, means[0], yerr=stds[0], fmt='none', ecolor=colBar, elinewidth=5, capsize=10)  # vec2 non-sign
    ax.errorbar(1, means[1], yerr=stds[1], fmt='none', ecolor=colBar, elinewidth=5, capsize=10)  # vec2 sign

    for bar in bars:
        bar.set_linewidth(5)

    # Add significance asterisks
    if p_val is not None:
        sig_label=significance_label(p_val)

        if sig_label:
            max_mean = max(means)
            ax.text(0.5, 1.1, sig_label, ha='center', va='top', transform=ax.transAxes, fontsize=DIM-10)
        
        ax.set_ylabel(label1, fontsize=DIM)
        
        # Format only for y-axis (NOT x-axis)
        from matplotlib.ticker import ScalarFormatter
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.offsetText.set_fontsize(20)

        ax.tick_params(axis='both', which='major', labelsize=DIM)
        ax.set_facecolor('none')
        sns.despine(ax=ax, trim=False)

        if title:
            ax.set_title(title, fontsize=DIM, y=1.2)
        
    if ax is None:
        if outf:
            ax.savefig(outf, bbox_inches='tight')
            if not show_plot:
                plt.close()
        else:
            plt.show()

#=================================================================================================#
#                           SPATIAL PROPERTIES OF THE PERTURBOME
#=================================================================================================#


#-------------------------------------------------------------------------------------------------#
# <connectivity> as a function of distance

def plot_binned_mean(C_mat, D_mat, xlabel='eucl. dist.(mm)', ylabel='IC', color='tab:green', N_bins=10, linewidths=0.2,
                     xmin=0, xmax=None, ymin=0, ymax=None, figsize=(4,4), ax=None, outf=None, show_plot=False):
    
    D_v = D_mat.flatten()   # distance matrix     (x-azis)
    C_v = C_mat.flatten()   # connectivity matrix (y-axis)
    
    if ax is None:
        fig,ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure() 
        
    # Bin the x-axis values and compute means and standard deviations for y-values in each bin
    bins = np.linspace(np.min(D_v), np.max(D_v), N_bins + 1)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    y_means = np.zeros(N_bins)
    y_stds  = np.zeros(N_bins)

    for j in range(N_bins):
        bin_mask      = (D_v >= bins[j]) & (D_v < bins[j+1])
        y_vals_in_bin = C_v[bin_mask]
        if len(y_vals_in_bin) > 0:
            y_means[j] = np.mean(y_vals_in_bin)
            y_stds[j]  = np.std(y_vals_in_bin)
        else:
            y_means[j] = np.nan
            y_stds[j]  = np.nan
    
    y_stds /= np.sqrt(len(y_means))  # s.e.m.

    # Plot error bars instead of scatter
    ax.fill_between(bin_centers, y_means-y_stds, y_means+y_stds,color=color, alpha=0.3)
    ax.plot(bin_centers, y_means,'-', color=color,lw=4)#, edgecolor=edgecolor,markersize=10)
    ax.errorbar(bin_centers, y_means, yerr=y_stds, fmt='o', color=color, ecolor=color, elinewidth=linewidths, capsize=4)

    ax.set_ylabel('$<$'+ylabel+'$>$', fontsize=DIM)
    ax.set_xlabel(xlabel, fontsize=DIM)

    if xmax:
        ax.set_xlim(xmin, xmax)
    if ymax:
        ax.set_ylim(xmax, ymax)

    set_format(ax, axis_ticks='both', cbar=None, DIM=DIM)

    if ax is None:
        if outf:
            ax.savefig(outf, bbox_inches='tight')
            if not show_plot:
                plt.close()
        else:
            plt.show()


#=================================================================================================#
#            PROBABILITY OF SIGNIFICANT CONNECTIONS AS A FUNCTION OF DISTANCE
#                               - exponential fit -

def func(x_, a_, b_, c_):
    return a_ * np.exp(-b_ * x_) + c_  

#-------------------------------------------------------------------------------------------------#
# Compute the probability of connection P as a function of distance: 
#      - bin the distance
#      - compute P for each bin
#      - fit the rule
# (recommended for connectivity matrices after significant test)

def fit_measure(meas_matrix, dist_matrix, N_bins = 1000, bounds=(-np.inf,np.inf)):
    
    def is_list_of_matrices(var):
        return isinstance(var, list)
    
    Pdf={}
    dist_poss={}

    if is_list_of_matrices(meas_matrix) is True:
        
        params_found = np.zeros((len(meas_matrix),3))
        
        for idx_meas in range(len(meas_matrix)):

            matr = np.copy(meas_matrix[idx_meas])
            bin_weights = matr!=0

            d_min = np.min(dist_matrix.flatten())
            d_max = np.max(dist_matrix.flatten())
            dD    = (d_max-d_min)/N_bins
            possible_distances = np.arange(d_min, d_max-dD, dD)

            D     = dist_matrix

            tot_distances  = np.zeros(len(possible_distances)-1)
            n_connections  = np.zeros(len(possible_distances)-1)
            prob_connetion = np.zeros(len(possible_distances)-1)

            for i in range(len(possible_distances)-1):

                condition        = np.logical_and(D>possible_distances[i], D<=possible_distances[i+1])
                tot_distances[i] = len(D[condition])
                n_connections[i] = np.sum(bin_weights[condition])

            tot_distances[tot_distances==0] = np.inf
            prob_connetion = np.divide(n_connections,tot_distances)/np.sum(np.divide(n_connections,tot_distances))

            x = possible_distances[1:len(possible_distances)-1]
            y = prob_connetion[1:len(possible_distances)-1]

            popt, pcov = curve_fit(func, x, y, bounds=bounds)

            Pdf[idx_meas]            = y
            dist_poss[idx_meas]      = x
            params_found[idx_meas,:] = popt
        
    else:
        
        params_found = np.zeros(3)

        matr = np.copy(meas_matrix)
        bin_weights = matr!=0

        d_min = np.min(dist_matrix.flatten())
        d_max = np.max(dist_matrix.flatten())
        dD    = (d_max-d_min)/N_bins
        possible_distances = np.arange(d_min,d_max-dD,dD)

        D = dist_matrix

        tot_distances  = np.zeros(len(possible_distances)-1)
        n_connections  = np.zeros(len(possible_distances)-1)
        prob_connetion = np.zeros(len(possible_distances)-1)

        for i in range(len(possible_distances)-1):

            condition        = np.logical_and(D>possible_distances[i], D<=possible_distances[i+1])
            tot_distances[i] = len(D[condition])
            n_connections[i] = np.sum(bin_weights[condition])

        tot_distances[tot_distances==0] = np.inf
        prob_connetion = np.divide(n_connections,tot_distances)/np.sum(np.divide(n_connections,tot_distances))

        x = possible_distances[1:len(possible_distances)-1]
        y = prob_connetion[1:len(possible_distances)-1]

        popt, pcov     = curve_fit(func, x, y, bounds=bounds)

        Pdf             = y
        dist_poss       = x
        params_found[:] = popt

    return Pdf,dist_poss,params_found

#-------------------------------------------------------------------------------------------------#
# plot exponential fit

def plot_fit_meas(Pdf, dist_poss, params_found, colors, label_meas, label_curve = 'exponential fit', 
                  xmax=None, ymax=None,  edgecolor='white', cmap=None, linewidths=0.2, figsize=(6,4), 
                  ax=None, outf : str = None, show_plot=False):

    if ax is None:
        fig,ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    y    = Pdf
    x    = dist_poss
    popt = params_found[:]          

    if cmap:
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        vmin = 0; vmax = np.max(z)
        # Plot with density-based color
        scatter = ax.scatter(x, y, c=z, s=100, edgecolor=edgecolor, linewidths=linewidths, cmap=cmap,
                             label=label_meas, alpha=0.7, zorder=2)
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
        cbar.set_label(r'density', fontsize=DIM)
        cbar.ax.tick_params(labelsize=DIM)
    else:
        ax.scatter(x,y, c=colors, s=100, edgecolor=edgecolor, linewidths=linewidths, label=label_meas, alpha=0.7, zorder=2)
    modeled_y = func(x, *popt)
    ax.plot(np.array(sorted(x)), np.array(sorted(modeled_y, reverse=True)), 'k-', color = '#E62A08', 
            lw=8, label=label_curve, zorder=1)

    ax.set_ylabel('pdf', fontsize=DIM)
    ax.set_xlabel('distance (mm)', fontsize=DIM)

    if ymax:
        ax.set_ylim(0, ymax)

    ax.set_xlim(-0.2,xmax)
    ax.set_xticks(np.arange(0,xmax,2))
    ax.legend(fontsize=DIM, ncol=1, loc='upper center', bbox_to_anchor=(0.5, 1.6), labelspacing=0.4, 
              handletextpad=0.8, handlelength = 1., frameon=False) 
    if cmap:
        set_format(ax, axis_ticks = 'both', cbar = cbar, DIM = DIM)
    else:
        set_format(ax, axis_ticks = 'both', cbar = None, DIM = DIM)

    if ax is None:
        if outf:
            ax.savefig(outf, bbox_inches='tight')
            if not show_plot:
                plt.close()
        else:
            plt.show()
        
#=================================================================================================#
#                  INTENSITY OF CONNECTIONS AS A FUNCTION OF DISTANCE
#                               -  model selection and fit  -
            
from scipy.optimize import curve_fit
from scipy.stats import zscore
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

# Funzioni modello
def exp_func(x, a, b, c):
    return a * np.exp(-b * x)+ c
                      
def power_func(x, a, b):
    return a * x ** (-b)

def linear_func(x, a, b):
    return a * x + b

# Outlier removal
def remove_outliers(x, y, z_thresh=3):
    mask = np.abs(zscore(np.column_stack((x, y)), axis=0)) < z_thresh
    return x[mask.all(axis=1)], y[mask.all(axis=1)]

def remove_outliers_quantiles(x, y, q=0.01):
    xq_low, xq_high = np.quantile(x, [q, 1 - q])
    yq_low, yq_high = np.quantile(y, [q, 1 - q])
    mask = (x >= xq_low) & (x <= xq_high) & (y >= yq_low) & (y <= yq_high)
    return x[mask], y[mask]

def fit_and_plot(x, y, fit_type="exp", rm_quantiles=True, q=0.02, z_thresh=5, xlabel='eucl. distance (mm)', ylabel='TE',
                 cmap=None, edgecolor=None, linewidths=0.2, dotsize=10, dotcolor='tab:blue', ax=None, plot=True):

    # clean data from outliers
    if rm_quantiles:
        x_clean, y_clean = remove_outliers_quantiles(x, y, q=q)
    else:
        x_clean, y_clean = remove_outliers(x, y, z_thresh=z_thresh)

    # for model selection
    def compute_aic_bic(y_true, y_pred, num_params):
        residuals = y_true - y_pred
        rss = np.sum(residuals**2)
        n = len(y_true)
        aic = 2*num_params + n * np.log(rss / n)
        bic = num_params * np.log(n) + n * np.log(rss / n)
        return aic, bic

    if fit_type == "exp":
        model_func = exp_func
        p0 = (np.max(y_clean), 0.01, np.min(y_clean))
        bounds = ([0, 0, -np.inf], [np.inf, np.inf, np.inf])
        popt, _ = curve_fit(model_func, x_clean, y_clean, p0=p0, bounds=bounds, maxfev=10000)
        y_pred = model_func(x_clean, *popt)
        r2 = r2_score(y_clean, y_pred)
        aic, bic = compute_aic_bic(y_clean, y_pred, len(popt))
        label = f"$ae^{{-bx}} + c$ \n $R^2$={r2:.2f}"# \n AIC={aic:.1f} \n BIC={bic:.1f}"

    elif fit_type == "power":
        model_func = power_func
        x_fit = x_clean[x_clean > 0]
        y_fit = y_clean[x_clean > 0]
        popt, _ = curve_fit(model_func, x_fit, y_fit, p0=(np.max(y_fit), 1.0), maxfev=10000)
        y_pred = model_func(x_fit, *popt)
        r2 = r2_score(y_fit, y_pred)
        aic, bic = compute_aic_bic(y_fit, y_pred, len(popt))
        x_clean, y_clean = x_fit, y_fit
        label = f"$ax^{{-b}}$ \n $R^2$={r2:.2f}"# \n AIC={aic:.1f} \n BIC={bic:.1f}"

    elif fit_type == "linear":
        model_func = linear_func
        reg = LinearRegression().fit(x_clean.reshape(-1, 1), y_clean)
        y_pred = reg.predict(x_clean.reshape(-1, 1))
        popt = (reg.coef_[0], reg.intercept_)
        r2 = r2_score(y_clean, y_pred)
        aic, bic = compute_aic_bic(y_clean, y_pred, 2)
        label = f"$ax + b$ \n $R^2$={r2:.2f}"#\n AIC={aic:.1f}\n BIC={bic:.1f}"

    elif fit_type == "log":
        mask_pos = (y_clean > 0)
        x_pos = x_clean[mask_pos]
        y_pos = y_clean[mask_pos]
        y_log = np.log(y_pos)
        reg = LinearRegression().fit(x_pos.reshape(-1, 1), y_log)
        y_pred_log = reg.predict(x_pos.reshape(-1, 1))
        y_pred = np.exp(y_pred_log)
        popt = (np.exp(reg.intercept_), -reg.coef_[0])

        r2 = r2_score(y_pos, y_pred)
        aic, bic = compute_aic_bic(y_pos, y_pred, 2) 
        x_clean, y_clean = x_pos, y_pos
        
        label = f"$ae^{{-bx}}$ log-fit \n $R^2$={r2:.2f}"#"\n AIC={aic:.1f}\n BIC={bic:.1f}"

    else:
        raise ValueError("fit_type must be one of: 'exp', 'power', 'linear', 'log'")
    
    if plot:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        if cmap:
            xy = np.vstack([x_clean, y_clean])
            z = gaussian_kde(xy)(xy)
            vmax = np.max(z); vmin = vmax; 
            # Plot with density-based color
            scatter = ax.scatter(x_clean, y_clean, c=z, s=dotsize, alpha=0.7, edgecolor=edgecolor, linewidths=linewidths, cmap=cmap)
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
            cbar.set_label(r'density', fontsize=DIM)
            cbar.ax.tick_params(labelsize=DIM)
        else:
            ax.scatter(x_clean, y_clean, s=dotsize, c=dotcolor, edgecolor=edgecolor, linewidths=linewidths, alpha=0.7, label="data")
        x_fit_line = np.linspace(min(x_clean), max(x_clean), 500)

        if fit_type in ["exp", "power"]:
            ax.plot(x_fit_line, model_func(x_fit_line, *popt),           'r-', lw=2, label=label)
        elif fit_type == "linear":
            ax.plot(x_fit_line, linear_func(x_fit_line, *popt),          'r-', lw=2, label=label)
        elif fit_type == "log":
            ax.plot(x_fit_line, popt[0] * np.exp(-popt[1] * x_fit_line), 'r-', lw=2, label=label)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.9), frameon=False)
        if cmap:
            pl.set_format(ax, pwr_x_min=-3, pwr_x_max=3, pwr_y_min=-2, pwr_y_max=2, axis_ticks = 'both', cbar = cbar, DIM = DIM)
        else:
            pl.set_format(ax, pwr_x_min=-3, pwr_x_max=3, pwr_y_min=-2, pwr_y_max=2, axis_ticks = 'both', cbar = None, DIM = DIM)

    return popt, r2, aic, bic
