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


#import load_EC as lec
#import correlations as cc

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

def set_format(ax, axis_ticks = 'both', pwr_x_min=-2, pwr_x_max=2, pwr_y_min=-2, pwr_y_max=2,  cbar = None, pwr_cbar_min=-1, pwr_cbar_max=1, dx_cbar = 0.02, dy_cbar = 0.1, dx= 15/72, dy = 15/72, DIM = None):

    import seaborn as sns
    
    sns.despine(ax=ax, trim=False)
    ax.set_facecolor('none')
    
    # - - -  TICKS
    if DIM is not None:
        ax.tick_params(axis=axis_ticks, which='major', labelsize=DIM)
    else:
        ax.tick_params(axis=axis_ticks, which='major')
    
    # - - -  FORMATTER x axis
    formatter_x = ScalarFormatter(useMathText=True)   
    formatter_x.set_scientific(True)
    formatter_x.set_powerlimits((pwr_x_min, pwr_x_max))
    ax.xaxis.set_major_formatter(formatter_x)
    if DIM is not None:
        ax.xaxis.offsetText.set_fontsize(DIM)
    
    from matplotlib.transforms import ScaledTranslation
    offset = ScaledTranslation(dx, dy, ax.figure.dpi_scale_trans)
    ax.xaxis.offsetText.set_transform(ax.xaxis.offsetText.get_transform() + offset)

    # - - -  FORMATTER y axis
    formatter_y = ScalarFormatter(useMathText=True)    
    formatter_y.set_scientific(True) 
    formatter_y.set_powerlimits((pwr_y_min, pwr_y_max))
    ax.yaxis.set_major_formatter(formatter_y);
    if DIM is not None:
        ax.yaxis.offsetText.set_fontsize(DIM)
    
    if cbar:
        # - - -  FORMATTER cbar
        formatter_cbar = ScalarFormatter(useMathText=True)   
        formatter_cbar.set_scientific(True)
        formatter_cbar.set_powerlimits((pwr_cbar_min, pwr_cbar_max))
        cbar.ax.yaxis.set_major_formatter(formatter_cbar); 
        cbar.ax.xaxis.set_major_formatter(formatter_cbar); 
        if DIM is not None:
            cbar.ax.yaxis.offsetText.set_fontsize(DIM)
            cbar.ax.xaxis.offsetText.set_fontsize(DIM)

        cbar.formatter = formatter_cbar
        cbar.update_ticks()
        
        # Move the offset text to the top of the colorbar
        cbar_offset = ScaledTranslation(dx_cbar, dy_cbar, cbar.ax.figure.dpi_scale_trans)
        cbar.ax.yaxis.offsetText.set_transform(cbar.ax.yaxis.offsetText.get_transform() + cbar_offset)

def despine_all(fig):
    for ax in fig.axes:
        for side in ("left", "right", "top", "bottom"):
            ax.spines[side].set_visible(False)
        ax.tick_params(left=False, right=False, top=False, bottom=False)

def despine_ax(ax):
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(left=False, right=False, top=False, bottom=False)

    
#=================================================================================================#
# plot matrix – aspect='auto'

def plot_mat_aspect(mat, vmin=None, vmax=None, cmap='viridis', title=None, xlabel='target', ylabel='source', ticksize=15, 
                    tick_rotation=45, cbarlabel=None, invert_y: bool = True, xticklabels: list = None, yticklabels: list = None, 
                    figsize=(15, 10), ax = None, outf: str = None, show_plot: bool = True):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure() 
        outf=None
        
    if vmin is None and vmax is None:
        vmin, vmax = np.percentile(mat, [5, 97.5])
        
    im = ax.imshow(mat, aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    if xticklabels is not None:
        ax.set_xticks(np.arange(len(xticklabels)))
        ax.set_xticklabels(xticklabels, rotation=tick_rotation, fontsize=ticksize)

    if yticklabels is not None:
        ax.set_yticks(np.arange(len(yticklabels)))
        ax.set_yticklabels(yticklabels, fontsize=ticksize)

    if invert_y:
        ax.invert_yaxis()

    if cbarlabel is not None:
        fig.colorbar(im, ax=ax, label=cbarlabel)

    ax.set_facecolor('none')
    
    if outf is not None:
        plt.savefig(outf, bbox_inches='tight')
        if not show_plot:
            plt.close()

#=================================================================================================#
#                                       a) channel map                                            # 
#=================================================================================================#

# plot array channels
def plot_channel_map(pos_,  channel, def_chans,      indices,  rec_list=None,
                       cmap=None, cbar_label='', title=None, DIM=None, 
                       chsize=4.5,    ch_color='lightgrey', ch_lw=0.1,  ch_ec='white',
                       recsize=1350,  rec_color='tab:blue', rec_lw=0.1, rec_ec='white',  rec_text=True, text_dim=18,
                       starsize=1350, star_color='k',       star_lw=0.1, star_ec='white', starsize_center = None, 
                       ncol_leg=2, y_leg=1.2, y_titl=None, figsize=(5, 3), ax=None, outf : str = None, show_plot = True):
    # optimized for small size plots also
    pos = pos_[:,:]/1000
    
    if ax is None:
        fig,ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure() 
        outf=None
    
    # Full map of recording channels 
    ax.scatter(pos[:, 0], pos[:, 1], s=chsize, lw=ch_lw, marker='s', color=ch_color, edgecolor=ch_ec, zorder=0, label='rec. channels')

    # Highlight specific recording channels
    if rec_list is not None:
        ax.scatter(pos[rec_list, 0], pos[rec_list, 1], s=recsize, c=rec_color, marker='o',  edgecolor=rec_ec, lw=rec_lw, zorder=0) #label=f'rec. channels'

    # stimulation channels
    if cmap is not None and np.ndim(star_color) > 0 and not isinstance(star_color, str):
        vmin, vmax = np.percentile(star_color, [2.5, 97.5])
    else:
        vmin, vmax = (None, None)
        
    scatter = ax.scatter(pos[indices, 0], pos[indices, 1], cmap=cmap, vmin=vmin, vmax=vmax, c=star_color, lw=star_lw,
                         edgecolor=star_ec, s=starsize, marker='*', alpha=1, zorder=1, label='stim. channels')
    if starsize_center is not None:
        ax.scatter(pos[indices, 0], pos[indices, 1], color='white', edgecolor='white', lw=ch_lw, 
                   s=starsize_center, marker='*', alpha=1, zorder=3)
    # cmap
    if cmap is not None:
        cbar = fig.colorbar(scatter, ax=ax, label=cbar_label)
    else:
        pass
            
    # labels for rec channels 
    import matplotlib.patheffects as pe
    if rec_list is not None and rec_text is True:
        a = 0
        for i_rec in rec_list:
            x = pos[channel == channel[i_rec], 0][0];    y = pos[channel == channel[i_rec], 1][0]
    
            dx, dy = -420, 180
                
            ax.text(x + dx/1000, y + dy/1000, str(i_rec),fontdict=dict(color='k', alpha=1, size=text_dim),
                   path_effects=[pe.withStroke(linewidth=3, foreground='white')])
            a += 1
    
    ax.set_xlabel('x (mm)');    ax.set_ylabel('y (mm)')
    set_format(ax, axis_ticks='both', cbar=None, DIM=DIM)

    ax.legend(ncol=ncol_leg, loc='upper center', bbox_to_anchor=(0.5, y_leg),
              labelspacing=0.4, handletextpad=0.8, handlelength=1., frameon=False)
    
    if title:
        if y_titl is None:
            y_titl=y_leg
        ax.set_title(title, y=y_titl)

    fig.set_facecolor('none')
    
    if outf is not None:
        plt.savefig(outf, bbox_inches='tight')
        if not show_plot:
            plt.close()

#=================================================================================================#
#                   b) scatter one-rec channel response to all stim. sites
#=================================================================================================#

# sort stim channels with increasing distance from rec. channel i
def sort_units_with_distance(i, channel, pos, def_units, def_chans):
    
    def eucl_dist(a1, a2):
        return np.sqrt( (a1[0]-a2[0])*(a1[0]-a2[0]) + (a1[1]-a2[1])*(a1[1]-a2[1]) )
    
    ch = channel[i]

    # dist between ch and stim. chans.
    distances = np.zeros(len(def_units))
    for idx in range(len(def_units)):
        id1 = np.where(channel==def_chans[idx])[0][0]
        id2 = np.where(channel==ch)[0][0]
        distances[idx] = eucl_dist(pos[id1,:],pos[id2,:])
    
    # stim. channels ordered in increasing distance
    L = np.argsort(distances)
    return L[::-1], distances[L[::-1]]

#--------------------------------------------------------------------------------------------------#
# get the color based on the motif
def get_col_post_auto(i_rec, sorted_units, counts_dict, bin_sz, pre_start, pre_stop, post_start, post_stop, col_u, col_d, col_n,
                      pre_guard=0.12, peak_win=0.05, late_start=0.15, k_peak=3.0, k_supp=3.0, q=None,
                      min_post_peak=1.0, min_dp=1.0, return_debug=False):

    u0, Ntr = sorted_units[0], len(counts_dict[sorted_units[0]])
    T = counts_dict[u0][0].shape[0]; t = np.arange(T) * bin_sz

    pre_base  = (t >= 0.2) & (t <= pre_stop - pre_guard)
    pre_peak  = (t >= (pre_stop - pre_guard - peak_win)) & (t < (pre_stop - pre_guard))
    post_peak = (t >= post_start) & (t < post_start + peak_win)
    post_late = (t >= post_start + late_start) & (t < post_stop)

    dpeak, dsupp, ppost = [], [], []
    for u in sorted_units:
        X = np.stack([counts_dict[u][n][:, i_rec] for n in range(Ntr)], axis=0)
        m = X.mean(0)
        mpre = m[pre_base].mean()
        pk_pre  = m[pre_peak].max()
        pk_post = m[post_peak].max()
        dpeak.append(pk_post - pk_pre)
        dsupp.append(mpre - m[post_late].mean())
        ppost.append(pk_post)

    dpeak, dsupp, ppost = np.array(dpeak), np.array(dsupp), np.array(ppost)

    if q is not None:
        thr_peak = np.quantile(dpeak, q); thr_supp = np.quantile(dsupp, q)
    else:
        mad = lambda x: 1.4826*np.median(np.abs(x - np.median(x))) + 1e-12
        thr_peak = np.median(dpeak) + k_peak*mad(dpeak)
        thr_supp = np.median(dsupp) + k_supp*mad(dsupp)

    cols, labels = [], []
    for dp, ds, pk in zip(dpeak, dsupp, ppost):
        exc  = (dp > thr_peak) and (pk >= min_post_peak) and (dp >= min_dp)
        supp = ds > thr_supp
        if exc and supp: cols.append(col_u); labels.append("exc_then_supp")
        elif supp:       cols.append(col_d); labels.append("suppressed")
        else:            cols.append(col_n); labels.append("baseline")

    if return_debug:
        return cols, labels, dict(dpeak=dpeak, dsupp=dsupp, ppost=ppost, thr_peak=thr_peak, thr_supp=thr_supp,
                                  min_post_peak=min_post_peak, min_dp=min_dp)
    return cols

#--------------------------------------------------------------------------------------------------#
# scatter with color-codes 
def plot_scatter_dots(KS, i_rec, col_post, cult_lab, _channel, _pos, _def_units, _def_chans, dotsize=50, show_corr = True, corr='spearman', xmin=-0.1, xmax=3.5, ymin=-0.01, ymax=0.8, DIM=None, xlabel = 'distance (mm)\n from stim. channel', ylabel = 'pert. effect (IC)', lw=0.1, ec='white', regcolor='k', reg_line=True, title=f'rec. chan.', figsize=(3.5,3), ax=None, outf=None, show_plot=True):
    from scipy.stats import pearsonr, spearmanr
    
    if ax is None:
        fig,ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure() 
        outf=None
        
    sorted_units, sorted_d = sort_units_with_distance(i_rec, _channel, _pos/1000, _def_units, _def_chans)
    
    x = sorted_d
    y = KS[sorted_units,i_rec]
    
    if reg_line:
        sns.regplot(x=x, y=y, ax=ax, scatter=False, color=regcolor, line_kws={'linewidth': 1,  'zorder': 0})
        band = ax.collections[-1]
        band.set_alpha(0.05)
        band.set_zorder(0)
    ax.scatter(x, y, s = dotsize, c=col_post, edgecolor=ec, linewidths=lw, zorder=1)
    
    if show_corr:
        if corr=='pearson':
            c = np.round(pearsonr(x,y)[0],2)
        else:
            c = np.round(spearmanr(x,y)[0],2)
        ax.set_title((title + '\n' if title is not None else '') + rf'$\rho={c}$')
    elif title is not None:
        ax.set_title(title)
        
    ax.set_ylim(ymin,ymax)
    ax.set_xlim(xmin,xmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    set_format(ax=ax, DIM=DIM, pwr_x_min=-2, pwr_y_min=-2)

    fig.set_facecolor('none')
    
    if outf is not None:
        plt.savefig(outf, bbox_inches='tight')
        if not show_plot:
            plt.close()



#=================================================================================================#
#                     c2) traces / plot trial average
#=================================================================================================#

# plot trial average
def plot_traces_zoom(_i_rec, _sorted_units, _indices, _counts_dict, _col_post, col_n, bin_sz=0.01, n_ampli=1, Ntrials=200, font_yticks=10, x_margins=0.05, n_pulses=24, IPI_s = 0.005, lw=1, pre_start=0.49, pre_stop=0.99, post_start=1+0.004, post_stop=1.504, stim0=1, stimEnd=1+0.003, font_DIM=15, figsize=(5,6), title=None, ax=None, outf:str=None, show_plot=False):

    
    d_pre  = stim0 - pre_stop
    d_post = post_start - stimEnd
    
    if ax is None:
        fig,ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure() 
        outf=None

    for i,i_unit in enumerate(_sorted_units):
        # put together trials for each stimulation from i_unit, for fixed _i_rec rec-channel
        trials = (_counts_dict[i_unit][0])[:,_i_rec]
        for n in range(1,Ntrials):
            trials = np.vstack([ trials, (_counts_dict[i_unit][n])[:,_i_rec]])

        # average spike counts across trials, for each i_unit
        N    = trials.shape[0]
        mean = trials.sum(axis=0) / N
        var  = (trials*trials).sum(axis=0) / N - mean**2
        var  = np.maximum(var, 0.0)          # evita piccoli negativi numerici
        std  = np.sqrt(var)
        t    = np.arange(len(mean))*bin_sz  

        # pulses of stimulation (the exact ones)
        for k in range(n_pulses):
            ax.plot([stim0+k*IPI_s,stim0+k*IPI_s], [i,i+0.3], color='black',lw=0.1)
        m_pre   = np.mean(mean[(t>pre_start) & (t<=pre_stop)])
        max_pre = np.max(np.abs(mean[(t>pre_start) & (t<=pre_stop)]))
        max_pre = max(max_pre, 1e-12)
        
        # pre and post traces
        y_pre  = (mean[(t>pre_start) & (t<=pre_stop)]   - m_pre) / max_pre*0.2 * n_ampli
        y_post = (mean[(t>=post_start) & (t<post_stop)] - m_pre) / max_pre*0.2 * n_ampli
        ax.plot(t[(t>pre_start) & (t<=pre_stop)] + d_pre - 0.01,   y_pre  + i, color=col_n,       lw=lw, alpha=1)
        ax.plot(t[(t>=post_start) & (t<post_stop)]-d_post + 0.01, y_post + i, color=_col_post[i], lw=lw, alpha=1)        

        # zero amplitude of each trial
        ax.plot([pre_start+ d_pre - 0.01,post_stop-d_post + 0.01], [i,i], color='black',lw=0.1)
        
    despine_ax(ax)

    import matplotlib.transforms as mtransforms
    # --- scale bars at the same height of the x-axes (y=0 in axes coords)
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    y_bar = 0#-0.04
    y_txt = -0.15
    
    ax.plot([pre_start+ d_pre - 0.01, pre_stop+ d_pre - 0.01],   [y_bar, y_bar], transform=trans, color='k', lw=0.9, clip_on=False)
    ax.plot([post_start-d_post + 0.01, post_stop-d_post + 0.01], [y_bar, y_bar], transform=trans, color='k', lw=0.9, clip_on=False)

    ax.text((pre_start+pre_stop+2*(d_pre - 0.01))/2,   y_txt, f'pre-stim.',
            transform=trans, ha='center', va='top', fontsize=font_DIM, clip_on=False)
    ax.text((post_start+post_stop +2*(-d_post + 0.01))/2, y_txt, f'post-stim.',
            transform=trans, ha='center', va='top', fontsize=font_DIM, clip_on=False)
    
    ax.text((pre_start+pre_stop+2*(d_pre - 0.01))/2,   y_txt+0.1, f'{(pre_stop-pre_start)*1000:.0f} ms',
            transform=trans, ha='center', va='top', fontsize=plt.rcParams['legend.fontsize'], clip_on=False)
    ax.text((post_start+post_stop +2*(-d_post + 0.01))/2, y_txt+0.1, f'{(post_stop-post_start)*1000:.0f} ms',
            transform=trans, ha='center', va='top', fontsize=plt.rcParams['legend.fontsize'], clip_on=False)
    ax.text((pre_stop+post_start+d_pre-d_post)/2, y_bar, '*', transform=trans, ha='center', va='center', fontsize=font_DIM, clip_on=False)
    

    # y-ticks: stim units
    ax.set_yticks(np.arange(len(_indices)))
    ax.set_yticklabels(_indices[_sorted_units], fontsize=font_yticks)
    ax.set_ylabel('stim. channel',fontsize=font_DIM)
    
    ax.margins(x=x_margins)
    
    # no x-ticks
    ax.set_xticks([])
    
    ax.set_facecolor('none')
    if title is not None:
        ax.set_title(title+'\nrec. chan '+str(_i_rec))
    else:
        ax.set_title('channel '+str(_i_rec)+ ' response')
    
    if outf:
        plt.savefig(outf, bbox_inches='tight')
        if show_plot==False:
            plt.close()

#=================================================================================================#
#                                      d) scatter plots
#=================================================================================================#

def plot_scatter2(ic_mat, ec_mat, zeroExp=-11, log=False, xlabel='IC', ylabel='EC', title=None, dotsize=0.1, DIM=None, cmap=None, edgecolor=None, linewidths=0.2, regcolor='tab:red', dotcolor='tab:blue', reg_line=False, show_corr=True, corr='spearman', xmin=None, xmax=None, ymin=None, ymax=None, ax=None, alpha=1, figsize=(3.5,3), outf=None, show_plot=False):
    import seaborn as sns
    from scipy.stats import pearsonr, spearmanr, gaussian_kde

    ic_vec = np.asarray(ic_mat).ravel()
    ec0 = np.asarray(ec_mat).ravel()

    if log:
        ec_vec = np.log10(ec0 + 10**zeroExp)
        mask = (ec0 != 0) & np.isfinite(ec_vec) & np.isfinite(ic_vec)
        x, y = ic_vec[mask], ec_vec[mask]
    else:
        ec_vec = ec0
        mask = np.isfinite(ec_vec) & np.isfinite(ic_vec)
        x, y = ic_vec[mask], ec_vec[mask]

    dc = dotcolor
    if hasattr(dc, "__len__") and not isinstance(dc, str):
        dc = np.asarray(dc, dtype=object)[mask]

    if ymin is not None and ymax is not None:
        m2 = (y >= ymin) & (y < ymax)
        x, y = x[m2], y[m2]
        if hasattr(dc, "__len__") and not isinstance(dc, str): dc = dc[m2]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
        outf = None

    if cmap:
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        scatter = ax.scatter(x, y, c=z, s=dotsize, edgecolor=edgecolor, linewidths=linewidths, cmap=cmap, alpha=alpha)
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
        cbar.set_label(r'density')
    else:
        ax.scatter(x, y, s=dotsize, c=dc, edgecolor=edgecolor, linewidths=linewidths, alpha=alpha)

    if reg_line:
        sns.regplot(x=x, y=y, ax=ax, scatter=False, color=regcolor, line_kws={'linewidth': 1})

    if show_corr and len(ic_vec) > 1:
        if corr=='pearson':
            c = np.round(pearsonr(ic_vec,ec_vec)[0],2)
        else:
            c = np.round(spearmanr(ic_vec,ec_vec)[0],2)
        ax.set_title((title + '\n' if title is not None else '') + rf'$\rho={c}$')
    elif title is not None:
        ax.set_title(title)

    if ymin is not None and ymax is not None: ax.set_ylim(ymin, ymax)
    if xmin is not None and xmax is not None: ax.set_xlim(xmin, xmax)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)

    if cmap:
        set_format(ax, pwr_x_min=-3, pwr_x_max=3, pwr_y_min=-2, pwr_y_max=2, axis_ticks='both',
                      cbar=cbar, pwr_cbar_min=-2, pwr_cbar_max=2, DIM=DIM)
    else:
        set_format(ax, pwr_x_min=-3, pwr_x_max=3, pwr_y_min=-2, pwr_y_max=2, axis_ticks='both', cbar=None, DIM=DIM)

    fig.set_facecolor('none')
    if outf is not None:
        plt.savefig(outf, bbox_inches='tight')
        if not show_plot: plt.close()


#--------------------------------------------------------------------------------------------------#

def plot_scatter_colors(sort_mat, y_mat, k_dict, _channel, _pos, _def_units, _def_chans, bin_sz=0.01, log=False, lw=0.05, regcolor='tab:red', xlabel='', ylabel='', dotsize=6, ec='grey', title=None, show_corr=True, corr='spearman', alpha=1, cmap=None, DIM=None, reg_line=True, n_chans=100, pre_start=0.4, pre_stop=0.9, post_start=1.12, post_stop=1.12+0.5, col_u='tab:red', col_d='tab:blue', col_n='lightgrey', pre_guard=0.12, peak_win=0.05, late_start=0.15, q=0.2, min_post_peak=-0.02, min_dp=-0.02, xmin=None, xmax=None, ymin=None, ymax=None, ax=None, figsize=(3.5,3), outf=None, show_plot=False):
    ##
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
        outf = None
        
    in_sign_bin = np.sum(sort_mat,axis=0)
    L           = np.argsort(in_sign_bin)[::-1]
    
    colors_post = np.array([], dtype=object)
    dists  = np.array([], float)
    y_vals = np.array([], float)
    for i in range(n_chans):
        i_rec   = int(np.where(in_sign_bin==in_sign_bin[L[i]])[0][0])
        sorted_units, sorted_d = sort_units_with_distance(i_rec, _channel, _pos/1000, _def_units, _def_chans)

        col_post_auto = get_col_post_auto(i_rec=i_rec, sorted_units=sorted_units, counts_dict=k_dict[bin_sz], bin_sz=bin_sz,
                            pre_start=pre_start, pre_stop=pre_stop, post_start=post_start, post_stop=post_stop, col_u=col_u, col_d=col_d, col_n=col_n,
                            pre_guard=pre_guard, peak_win=peak_win, late_start=late_start, q=q, min_post_peak=min_post_peak, min_dp=min_dp)
        
        colors_post = np.concatenate([colors_post, np.asarray(col_post_auto, dtype=object)])
        dists       = np.concatenate([dists, sorted_d])
        y_vals      = np.concatenate([y_vals, y_mat[sorted_units,i_rec]])

    xlabel  = 'distance (mm)\n from stim. channel'
    
    if log==False:
        plot_scatter2(dists, y_vals, -8, xlabel=xlabel, ylabel=ylabel, dotsize=dotsize, edgecolor=ec, linewidths=lw, alpha=alpha, 
                      show_corr=show_corr, corr=corr,
                      cmap=cmap, dotcolor=colors_post, figsize=figsize, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, log=log, reg_line=reg_line,  
                      ax = ax, DIM=DIM, regcolor=regcolor, title=title)
    else:
        plot_scatter2(dists, y_vals, -8, xlabel=xlabel, ylabel=ylabel, dotsize=dotsize, edgecolor=ec, linewidths=lw, alpha=alpha, 
                      show_corr=show_corr, corr=corr,
                      cmap=cmap, dotcolor=colors_post, figsize=figsize, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, log=log, reg_line=reg_line,  
                      ax = ax, DIM=DIM, regcolor=regcolor, title=title)

    fig.set_facecolor('none')
    if outf is not None:
        plt.savefig(outf, bbox_inches='tight')
        if not show_plot: plt.close()
        


