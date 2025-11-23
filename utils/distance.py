import numpy as np

import matplotlib.pyplot as plt

from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.optimize import curve_fit
from scipy.stats import zscore, gaussian_kde, spearmanr, pearsonr
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

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
# Models
'''def logistic_func(x, L, k, x0, c):
    # 4-parameter logistic: lower asymptote c, amplitude L, slope k, inflection x0
    return L / (1 + np.exp(-k * (x - x0))) + c
'''
from scipy.special import expit

def logistic_func(x, A, d0, lam, c):
    # c + A / (1 + exp((x-d0)/lam))  ==  c + A * expit(-(x-d0)/lam)
    z = -(x - d0) / lam
    return c + A * expit(z)
    
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

def fit_and_plot(x, y, fit_type="exp", rm_quantiles=False, use_iqr=True, q=0.02, k=1.5, z_thresh=5, xlabel='eucl. distance (mm)', ylabel='TE',
                 cmap=None, edgecolor=None, linewidths=0.2, dotsize=10, dotcolor='tab:blue', plot=True, ax=None, 
                 outf : str = None, show_plot = True):

    # clean data from outliers
    if rm_quantiles==True and use_iqr==False:
        x_clean, y_clean = remove_outliers_quantiles(x, y, q=q)
    elif use_iqr==True:
        if rm_quantiles==True:
            print('both rm_quantiles and use_iqr are True. Choosing use_iqr option by default.')
        mask_x = iqr_filter(x, k)
        mask_y = iqr_filter(y, k)
        mask = mask_x & mask_y
        x_clean, y_clean = x[mask], y[mask]
    else:
        print('both rm_quantiles and use_iqr are False. Choosing z-scoring option to remove outliers.')
        x_clean, y_clean = remove_outliers(x, y, z_thresh=z_thresh)

    def compute_aic_bic(y_true, y_pred, num_params):
        residuals = y_true - y_pred
        rss = np.sum(residuals**2)
        n = len(y_true)
        aic = 2*num_params + n * np.log(rss / n)
        bic = num_params * np.log(n) + n * np.log(rss / n)
        return aic, bic
    
    # models
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
        
    elif fit_type in ["logit", "logistic"]:
        model_func = logistic_func
        tiny = 1e-12
        order = np.argsort(x_clean)
        x = x_clean[order]
        y = y_clean[order]
    
        # init
        c0 = float(np.nanpercentile(y, 10))
        A0 = max(tiny, float(np.nanmax(y) - c0))
        y_mid = c0 + 0.5 * A0
        idx_mid = int(np.argmin(np.abs(y - y_mid)))
        d0_0 = float(x[idx_mid]) if x.size else 0.0
    
        # lambda0 on length 10–90 (fallback on range/6)
        def interp_x_at_level(y_level):
            diffs = y - y_level
            sign = np.sign(diffs)
            cross = np.where(sign[:-1] * sign[1:] <= 0)[0]
            for k in cross:
                x0, y0, x1, y1 = x[k], y[k], x[k+1], y[k+1]
                if y1 == y0:
                    return x0
                t = (y_level - y0) / (y1 - y0)
                return x0 + t * (x1 - x0)
            return np.nan
    
        y10 = c0 + 0.10 * A0
        y90 = c0 + 0.90 * A0
        x10 = interp_x_at_level(y10)
        x90 = interp_x_at_level(y90)
        if np.isfinite(x10) and np.isfinite(x90) and (x90 > x10):
            lam0 = max(1e-6, (x90 - x10) / 4.394)
        else:
            xr = float(np.nanmax(x) - np.nanmin(x)) if x.size else 1.0
            lam0 = max(1e-6, xr / 6.0)
    
        p0 = (A0, d0_0, lam0, c0)
    
        # lam min bounds
        '''xr = float(np.nanmax(x_clean) - np.nanmin(x_clean)) if x_clean.size else 1.0
        lam_min = max(1e-6, xr / 1e3)
        bounds = ([0.0, -np.inf, lam_min, 0.0], [np.inf, np.inf, np.inf, 1.0])
    
        popt, _ = curve_fit(logistic_func, x_clean, y_clean, p0=p0, bounds=bounds, maxfev=20000)'''

        xr = float(np.nanmax(x_clean) - np.nanmin(x_clean)) if x_clean.size else 1.0
        lam_min = max(1e-6, xr / 1e3)
        lam0 = max(lam0, lam_min * (1.0 + 1e-12))  # p0[2] dentro ai bounds
        c_lower, c_upper = -np.inf, np.inf
        bounds = ([0.0, -np.inf, lam_min, c_lower],
                  [np.inf,  np.inf,  np.inf, c_upper])
        p0 = (A0, d0_0, lam0, c0)
        popt, _ = curve_fit(logistic_func, x_clean, y_clean, p0=p0, bounds=bounds, maxfev=20000)
        y_pred = logistic_func(x_clean, *popt)
        r2 = r2_score(y_clean, y_pred)
        aic, bic = compute_aic_bic(y_clean, y_pred, len(popt))
        label = r"$c+\frac{A}{1+\exp\!\left(\frac{x-d_0}{\lambda}\right)}$" + f"\n $R^2$={r2:.2f}"


    else:
        raise ValueError("fit_type must be one of: 'exp', 'power', 'linear', 'log'")
    
    if plot:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
            outf=None
            
        if cmap:
            xy = np.vstack([x_clean, y_clean])
            z = gaussian_kde(xy)(xy)
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
        elif fit_type in ["logit", "logistic"]:
            ax.plot(x_fit_line, model_func(x_fit_line, *popt), 'r-', lw=2, label=label)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.7), frameon=False)
        if cmap:
            pl.set_format(ax, pwr_x_min=-3, pwr_x_max=3, pwr_y_min=-2, pwr_y_max=2, axis_ticks = 'both', cbar = cbar, DIM = DIM)
        else:
            pl.set_format(ax, pwr_x_min=-3, pwr_x_max=3, pwr_y_min=-2, pwr_y_max=2, axis_ticks = 'both', cbar = None, DIM = DIM)

    
        if outf is not None:
            plt.savefig(outf, bbox_inches='tight')
            if not show_plot:
                plt.close()

    return popt, r2, aic, bic


def scatter_residuals(mat, lab, res_mat, res_lab, IC, res_IC, pval_IC, i, indices_, figsize=(9,3),
                      reg_line=False, ymin:float=None, ymax:float=None, outf : str = None, show_plot = True):

    ec_v  = mat[i]
    ic_v  = IC[i]
    res_e = res_mat[i]
    res_i = res_IC[i]
    p_IC  = pval_IC[i]
    
    cols_  = ['#E93423','#3333F6','#999AF8','#030062']

    fig,axs = plt.subplots(1,2,figsize=figsize)
    
    ax=axs[0]
    ax.scatter(ic_v[p_IC>=0.05],ec_v[p_IC>=0.05],s=150,c=cols_[2],edgecolor='white')
    ax.scatter(ic_v[p_IC<0.05], ec_v[p_IC<0.05], s=150,c=cols_[0],edgecolor='white')

    if reg_line:
        sns.regplot(x=ic_v, y=ec_v, scatter=False, ax=ax, line_kws=dict(color=colorz[4]))
    
    pl.set_format(ax=ax)
    ax.set_xlabel('IC')
    ax.set_ylabel(lab)
    ax.set_xlim(0,1)
    ax.set_title(fr'$\rho_{{sp}}={np.round(spearmanr(ec_v,ic_v)[0],2)}$'+
                 '\n'+fr'$\rho_{{p}}={np.round(pearsonr(ec_v,ic_v)[0],2)}$',y=1.1)
    
    ax=axs[1]
    ax.scatter(res_i[p_IC>=0.05],res_e[p_IC>=0.05],s=150,c=cols_[2],edgecolor='white')
    ax.scatter(res_i[p_IC<0.05], res_e[p_IC<0.05], s=150,c=cols_[0],edgecolor='white')

    if reg_line:
        sns.regplot(x=res_i, y=res_e, scatter=False, ax=ax, line_kws=dict(color=colorz[4]))
    
    pl.set_format(ax=ax)
    ax.set_xlabel('$\epsilon_{IC}$')

    ax.set_ylabel(f'$\epsilon_{{{lab}}}$')
    if ymin!=None and ymax!=None:
        ax.set_ylim(ymin,ymax)
    ax.set_title(fr'$\rho_{{sp}}={np.round(spearmanr(res_e,res_i)[0],2)}$'+
                 '\n'+fr'$\rho_{{p}}={np.round(pearsonr(res_e,res_i)[0],2)}$',y=1.1)
    
    #-------------------------------------------------------------------------------------------#
    
    fig.subplots_adjust(wspace=0.5, hspace=0.7)
    fig.suptitle(f'stim. chan. {indices_[i]}',y=1.4)

    if outf is not None:
        plt.savefig(outf, bbox_inches='tight')
        if not show_plot:
            plt.close()
        



#------------------------------------------------------------------------------------------------------------#
# COMPUTE PROBABILITIES

def distance_probabilities(meas_matrix, dist_matrix, which=1, N_bins=25, edges=None, density=False):
    """
    Always flattens inputs and excludes distances <= 0 and non-finite.
    Works with a single matrix/vector or a list of matrices/vectors.
    
    Definitions:
    i   : index of the distance bin; 
    n_i : number of nonzero connections in bin i; 
    N_i : total number of source-target pairs in bin i; 
    S   : sum_i n_i: total nonzero connections; 
    T   : sum_i N_i: total pairs overall
    
    Quantities (choose with 'which'):
      1) P(nonzero | d in bin i)                  = n_i / N_i
      2) P(nonzero)                               = S / T                         (scalar)
      3) P(nonzero ∧ d in bin i)                  = n_i / T
      4) P(d in bin i) or f(d)                    = N_i / T  (mass)  OR /Δi if density=True
      5) P(d in bin i | nonzero) or f(d|nonzero)  = n_i / S  (mass)  OR /Δi if density=True
      6) Enrichment(d)                            = (n_i/S) / (N_i/T) = P(nonzero|d)/P(nonzero)
      7) P(zero | d in bin i)                     = 1 - n_i / N_i

    Returns:
      - which in {1,3,4,5,6,7}: (vals_per_bin, bin_centers)
      - which == 2: scalar (or array of scalars if input is a list)
      - If input is a list: lists of arrays for per-bin quantities; shared edges if edges=None.
    """
    
    def _as_list(x):
        return x if isinstance(x, list) else [x]

    def _pick_dist(dm, k):
        return dm if not isinstance(dm, list) else dm[k]

    def _compute_edges(dm_list, nbins, provided_edges):
        if provided_edges is not None:
            e = np.asarray(provided_edges).ravel()
            if e.ndim != 1 or e.size < 2:
                raise ValueError("edges must be a 1D array with at least two values.")
            return e
        # Build edges from all distances (flattened), excluding d<=0 and non-finite
        D_all = np.concatenate([np.ravel(D) for D in dm_list])
        mask = np.isfinite(D_all) & (D_all > 0)
        if not np.any(mask):
            raise ValueError("No valid distances (>0 and finite) found to build edges.")
        dmin, dmax = D_all[mask].min(), D_all[mask].max()
        if dmax <= dmin:
            dmax = dmin + 1e-6
        return np.linspace(dmin, dmax, nbins + 1)

    def _counts_one(matr, D, edges_use):
        # Flatten, mask distances > 0 and finite, align meas -> same mask
        x = np.ravel(D).astype(float)
        y = (np.ravel(matr) != 0).astype(int)
        m = np.isfinite(x) & (x > 0)
        x, y = x[m], y[m]
        # Histogram counts
        N_i, _ = np.histogram(x, bins=edges_use)              # total pairs per bin
        n_i, _ = np.histogram(x, bins=edges_use, weights=y)   # nonzero per bin
        T = N_i.sum()
        S = n_i.sum()
        centers = 0.5 * (edges_use[:-1] + edges_use[1:])
        widths = np.diff(edges_use)
        return n_i.astype(float), N_i.astype(float), S, T, centers, widths

    # Normalize inputs to list for unified processing (always flatten internally)
    M_list = _as_list(meas_matrix)
    D_list = _as_list(dist_matrix)
    # If a single dist_matrix was provided for many meas, reuse it; else they must be aligned
    if len(D_list) == 1 and len(M_list) > 1:
        D_list = D_list * len(M_list)
    if len(D_list) != len(M_list):
        raise ValueError("meas_matrix and dist_matrix must have the same number of items.")

    # Shared edges if not provided
    edges_use = _compute_edges(D_list, N_bins, edges)

    if which == 2:
        # Global P(nonzero) per item (scalar)
        out = np.zeros(len(M_list), dtype=float)
        for k, M in enumerate(M_list):
            n_i, N_i, S, T, _, _ = _counts_one(M, D_list[k], edges_use)
            out[k] = (S / T) if T > 0 else np.nan
        return out if isinstance(meas_matrix, list) else out[0]

    vals_list, centers_list = [], []
    for k, M in enumerate(M_list):
        n_i, N_i, S, T, centers, widths = _counts_one(M, D_list[k], edges_use)

        if which == 1:      # P(nonzero | d)
            with np.errstate(invalid='ignore', divide='ignore'):
                with np.errstate(invalid='ignore', divide='ignore'):
                    vals = np.divide(n_i, N_i, out=np.zeros_like(n_i, dtype=float), where=N_i > 0)

        elif which == 3:    # P(nonzero ∧ d)
            vals = (n_i / T) if T > 0 else np.zeros_like(n_i)

        elif which == 4:    # P(d) or f(d)
            vals = (N_i / T) if T > 0 else np.zeros_like(N_i)
            if density:
                vals = vals / widths

    
        elif which == 5:    # P(d | nonzero) or f(d|nonzero)
            vals = (n_i / S) if S > 0 else np.zeros_like(n_i)
            if density and S > 0:
                vals = vals / widths

        elif which == 6:    # Enrichment(d) = P(nonzero|d)/P(nonzero)
            with np.errstate(invalid='ignore', divide='ignore'):
                p_nd = np.divide(n_i, N_i, out=np.zeros_like(n_i, dtype=float), where=N_i > 0)
            p_n = (S / T) if T > 0 else 0.0
            vals = (p_nd / p_n) if p_n > 0 else np.zeros_like(p_nd)

        elif which == 7:    # P(zero | d)
            with np.errstate(invalid='ignore', divide='ignore'):
                p_nd = np.divide(n_i, N_i, out=np.zeros_like(n_i, dtype=float), where=N_i > 0)
            vals = 1.0 - p_nd

        else:
            raise ValueError("Parameter 'which' must be an integer in {1,...,7}.")

        vals_list.append(vals)
        centers_list.append(centers)

    if isinstance(meas_matrix, list):
        return vals_list, centers_list
    else:
        return vals_list[0], centers_list[0]

# 1 Probabilità condizionata per distanza
# 2 Probabilità marginale di connessione
# 3 Joint (massa) su distanza
# 4 Distribuzione delle distanze tra TUTTE le coppie
# 5 Distribuzione delle distanze tra le connessioni
# 6 Enrichment per distanza:  sovra/sotto-rappresentata rispetto al caso null (P(nonzero∣d) ma normalizzata.)
# 7 Complementare per distanza

#------------------------------------------------------------------------------------------------------------#
from scipy.optimize import curve_fit

def fit_decay_length(d, P, bounds=([0.0, 0.0, 0.0], [1.0, np.inf, 1.0]), plot=False, color='tab:blue', ylabel='P(IC|dist)',
                     label_meas='IC', label_curve='exponential fit',
                     xmax=None, ymax=None, edgecolor='white', cmap=None, linewidths=0.2, figsize=(6,4),
                     ax=None, outf: str = None, show_plot=False):
    """
    Fit P(d) = a*exp(-b*d) + c on (d, P) and return lambda = 1/b (mm) and (a,b,c).
    If data are all ~0 or fit fails / b<=0: lambda=0, popt=(0,0,0); plot still shown with flat curve at 0.
    """
    d = np.asarray(d, float).ravel()
    P = np.asarray(P, float).ravel()
    m = np.isfinite(d) & np.isfinite(P)
    d, P = d[m], P[m]

    def func(x, a, b, c): return a * np.exp(-b * x) + c

    # --- trivial/degenerate case: all zeros or too few points ---
    tiny = 1e-12
    good_fit = True
    if (P.size < 3) or (np.nanmax(P) <= tiny):
        good_fit = False
        popt = (0.0, 0.0, 0.0)
        lambda_mm = 0.0
        P0 = 0.0
    else:
        # inits
        a0 = np.nanmax(P)
        c0 = max(0.0, np.nanmin(P)*0.25)
        b0 = 1.0 / max(1e-6, np.nanmedian(d[d > 0])) if np.any(d > 0) else 1.0

        # robust fit
        try:
            popt, _ = curve_fit(func, d, P, p0=(a0, b0, c0), bounds=bounds, maxfev=20000)
        except Exception:
            good_fit = False
            popt = (0.0, 0.0, 0.0)

        a, b, c = popt
        if not (np.isfinite(b) and b > 0):
            good_fit = False
            popt = (0.0, 0.0, 0.0)

        lambda_mm = (1.0 / b) if good_fit else 0.0
        P0 = a + c if good_fit else 0.0
    
    # --- plotting (always if plot=True; flat line if bad fit) ---
    if plot:
        y = np.copy(P); x = np.copy(d)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
            outf=None

        use_cbar = None
        if cmap:
            try:
                from scipy.stats import gaussian_kde
                xy = np.vstack([x, y])
                z = gaussian_kde(xy)(xy)
                scatter = ax.scatter(x, y, c=z, s=100, edgecolor=edgecolor, linewidths=linewidths,
                                     cmap=cmap, label=label_meas, alpha=0.7, zorder=2)
                use_cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
                use_cbar.set_label(r'density', fontsize=DIM)
                use_cbar.ax.tick_params(labelsize=DIM)
            except Exception:
                ax.scatter(x, y, c=color, s=100, edgecolor=edgecolor, linewidths=linewidths,
                           label=label_meas, alpha=0.7, zorder=2)
        else:
            ax.scatter(x, y, c=color, s=100, edgecolor=edgecolor, linewidths=linewidths,
                       label=label_meas, alpha=0.7, zorder=2)

        order = np.argsort(x)
        if good_fit:
            modeled_y = func(x[order], *popt)
            curve_col = '#E62A08'
        else:
            modeled_y = np.zeros_like(x[order])
            curve_col = '#666666'  # flat when no fit

        ax.plot(x[order], modeled_y, '-', color=curve_col, lw=8, label=label_curve, zorder=1)

        ax.set_ylabel(ylabel, fontsize=DIM)
        ax.set_xlabel('distance (mm)', fontsize=DIM)

        if ymax is not None:
            ax.set_ylim(0, ymax)
        if xmax is not None:
            ax.set_xlim(-0.2, xmax)
            ax.set_xticks(np.arange(0, xmax, 2))

        ax.legend(fontsize=DIM, ncol=1, loc='upper center', bbox_to_anchor=(0.5, 1.6),
                  labelspacing=0.4, handletextpad=0.8, handlelength=1., frameon=False)

        if cmap:
            pl.set_format(ax, axis_ticks='both', cbar=use_cbar, DIM=DIM)
        else:
            pl.set_format(ax, axis_ticks='both', cbar=None, DIM=DIM)

        if outf is not None:
            plt.savefig(outf, bbox_inches='tight')
            if not show_plot:
                plt.close()

    return float(lambda_mm), P0, popt  # (a,b,c)


from scipy.optimize import curve_fit
#for probability
def fit_sigmoid_length(d, P,
                       bounds=([0.0, -np.inf, 1e-6, 0.0], [np.inf,  np.inf,  np.inf, 1.0]),
                       plot=False, color='tab:blue', ylabel='P(IC|dist)',
                       label_meas='IC', label_curve='logistic fit',
                       xmax=None, ymax=None, edgecolor='white', cmap=None, linewidths=0.2,
                       figsize=(6,4), ax=None, outf: str = None, show_plot=False,
                       sigma=None):
    """
    Fit logistico: P(d) = c + A / (1 + exp((d - d0) / lambda))
    Ritorna: lambda_mm (= lambda), P0 (= P(d=0) del modello), popt = (A, d0, lambda, c)

    - bounds default: A>=0, lambda>0, c in [0,1]; d0 libero
    - P0 è il valore di modello a d=0 (come nella tua exp)
    - se il fit fallisce o A ~ 0: lambda_mm = 0, popt=(0,0,0,0), curva piatta
    """
    d = np.asarray(d, float).ravel()
    P = np.asarray(P, float).ravel()
    m = np.isfinite(d) & np.isfinite(P)
    d, P = d[m], P[m]
    if sigma is not None:
        sigma = np.asarray(sigma, float).ravel()[m]

    def func(x, A, d0, lam, c):
        return c + A / (1.0 + np.exp((x - d0) / lam))

    tiny = 1e-12
    good_fit = True

    # --- casi degeneri ---
    if (P.size < 3) or (np.nanmax(P) - np.nanmin(P) <= tiny):
        good_fit = False
        popt = (0.0, 0.0, 0.0, 0.0)
        lambda_mm = 0.0
        P0 = 0.0
    else:
        # ordina per distanza (aiuta per inits robusti)
        order = np.argsort(d)
        x = d[order]; y = P[order]
        s = None if sigma is None else sigma[order]

        # inits robusti:
        c0 = float(np.nanpercentile(y, 10))                      # plateau lontano approx
        A0 = max(tiny, float(np.nanmax(y) - c0))                 # ampiezza positiva
        y_mid = c0 + 0.5 * A0                                    # livello mediano
        # stima d0: distanza più vicina al livello mediano
        idx_mid = int(np.argmin(np.abs(y - y_mid)))
        d0_0 = float(x[idx_mid]) if x.size else 0.0

        # lambda0: da larghezza 10–90 (se disponibile), altrimenti fallback
        def interp_x_at_level(y_level):
            # lineare tra punti adiacenti
            diffs = y - y_level
            sign = np.sign(diffs)
            cross = np.where(sign[:-1] * sign[1:] <= 0)[0]  # cambi di segno
            for k in cross:
                x0, y0, x1, y1 = x[k], y[k], x[k+1], y[k+1]
                if y1 == y0:  # segmento piatto
                    return x0
                t = (y_level - y0) / (y1 - y0)
                return x0 + t * (x1 - x0)
            return np.nan

        y10 = c0 + 0.10 * A0
        y90 = c0 + 0.90 * A0
        x10 = interp_x_at_level(y10)
        x90 = interp_x_at_level(y90)

        if np.isfinite(x10) and np.isfinite(x90) and (x90 > x10):
            W1090 = x90 - x10
            lam0 = max(1e-6, W1090 / 4.394)  # W10–90 ≈ 4.394 * lambda
        else:
            xr = float(np.nanmax(x) - np.nanmin(x))
            lam0 = max(1e-6, xr / 6.0)       # fallback ragionevole

        p0 = (A0, d0_0, lam0, c0)

        # fit non lineare (con pesi opzionali)
        try:
            popt, _ = curve_fit(func, x, y, p0=p0, bounds=bounds,
                                sigma=s, absolute_sigma=(s is not None), maxfev=20000)
            A, d0, lam, c = popt
            if not (np.isfinite(A) and np.isfinite(d0) and np.isfinite(lam) and np.isfinite(c) and lam > 0 and A >= 0):
                good_fit = False
        except Exception:
            good_fit = False
            popt = (0.0, 0.0, 0.0, c0)

        lambda_mm = float(popt[2]) if good_fit else 0.0
        # P(d=0)
        P0 = float(func(0.0, *popt)) if good_fit else 0.0

    # --- plotting ---
    if plot:
        xplot = np.copy(d); yplot = np.copy(P)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        use_cbar = None
        if cmap:
            try:
                from scipy.stats import gaussian_kde
                xy = np.vstack([xplot, yplot])
                z = gaussian_kde(xy)(xy)
                scatter = ax.scatter(xplot, yplot, c=z, s=1, edgecolor=edgecolor, linewidths=linewidths,
                                     cmap=cmap, label=label_meas, alpha=0.7, zorder=1)
                use_cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
                use_cbar.set_label(r'density', fontsize=DIM)
                use_cbar.ax.tick_params(labelsize=DIM)
            except Exception:
                ax.scatter(xplot, yplot, c=color, s=1, edgecolor=edgecolor, linewidths=linewidths,
                           label=label_meas, alpha=0.7, zorder=1)
        else:
            ax.scatter(xplot, yplot, c=color, s=1, edgecolor=edgecolor, linewidths=linewidths,
                       label=label_meas, alpha=0.7, zorder=1)

        order = np.argsort(xplot)
        if good_fit:
            modeled_y = func(xplot[order], *popt)
            curve_col = '#0B62A9'
        else:
            modeled_y = np.zeros_like(xplot[order])
            curve_col = '#666666'

        ax.plot(xplot[order], modeled_y, '-', color=curve_col, lw=8, label=label_curve, zorder=2)

        ax.set_ylabel(ylabel, fontsize=DIM)
        ax.set_xlabel('distance (mm)', fontsize=DIM)
        if ymax is not None:
            ax.set_ylim(0, ymax)
        if xmax is not None:
            ax.set_xlim(-0.2, xmax)
            ax.set_xticks(np.arange(0, xmax, 2))

        ax.legend(fontsize=DIM, ncol=1, loc='upper center', bbox_to_anchor=(0.5, 1.6),
                  labelspacing=0.4, handletextpad=0.8, handlelength=1., frameon=False)

        try:
            pl.set_format(ax, axis_ticks='both', cbar=use_cbar if cmap else None, DIM=DIM)
        except Exception:
            pass

        if ax is None:
            if outf:
                ax.savefig(outf, bbox_inches='tight')
                if not show_plot:
                    plt.close()
            else:
                plt.show()

    return float(lambda_mm), P0, popt  # popt = (A, d0, lambda, c)
        