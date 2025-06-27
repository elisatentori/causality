import numpy as np

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.stats import zscore, gaussian_kde
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
