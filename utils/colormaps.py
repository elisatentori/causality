import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgba

# ================================================================================================================ #
# - - - -           continuous cmap           - - - - #


def generate_continuous_cmap(start_color, end_color):

    start_rgba = np.array(to_rgba(start_color))
    end_rgba = np.array(to_rgba(end_color))
    cmap_data = {'red':   [(0.0, start_rgba[0], start_rgba[0]),
                           (1.0, end_rgba[0], end_rgba[0])],

                 'green': [(0.0, start_rgba[1], start_rgba[1]),
                           (1.0, end_rgba[1], end_rgba[1])],

                 'blue':  [(0.0, start_rgba[2], start_rgba[2]),
                           (1.0, end_rgba[2], end_rgba[2])],

                 'alpha': [(0.0, start_rgba[3], start_rgba[3]),
                           (1.0, end_rgba[3], end_rgba[3])]}
    cmap = LinearSegmentedColormap('custom_cmap', cmap_data)
    return cmap


# ================================================================================================================ #
# - - - -           hotcold cmap           - - - - #


def create_cmaphot():
    
    cmaphot = np.array([
        [71/255, 93/255, 172/255],      
        [0.3, 0.6, 1],  
        [1, 1, 1],        
        [1, 0.8, 0],    
        [225/255,43/255,42/255]        
    ])
    len_colorbar = 256
    r = np.interp(np.linspace(0, 1, len_colorbar), np.linspace(0, 1, cmaphot.shape[0]), cmaphot[:, 0])
    g = np.interp(np.linspace(0, 1, len_colorbar), np.linspace(0, 1, cmaphot.shape[0]), cmaphot[:, 1])
    b = np.interp(np.linspace(0, 1, len_colorbar), np.linspace(0, 1, cmaphot.shape[0]), cmaphot[:, 2])
    coldhot_cmap = LinearSegmentedColormap.from_list('custom_cmap', list(zip(np.linspace(0, 1, len_colorbar), list(zip(r, g, b)))), N=len_colorbar)
    
    return coldhot_cmap


# ================================================================================================================ #
# - - - -           hotcold cmap reverse          - - - - #


def create_cmaphot_r():
    
    cmaphot2 = np.array([
        [71/255, 93/255, 172/255],      
        [0.3, 0.6, 1],  
        [1, 1, 1],        
        [1, 0.8, 0],    
        [225/255,43/255,42/255]        
    ])
    cmaphot = cmaphot2[::-1]

    len_colorbar = 256
    r = np.interp(np.linspace(0, 1, len_colorbar), np.linspace(0, 1, cmaphot.shape[0]), cmaphot[:, 0])
    g = np.interp(np.linspace(0, 1, len_colorbar), np.linspace(0, 1, cmaphot.shape[0]), cmaphot[:, 1])
    b = np.interp(np.linspace(0, 1, len_colorbar), np.linspace(0, 1, cmaphot.shape[0]), cmaphot[:, 2])
    coldhot_cmap = LinearSegmentedColormap.from_list('custom_cmap', list(zip(np.linspace(0, 1, len_colorbar), list(zip(r, g, b)))), N=len_colorbar)
    
    return coldhot_cmap


# ================================================================================================================ #



