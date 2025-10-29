import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

#=================================================================================================#
from matplotlib import font_manager, rcParams

#------- if you want to you use a specific font
#        download the font file chosen, decomment and modify the path with the one of your font file
#font_file = "/home/tentori/.local/avenir_ff/AvenirLTStd-Roman.otf"
#font_file_b = "/home/tentori/.local/avenir_ff/AvenirLTStd-Black.otf"
#font_file_c = "/home/tentori/.local/avenir_ff/AvenirLTStd-Book.otf"
#font_manager.fontManager.addfont(font_file)
#font_manager.fontManager.addfont(font_file_b)
#font_manager.fontManager.addfont(font_file_c)
#--predef font: Avenir
#rcParams['font.family'] = "Avenir LT Std"
#--------

# these could be commented
DIM = 25

plt.rcParams.update({
    'font.size': DIM,
    'axes.labelsize': DIM,
    'axes.titlesize': DIM,
    'xtick.labelsize': DIM,
    'ytick.labelsize': DIM
})
#=================================================================================================#

#=================================================================================================#
# formatter for plots 
# to control axes and colobar formats (both ticks dimension and scientific notation pwr)
# to despine the box  (comment sns.despine(ax=ax, trim=False) if you do not want despine)

def set_format(ax, axis_ticks = 'both', pwr_x_min=-2, pwr_x_max=2, pwr_y_min=-2, pwr_y_max=2,  cbar = None, pwr_cbar_min=-1, pwr_cbar_max=1,  DIM = 30):
    
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

        cbar.formatter = formatter_cbar
        cbar.update_ticks()
        
        # Move the offset text to the top of the colorbar
        dx, dy = 0.8, 0.3  # Adjust dy for vertical and dx for horizontal shifts
        cbar_offset = ScaledTranslation(dx, dy, cbar.ax.figure.dpi_scale_trans)
        cbar.ax.yaxis.offsetText.set_transform(cbar.ax.yaxis.offsetText.get_transform() + cbar_offset)
        
