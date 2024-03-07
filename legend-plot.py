"""
K.E. Spaans 2024
This script contains code for setting the legends for the plots.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

fig, ax = plt.subplots(figsize=(6, 2))
fig.subplots_adjust(bottom=0.7)

cmap = mpl.cm.gist_gray
norm = mpl.colors.Normalize(vmin=0, vmax=0.008)

cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
cb1.ax.locator_params(nbins=5)

cb1.set_label('mm$^2$/s', size=15) #mm$^2$/s
cb1.ax.tick_params(labelsize=15)

plt.savefig("/data/projects/followup-NOIV/students/kespaans/data/thesis_figures/legend_Dstar_axial.png")