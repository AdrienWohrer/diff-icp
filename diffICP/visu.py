'''
Visualization functions used in the LDDMM+GMM combination
'''

import os, time, math

import numpy as np
import torch
from matplotlib import pyplot as plt
#import plotly.express as px
from mpl_toolkits.mplot3d import axes3d

###################################################################
### Scatter plot
###################################################################

# USAGE: my_scatter(x1,x2,x3,..., **kwargs)
# - each xi of size (M,2) is a torch tensor reprensenting the i-th point set to plot
# - each point set is given a color following the current color order (except if a (unique) color is given in **kwargs).

def my_scatter(*xlist, **kwargs):

    # dimensionality of the plot (should be homogeneous, else error)
    D = xlist[0].shape[1]
    if D != 2 and D != 3 :
        raise ValueError("Can only plot datapoints in 2d and 3d")

    # Some default arguments
    autocolor = "color" not in kwargs.keys()
    if "linewidth" not in kwargs.keys():
        kwargs["linewidth"] = 3
    if "s" not in kwargs.keys():
        kwargs["s"] = 1            # ad hoc

    for i,x in enumerate(xlist):
        xy = x.data.cpu().numpy()
        if autocolor:
                kwargs["color"] = "C" + str(i)
        if len(xy)>0 :
            if D==2:
                plt.scatter(xy[:,0], xy[:,1], **kwargs)
            else:
                ax = plt.gca(projection='3d')
                ax.scatter(xy[:,0], xy[:,1], xy[:,2], **kwargs)

###################################################################
### Visualization of LDDMM geodesics (obsolete, better use PSR.plot_trajectories)
###################################################################

# - shoot = list, as returned by function LDDMMModel.Shoot().
# - toplot : support vs. non-support points. 0 = support points (q), 3 = non-support points (x)
# - styles, widths, alphas : corresponding plot parameters, for both types of points
# - kwargs = other plot options

def plot_shoot(shoot, is_decim=False, **kwargs):

    if not is_decim:
        toplot, styles, widths, alphas = [0], ['-'], [1], [.5]
    else:
        toplot, styles, widths, alphas = [0,3], ['-','-'], [2,1], [.7,.4]

    for i,id in enumerate(toplot):
        x = [ tup[id] for tup in shoot ]
        for n in range(x[0].shape[0]):
            xnt = np.array([ xt[n,:].tolist() for xt in x ])
            plt.plot(xnt[:,0],xnt[:,1], linestyle=styles[i], linewidth=widths[i], alpha=alphas[i], **kwargs)
