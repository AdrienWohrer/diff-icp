'''
Various visualization functions used in the LDDMM+GMM combination (only one for the moment!)
'''

import os, time, math, warnings

import numpy as np

import matplotlib
#matplotlib.use('TkAgg') # should normally be the case by default
from matplotlib import pyplot as plt

import torch
#import plotly.express as px
from mpl_toolkits.mplot3d import axes3d

###################################################################

def on_top(fig):
    '''Set a plot window on top of the others (requires TkAgg)'''
    try:
        # Put figure window on top of all other windows
        fig.canvas.manager.window.attributes('-topmost', 1)
        # After placing figure window on top, allow other windows to be on top of it later
        fig.canvas.manager.window.attributes('-topmost', 0)
    except:
        warnings.warn("Function on_top() requires TkAgg backend")


###################################################################

# USAGE: xmin,xmax,ymin,ymax = get_bounds(x1,x2,x3,..., relmargin=0.2, step=0.1)
# - each xi of size (M,2) is a torch tensor reprensenting a point set
# step=TODO if necessary

def get_bounds(*xlist, **kwargs):
    '''Automatic bounds extraction with a relative margin `relmargin` and an imposed multiple of `step`'''

    mins = torch.cat(tuple(xy.min(0).values.reshape(1, 2) for xy in xlist if len(xy) > 0), 0).min(0).values.numpy()
    maxs = torch.cat(tuple(xy.max(0).values.reshape(1, 2) for xy in xlist if len(xy) > 0), 0).max(0).values.numpy()

    if "relmargin" in kwargs.keys():
        relmargin = kwargs["relmargin"]
    else:
        relmargin = 0.2
    gmin = ((1 + relmargin) * mins - relmargin * maxs)
    gmax = ((1 + relmargin) * maxs - relmargin * mins)

    return gmin[0],gmax[0],gmin[1],gmax[1]


###################################################################

def my_scatter(*xlist, **kwargs):
    '''USAGE: my_scatter(x1,x2,x3,..., **kwargs)
    - each xi of size (M,2) is a torch tensor reprensenting the i-th point set to plot
    - each point set is given a color following the current color order (except if a (unique) color is given in **kwargs).
    '''

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

