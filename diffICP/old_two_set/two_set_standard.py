'''
Standard diffeomorphic point-set-matching algorithm from Glaunès et al 04

This is a reimplementation using personal code, mostly PSR_standard.py
'''

import os
import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

plt.ion()

import torch

# Manual random generator seeds (to always reproduce the same point sets if required)
# torch.random.manual_seed(1234)


###################################################################
# Import from diffICP module

from diffICP.core.LDDMM import LDDMMModel
from diffICP.visualization.visu import my_scatter, get_bounds, on_top
from diffICP.visualization.grid import Gridlines

from diffICP.tools.kernel import GaussKernel
from diffICP.tools.spec import defspec

from diffICP.core.PSR_standard import DiffPSR_std


###################################################################
# Saving simulation results (with dill, a generalization of pickle)
savestuff = False
import dill
# Nota: working directory is always assumed to be the Python project home (hence, no need for ../ to return to home directory)
# When the IDE used is Pycharm, this requires to set the default run directory, as follows:
# Main Menu > Run > Edit Configurations > Edit Configuration templates > Python > Working directory [-> select project home dir]
savefile = "figs/glaunes_two_sets/savefile.pkl"
savelist = []       # store names of variables to be saved

# Plot figures ?
plotstuff = True
# Save figures ?
savefigs = False
savefigs_name = 'figs/glaunes_two_sets/figure'
savefigs_format = 'png'

# Number of global loop iterations
nIter = 40
# Number of optimization iterations per global loop
nmax_one_iter = 10

### Load data = two point sets
##############################

import scipy.io
chui_dataset = 3    # (1 to 5)
yo = scipy.io.loadmat(f"diffICP/examples/chui-data/demodata_ex{chui_dataset}.mat")
x_name = f'x{[1,2,3,1,1][chui_dataset-1]}'   # name of the variables in the chui file (no general rule :))
y_name = f'y{[1,2,3,"2a","2a"][chui_dataset-1]}'   # name of the variables in the chui file (no general rule :))
xA = torch.tensor(yo[x_name], **defspec).contiguous()   # xA will correspond to fixed GMM centroids
xB = torch.tensor(yo[y_name], **defspec).contiguous()   # xB will be the registered point set
if False:    # try the reversed version
    xB, xA = xA, xB

savelist.extend(("chui_dataset","xA","xB"))

D = xA.shape[1]

#### DATA RKHS : used to embed the point sets as elements of a RKHS

sigma_data = 0.1
DataKernel = GaussKernel(sigma_data, D, spec=defspec)

### Registration model

LMi = LDDMMModel(D=D, sigma = 0.1,         # sigma of the Gaussian kernel (note: this is a different sigma than sigma_data)
                     lambd= 2,             # lambda of the LDDMM regularization. Always 2 in the case of Glaunès et al.
                     version = "classic")  # always "classic" in the case of Glaunès et al.

### Point-Set Registration object à la Glaunès

noise_std = 0.2
PSR = DiffPSR_std(xA, xB, noise_std, LMi, DataKernel)
savelist.extend(("PSR",))

### Plotting routine
####################

bounds = get_bounds(xA, xB, relmargin=0.1)
use_diffPSR = True

def plot_step(only_data=False):
    plt.clf()
    plot_complete = not only_data
    ### Grid lines
    if plot_complete and use_diffPSR:
        gridlines = Gridlines(np.linspace(bounds[0], bounds[1], 10), np.linspace(bounds[2], bounds[3], 10))
        # gridlines.plot(color='gray', linewidth=1, linestyle='dotted')
        reglines = gridlines.register(PSR.Registration())
        reglines.plot(color=(0.8, 0.5, 0.5), linewidth=1)
    ### Two point sets
    my_scatter(PSR.get_data_points(), alpha=1, color="b")
    my_scatter(PSR.get_warped_template(), alpha=1, color="red")


    plt.xticks(np.arange(-10, 10 + 0.1, 0.5))
    plt.yticks(np.arange(-10, 10 + 0.1, 0.5))
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))  # Ensure correctly formated ticks
    plt.gca().set_aspect('equal')
    plt.xlim(bounds[:2])
    plt.ylim(bounds[2:])
    on_top(plt.gcf())  # from diffICP.visu
    # plt.gca().autoscale(tight=True)
    plt.pause(.1)

# Figure 1 : basic point sets
if plotstuff:
    fig = plt.figure()
    plot_step(only_data=True)
    if savefigs:
        savefigs_path = os.path.dirname(os.path.realpath(savefigs_name))
        os.makedirs(savefigs_path, exist_ok=True)
        plt.savefig(f"{savefigs_name}_data.{savefigs_format}", format=savefigs_format, bbox_inches='tight')

### Optimization loop
#####################

# for storing parameter evolution in time
param_evol = []

if plotstuff:
    plt.figure()

start = time.time()
for it in range(nIter):
    print("ITERATION NUMBER ", it)

    ### Store stuff (for saving results to file)

    if use_diffPSR:
        par = {'a0': PSR.a0[0].cpu()}  # param_evol[version][it]['a0'] = current a0 tensor(N,2) , etc.
        # par = {'a0': PSR.a0.cpu()}  # param_evol[version][it]['a0'] = current a0 tensor(N,2) , etc.
    else:
        par = {}                    # TODO also implement affine ?
    param_evol.append(par)

    ### One optimization step (to view evolution of optimizer)
    PSR.Reg_opt(nmax=nmax_one_iter, tol=1e-5)

    if plotstuff:
        plot_step()
        if savefigs:
            plt.savefig(f"{savefigs_name}_{it}.{savefigs_format}", format=savefigs_format, bbox_inches='tight')

# Done.

print(time.time() - start)
savelist.extend(("PSR", "param_evol","nmax_one_iter"))

if savestuff:
    print("Saving stuff")
    tosave = {k: globals()[k] for k in savelist}
    with open(savefile, 'wb') as f:
        dill.dump(tosave, f)

# Fini ! Wait for click
print("Done.")
if plotstuff:
    input()