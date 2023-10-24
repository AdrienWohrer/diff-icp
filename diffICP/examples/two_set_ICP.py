'''
Compare different algorithms for classic ("two set") ICP or diffICP
'''

import os, time
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter


plt.ion()

import torch

#pykeops.clean_pykeops()

# Manual random generator seeds (to always reproduce the same point sets if required)
#torch.random.manual_seed(1234)


###################################################################
# Import from diffICP module

from diffICP.core.GMM import GaussianMixtureUnif
from diffICP.core.LDDMM import LDDMMModel
from diffICP.core.affine import AffineModel
from diffICP.core.PSR import DiffPSR, AffinePSR
from diffICP.visualization.visu import my_scatter, get_bounds, on_top
from diffICP.tools.spec import defspec
from diffICP.visualization.grid import Gridlines

###################################################################
# Saving simulation results
savestuff = False
# Nota: working directory is always assumed to be the Python project home (hence, no need for ../ to return to home directory)
# When the IDE used is Pycharm, this requires to set the default run directory, as follows:
# Main Menu > Run > Edit Configurations > Edit Configuration templates > Python > Working directory [-> select project home dir]
savefile = "figs/ICP_diffeomorphe/savefile.pkl"
savelist = []       # store names of variables to be saved

# Plot figures ?
plotstuff = True
# Save figures ?
savefigs = False
savefigs_name = 'figs/ICP_diffeomorphe/figure'
savefigs_format = 'png'

# Number of global loop iterations
nIter = 40

# Basic (non probabilistic) ICP ?
basic_ICP = False
# GMM parameters for probabilistic ICP :
if not basic_ICP:
    sigma_opt = True            # optimize sigma (True or False) ?
    sigma_start = 0.5           # starting sigma for GMMi (if sigma_opt=True) or FIXED value for GMMi (if sigma_opt=False)

# Use new diffPSR algorithm ? Else, use affinePSR (= "classic" (probabilistic) ICP, more or less)
use_diffPSR = True

if not use_diffPSR:
    ### Affine registration model (== classic probablistic ICP)
    AffMi = AffineModel(D=2, version = 'euclidian', withlogdet=False)
else:
    ### Diffeomorphic registration model (new diffICP algorithm)
    LMi = LDDMMModel(D=2, sigma = 0.2,          # sigma of the Gaussian kernel
                     lambd= 2e2,                # lambda of the LDDMM regularization
                     computversion="keops",
                     version = "logdet")        # "logdet", "classic" or "hybrid"
    # Without support decimation (rho=None) or with support decimation (rho>0)
    rho = 0.7

savelist.extend(("basic_ICP","use_diffPSR","nIter"))

#######################################

### Load data = two point sets
##############################

import scipy.io
chui_dataset = 1    # (1 to 5)
yo = scipy.io.loadmat(f"diffICP/examples/chui-data/demodata_ex{chui_dataset}.mat")
x_name = f'x{[1,2,3,1,1][chui_dataset-1]}'   # name of the variables in the chui file (no general rule :))
y_name = f'y{[1,2,3,"2a","2a"][chui_dataset-1]}'   # name of the variables in the chui file (no general rule :))
xA = torch.tensor(yo[x_name], **defspec).contiguous()   # xA will correspond to fixed GMM centroids
xB = torch.tensor(yo[y_name], **defspec).contiguous()   # xB will be the registered point set
if False:    # try the reversed version
    xB, xA = xA, xB

savelist.extend(("chui_dataset","xA","xB"))

### GMM model
GMMi = GaussianMixtureUnif(xA)
if basic_ICP:
    GMMi.sigma = 0.001      # small fixed value, our numerical replacement for "0"
    GMMi.to_optimize = {'mu':False, 'sigma':False, 'w':False}
else:
    GMMi.sigma = sigma_start
    GMMi.to_optimize = {'mu':False, 'sigma':sigma_opt, 'w':False}

### And thus : full Point Set Registration algorithm
if not use_diffPSR:
    PSR = AffinePSR(xB, GMMi, AffMi)
else:
    PSR = DiffPSR(xB, GMMi, LMi)
    if rho is not None:
        PSR.set_support_scheme("decim", rho)

### Plotting routine
####################

bounds = get_bounds(xA, xB, relmargin=0.1)

def plot_step(only_data=False):
    plt.clf()
    x1 = PSR.get_warped_data_points()
    plot_complete = not only_data
    ### GMM Heatmap and contours
    if plot_complete and not basic_ICP:
        PSR.GMMi[0].plot(bounds=bounds, color="#A1C8C8", cmap="RdBu", heatmap_amplification=0.7)  # https://matplotlib.org/stable/gallery/color/colormap_reference.html
    ### Association between each point and its quadratic target
    if plot_complete and True:
        assoc = torch.stack((x1, PSR.y[0,0]))
        for n in range(x1.shape[0]):
            plt.plot(assoc[:,n,0], assoc[:,n,1], color="purple", linewidth=0.5)
    ### Grid lines
    if plot_complete and use_diffPSR:
        gridlines = Gridlines(np.linspace(bounds[0],bounds[1],10), np.linspace(bounds[2],bounds[3],10))
        # gridlines.plot(color='gray', linewidth=1, linestyle='dotted')
        reglines = gridlines.register(PSR.Registration())
        reglines.plot(color=(0.8,0.5,0.5), linewidth=1)
    ### Two point sets
    my_scatter(PSR.GMMi[0].mu, alpha=1, color="b")
    my_scatter(x1, alpha=1, color="red")

    plt.xticks(np.arange(-10, 10 + 0.1, 0.5))
    plt.yticks(np.arange(-10, 10 + 0.1, 0.5))
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))  # Ensure correctly formated ticks
    plt.gca().set_aspect('equal')
    plt.xlim(bounds[:2])
    plt.ylim(bounds[2:])
    on_top(plt.gcf())        # from diffICP.visu
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
    print("ITERATION NUMBER ",it)

    ### Store stuff (for saving results to file)

    if use_diffPSR:
        par = {'a0':PSR.a0[0].cpu()}        # param_evol[version][it]['a0'] = current a0 tensor(N,2) , etc.
    else:
        par = {'M':PSR.M[0].cpu(), 't':PSR.t[0].cpu()}
    par['sigma'] = PSR.GMMi[0].sigma
    param_evol.append(par)

    ### E step for GMM model
    PSR.GMM_opt()

    if not basic_ICP and sigma_opt:
        print(f"Sigma: {PSR.GMMi[0].sigma}")

    if plotstuff:
        plot_step()
        if savefigs:
            plt.savefig(f"{savefigs_name}_{it}_a.{savefigs_format}", format=savefigs_format, bbox_inches='tight')

    ### LDDMM step optimization for diffeomorphisms (or procrustes optimization for affine transform)
    PSR.Reg_opt(tol=1e-5)

    if plotstuff:
        plot_step()
        if savefigs:
            plt.savefig(f"{savefigs_name}_{it}_b.{savefigs_format}", format=savefigs_format, bbox_inches='tight')

# Done.
print(f"Elapsed time : {time.time()-start} seconds")

savelist.extend(("PSR","param_evol"))

if savestuff:
    print("Saving stuff")
    tosave = {k:globals()[k] for k in savelist}
    with open(savefile, 'wb') as f:
        pickle.dump(tosave, f)
        
# Fini ! Wait for click
print("Done.")
if plotstuff:
    input()
