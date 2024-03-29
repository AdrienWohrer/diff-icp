'''
DiffICP algorithm on multiple point set registration (statistical atlas)

This is an example script manipulating directly the underlying core models (GMM, LDDMM, PSR).
For a directly usable function, see api/ICP_atlas.py
'''

import time, copy, os
import pickle
import numpy as np
from matplotlib import pyplot as plt
plt.ion()
import torch

#pykeops.clean_pykeops()

# Manual random generator seeds (to always reproduce the same point sets if required)
# torch.random.manual_seed(1234)

###################################################################
# Import from diffICP module

from diffICP.core.GMM import GaussianMixtureUnif
from diffICP.core.LDDMM import LDDMMModel
from diffICP.core.PSR import DiffPSR
from diffICP.visualization.visu import my_scatter
from diffICP.examples.generate_spiral_point_sets import generate_spiral_point_sets

###################################################################
# Saving simulation results

savestuff = False
# Nota: working directory is always assumed to be the Python project home (hence, no need for ../ to return to home directory)
# When the IDE used is Pycharm, this requires to set the default run directory, as follows:
# Main Menu > Run > Edit Configurations > Edit Configuration templates > Python > Working directory [-> select project home dir]
savefile = "saving/registration_multi_spiral_1.pkl"
savelist = []       # store names of variables to be saved

# Plot figures ?
plotstuff = True

# Number of global loop iterations
nIter = 25

###################################################################
### Part 1 : Synthetic data : 'spiral' point sets
###################################################################

# Recover or create some sample point sets
datafile = "saving/sample_spiral_points_1.pkl"
if os.path.exists(datafile):
    with open(datafile, 'rb') as f:
        yo = pickle.load(f)
        x0 = yo["x0"]       # Sample point sets
        GMMg = yo["GMMg"]   # Generating GMM
        LMg = yo["LMg"]     # Generating diffeomorphism
else:
    N = 100
    K = 10
    x0, GMMg, LMg = generate_spiral_point_sets(K=K, Nkbounds=(N, N + 40),
                                          sigma_GMM=0.025,
                                          sigma_LDDMM=0.1, lambda_LDDMM=1e2)

K = len(x0)                         # Number of point sets
Nk = [x.shape[0] for x in x0]       # Number of points in each set

### Plot some data points
GMMg.plot(*x0)
my_scatter(*x0[:5])
plt.pause(.2)

###################################################################
### Part 2 : Registration/inference (new diffICP algorithm)
###################################################################

### GMM model

C = 20
GMMi = GaussianMixtureUnif(torch.zeros(C,2))    # initial value for mu = whatever (will be changed by PSR.reinitialize_GMM)
GMMi.to_optimize = {
    "mu" : True,
    "sigma" : True,
    "w" : True
}

### Point Set Registration model : diffeomorphic version

LMi = LDDMMModel(sigma = 0.2,                           # sigma of the Gaussian kernel
                          D=2,                          # dimension of space
                          lambd= 5e2,                   # lambda of the LDDMM regularization
                          version = "hybrid",           # "logdet", "classic" or "hybrid"
                          computversion="keops",        # "torch" or "keops"
                          scheme="Euler")               # "Euler" or "Ralston"

PSR = DiffPSR(x0, GMMi, LMi)
PSR.reinitialize_GMM()

# Change support scheme ?
PSR.set_support_scheme("grid", rho=np.sqrt(2))
# PSR.set_support_scheme("decim", rho=0.7)

### Point Set Registration model : affine version

# PSR = AffinePSR(x0, GMMi, AffineModel(D=2, version = 'rigid'))
# PSR = AffinePSR(x0, GMMi, AffineModel(D=2, version = 'general_affine'))

# for storing results
a0_evol = []
GMMi_evol = []

### Optimization loop

plt.figure()

for it in range(nIter):
    print("ITERATION NUMBER ",it)

    ### Store stuff (for saving results to file)
    # GMMi_evol[it] = current GMMi object
    # a0_evol[it][k] = current a0 tensor(Nk[k],2)

    GMMi_evol.append( copy.deepcopy( PSR.GMMi[0] ) )
    if isinstance(PSR, DiffPSR):
        a0_evol.append( [ a0k.clone().detach().cpu() for a0k in PSR.a0 ] ) # only in diffPSR case

    ### EM step for GMM model
    PSR.GMM_opt(max_iterations=10)

    ### M step optimization for diffeomorphisms (individually for each k)
    PSR.Reg_opt(tol=1e-5, nmax=1)

    ### Plot resulting point sets (and some trajectories)

    plt.clf()
    x1 = PSR.x1[:,0]
    PSR.GMMi[0].plot(*x0, *x1)
    my_scatter(*x1[0:min(5,PSR.K)], alpha=.6)
    for k in range(min(5,PSR.K)):
        # ...
        PSR.plot_trajectories(k)
        # PSR.plot_trajectories(k, support=True, linewidth=2, alpha=1)     # only useful in diffPSR class
    plt.show()
    plt.pause(.1)

# Wait for click
print('Done.')
if plotstuff:
    input()

