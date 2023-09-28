'''
Testing the diffICP algorithm on multiple point set registration (statistical atlas)
'''

import time, copy
import pickle
import numpy as np
from matplotlib import pyplot as plt
plt.ion()
import torch

#pykeops.clean_pykeops()

# Manual random generator seeds (to always reproduce the same point sets if required)
torch.random.manual_seed(1234)

###################################################################
# Import from diffICP module

from diffICP.core.GMM import GaussianMixtureUnif
from diffICP.core.LDDMM_logdet import LDDMMModel
from diffICP.core.PSR import diffPSR
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
plotstuff = False

# Number of global loop iterations
nIter = 5

###################################################################
### Part 1 : Synthetic data : 'spiral' point sets
###################################################################

reload = True
if reload:
    loadfile = "saving/sample_spiral_points_1.pkl"
    print("Loading data points from existing file : ",loadfile)
    with open(loadfile, 'rb') as f:
        yo = pickle.load(f)
    for key in ["GMMg","LMg","x0"]:
        globals()[key] = yo[key]

else:
    x0, GMMg, LMg = generate_spiral_point_sets(K=10, Nkbounds=(100,121),
                                               sigma_GMM=0.025,
                                               sigma_LDDMM=0.1, lambda_LDDMM=1e2)

K = len(x0)                         # Number of point sets
Nk = [x.shape[0] for x in x0]       # Number of points in each set

### Plot some data points
if plotstuff:
    GMMg.plot(*x0)
    my_scatter(*x0[:5])
    plt.pause(.2)

### Variables that will be saved
savelist.extend(("GMMg","LMg","K","Nk","x0"))


###################################################################
### Part 2 : Registration/inference (new diffICP algorithm)
###################################################################


### GMM model

C = 20
GMMi = GaussianMixtureUnif(torch.zeros(C,2))    # initial value for mu = whatever (will be changed by PSR algo)
GMMi.to_optimize = {
    "mu" : True,
    "sigma" : True,
    "w" : True
}

### Point Set Registration model : diffeomorphic version

LMi = LDDMMModel(sigma = 0.2,                           # sigma of the Gaussian kernel
                          D=2,                          # dimension of space
                          lambd= 5e2,                   # lambda of the LDDMM regularization
                          version = "logdet",           # "logdet", "classic" or "hybrid"
                          computversion="keops",        # "torch" or "keops"
                          scheme="Euler")               # "Euler" or "Ralston"

# Without support decimation (Rdecim=None) or with support decimation (Rdecim>0)
# PSR = diffPSR(x0, GMMi, LMi, Rdecim=0.7, Rcoverwarning=1)
PSR = diffPSR(x0, GMMi, LMi)
# Add a decimation scheme ?
PSR.set_support_scheme("decim", rho=0.7)
# PSR.set_support_scheme("grid", rho=1.0)

### Point Set Registration model : affine version

# PSR = affinePSR(x0, GMMi, AffineModel(D=2, version = 'rigid'))
# PSR = affinePSR(x0, GMMi, AffineModel(D=2, version = 'affine', withlogdet=False))

# for storing results
a0_evol = []
GMMi_evol = []

### Optimization loop

if plotstuff:
    plt.figure()

start = time.time()
for it in range(nIter):
    print("ITERATION NUMBER ",it)

    ### Store stuff (for saving results to file)
    # GMMi_evol[it] = current GMMi object
    # a0_evol[it][k] = current a0 tensor(Nk[k],2)

    GMMi_evol.append( copy.deepcopy( PSR.GMMi[0] ) )
    if isinstance(PSR, diffPSR):
        a0_evol.append( [ a0k.clone().detach().cpu() for a0k in PSR.a0 ] ) # only in diffPSR case

    ### EM step for GMM model
    PSR.GMM_opt(repeat=10)

    ### M step optimization for diffeomorphisms (individually for each k)
    PSR.Reg_opt(tol=1e-5)

    ### Plot resulting point sets (and some trajectories)

    if plotstuff:
        plt.clf()
        x1 = PSR.x1[:,0]
        PSR.GMMi[0].plot(*x0, *x1)
        my_scatter(*x1[0:min(5,PSR.K)], alpha=.6)
        for k in range(min(5,PSR.K)):
            # ...
            # PSR.plot_trajectories(k)
            PSR.plot_trajectories(k, support=True, linewidth=2, alpha=1)     # only useful in diffPSR class
        plt.show()
        plt.pause(.1)

# Done !
print(f"Elapsed time : {time.time()-start} seconds")

savelist.extend(("PSR","GMMi_evol","a0_evol"))

if savestuff:
    print("Saving stuff")
    tosave = {k:globals()[k] for k in savelist}
    with open(savefile, 'wb') as f:
        pickle.dump(tosave, f)
        
# Wait for click
print('Done.')
if plotstuff:
    input()

