'''
Testing the diffICP algorithm on multiple point set registration (statistical atlas)
'''

import time, copy
import numpy as np
from matplotlib import pyplot as plt
plt.ion()

import torch

from pykeops.torch import Vi, Vj, LazyTensor, Pm
import pykeops

#pykeops.clean_pykeops()

# Manual random generator seeds (to always reproduce the same point sets if required)
torch.random.manual_seed(1234)

###################################################################
# Import from diffICP module

from diffICP.GMM import GaussianMixtureUnif
from diffICP.LDDMM_logdet import LDDMMModel
from diffICP.Affine_logdet import AffineModel
from diffICP.PSR import diffPSR, affinePSR
from diffICP.visu import my_scatter
from diffICP.spec import defspec, getspec

###################################################################
# Saving simulation results (with dill, a generalization of pickle)
savestuff = False
import dill
# Nota: working directory is always assumed to be the Python project home (hence, no need for ../ to return to home directory)
# When the IDE used is Pycharm, this requires to set the default run directory, as follows:
# Main Menu > Run > Edit Configurations > Edit Configuration templates > Python > Working directory [-> select project home dir]
savefile = "saving/registration_multi_1.pkl"
savelist = []       # store names of variables to be saved

# Plot figures ?
plotstuff = True

# Number of global loop iterations
nIter = 30

###################################################################
### Part 1 : Synthetic data generation
###################################################################


###################################################################
### "Ground truth" generative GMM model

# Use the spiral formula to draw (deterministic) centroids for GMMg
C = 20
t = torch.linspace(0, 2 * np.pi, C + 1)[:-1]
mu0 = torch.stack((0.5 + 0.4 * (t / 7) * t.cos(), 0.5 + 0.3 * t.sin()), 1)

GMMg = GaussianMixtureUnif(mu0)
GMMg.sigma = 0.025          # ad hoc

if False:
    print(GMMg)
    GMMg.plot()
    plt.show()
    input()

###################################################################
### "Ground truth" generative LDDMM model

LMg = LDDMMModel(sigma = 0.2,  # sigma of the Gaussian kernel
                 D=2,  # dimension of space
                 lambd= 1e2,  # lambda of the LDDMM regularization
                 version = "classic",
                 nt = 10)                   # time discretization of interval [0,1] for ODE resolution

###################################################################
### Generate (or reload) samples

reload = False
if reload:
    # Load from a previous simulation file

    loadfile = "saving/registration_multi_1.pkl"
    print("Loading data points from previous file : ",loadfile)
    with open(loadfile, 'rb') as f:
        yo = dill.load(f)
    for key in savelist:
        globals()[key] = yo[key]

else:
    # Generate new samples

    K = 10                                      # Number of point sets
    Nkbounds = 100,120                          # Bounds on the number of points in each point set
    Nk = torch.randint(*Nkbounds, (K,))         # (Random) number of points in each point set
    print("Number of point sets: ", K)
    print("Number of points in each point set:\n", Nk)

    x0 = []
    for k in range(K):
        xb = GMMg.get_sample(Nk[k])             # basic GMM sample

        # Random deformation moments (from LDDMM model LMg).
        a0b = LMg.random_p(xb,
    ##        version="svd", rcond=1/LMg.lam)     #  (Value of rcond is ad hoc)
            version="ridge", alpha=10)          # (Value of alpha is ad hoc)
        
        shoot = LMg.Shoot(xb, a0b)              # shooting !
        phi1 = shoot[-1][0]                     # arrival (deformed) points
        x0.append(phi1)                         # store point set

### Plot some data points

if plotstuff:
    GMMg.plot(*x0)
    my_scatter(*x0[:5])
    plt.pause(.2)

### Variables that will be saved (also add a0g and x0g for illustration!)

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

LMi = LDDMMModel(sigma = 0.2,                   # sigma of the Gaussian kernel
                          D=2,                  # dimension of space
                          lambd= 5e2,           # lambda of the LDDMM regularization
                          version = "logdet")   # "logdet", "classic" or "hybrid"

# Without support decimation (Rdecim=None) or with support decimation (Rdecim>0)
PSR = diffPSR(x0, GMMi, LMi, Rdecim=0.7, Rcoverwarning=1)
# PSR = diffPSR(x0, GMMi, LMi, Rdecim=None)

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
print(time.time()-start)

savelist.extend(("PSR","GMMi_evol","a0_evol"))

if savestuff:
    print("Saving stuff")
    tosave = {k:globals()[k] for k in savelist}
    with open(savefile, 'wb') as f:
        dill.dump(tosave, f)
        
# Wait for click
print('Done.')
if plotstuff:
    input()

