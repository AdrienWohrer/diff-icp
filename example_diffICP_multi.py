'''
Testing the diffGMM algorithm on multiple point set registration (statistical atlas)
'''

import os, time, math

import copy

import numpy as np
rng = np.random.default_rng(seed=1234)

from matplotlib import pyplot as plt
plt.ion()

import torch

# torch type and device
use_cuda = torch.cuda.is_available()
torchdeviceId = torch.device("cuda:0") if use_cuda else "cpu"
torchdtype = torch.float32
tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
torch.manual_seed(1234)

###################################################################
# Import from other files in this directory :

from diffICP.GMM import GaussianMixtureUnif
from diffICP.LDDMM_logdet import LDDMMModel
from diffICP.Affine_logdet import AffineModel
from diffICP.visu import my_scatter, plot_shoot
from diffICP.PSR import diffPSR, affinePSR


###################################################################
# Saving simulation results (with dill, a generalization of pickle)
savestuff = False
import dill
#savefile = "saving/last_registration_multi_lam5e2.pkl"
savefile = "saving/test_multi.pkl"
savelist = []       # store names of variables to be saved

# Plot figures ?
plotstuff = True

# Number of global loop iterations
nIter = 50


###################################################################
### Part 1 : Synthetic data generation
###################################################################


###################################################################
### "Ground truth" generative GMM model

# Use the spiral formula to draw (deterministic) centroids for GMMg
C = 20
t = torch.linspace(0, 2 * np.pi, C + 1)[:-1]
mu0 = torch.stack((0.5 + 0.4 * (t / 7) * t.cos(), 0.5 + 0.3 * t.sin()), 1)
mu0 = mu0.type(torchdtype)

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

### (Names of) variables that will be saved

savelist.extend(("GMMg","LMg","K","Nk","x0"))

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
    Nk = rng.integers(*Nkbounds,size=K)         # (Random) number of points in each point set
    print("Number of point sets: ", K)
    print("Number of points in each point set:\n", Nk)

    x0 = []
    for k in range(K):
        xb = GMMg.get_sample(Nk[k])             # basic GMM sample
#        xb = GMMg.get_sample(Nk[k])#.requires_grad_(True)    # for 'oldkeep' versions. Forget it !

        # Random deformation moments (from LDDMM model LMg).    # Here also, requires_grad_ is not needed anymore
        a0b = LMg.random_p(xb,
    ##        version="svd", rcond=1/LMg.lam)#.requires_grad_(True)     #  (Value of rcond is ad hoc)
            version="ridge", alpha=10)#.requires_grad_(True)          # (Value of alpha is ad hoc)
        
        shoot = LMg.Shoot(xb, a0b)              # shooting !
        phi1 = shoot[-1][0]                     # arrival (deformed) points
        x0.append(phi1)                         # store point set

### Plot some data points

if plotstuff:
    GMMg.plot(*x0)
    my_scatter(*x0[:5])
    plt.pause(.2)



###################################################################
### Part 2 : Registration/inference (new diffGMM algorithm)
###################################################################


LMi = {}        # two versions
GMMi_evol = {}  # to store results
a0_evol = {}    # idem

for ver in ["logdet"]:

    ### LDDMM model parameters (two versions)

    LMi[ver] = LDDMMModel(sigma = 0.2,  # sigma of the Gaussian kernel
                          D=2,  # dimension of space
                          lambd= 5e2,  # lambda of the LDDMM regularization
                          version = ver,
                          computversion = "keops",    # For testing. Default="keops"
                          #nonsupprev = True,   # For testing. Default=False (faster on first tests)
                          nt = 10)                   # time discretization of interval [0,1] for ODE resolution

    # Définition à la main d'un "hybride", sans gradcomponent mais avec divcost
    # LMi[ver] = LDDMMModel(sigma= 0.2,  # sigma of the Gaussian kernel
    #                       D=2,  # dimension of space
    #                       lambd= 5e2,  # lambda of the LDDMM regularization
    #                       gradcomponent = False,
    #                       withlogdet = True,
    #                       usetrajcost = True,
    #                       nt = 10)                   # time discretization of interval [0,1] for ODE resolution

    ### GMM model

    C = 20
    GMMi = GaussianMixtureUnif(torch.zeros(C,2))    # initial value for mu = whatever (will be changed by PSR algo)
    GMMi.to_optimize = {
        "mu" : True,
        "sigma" : True,
        "w" : True
    }

    ### Resulting Point Set Registration model
    # Without support decimation (Rdecim=None) or with support decimation (Rdecim>0)
    #PSR = diffPSR(x0, GMMi, LMi[ver], Rdecim=0.7, Rcoverwarning=1)
    # PSR = diffPSR(x0, GMMi, LMi[ver], Rdecim=None)

    PSR = affinePSR(x0, GMMi, AffineModel(D=2, version = 'rigid'))

    ### Optimization loop

    if plotstuff:
        plt.figure()
    
    # to store results
    GMMi_evol[ver] = []
    a0_evol[ver] = []

    start = time.time()
    for it in range(nIter):
        print("ITERATION NUMBER ",it)
        
        ### Store stuff (for saving results to file)
        # GMMi_evol[version][it] = current GMMi object
        # a0_evol[version][it][k] = current a0 tensor(Nk[k],2)
        
        GMMi_evol[ver].append( copy.deepcopy( PSR.GMMi[0] ) )
        if isinstance(PSR, diffPSR):
            a0_evol[ver].append( [ a0k.clone().detach() for a0k in PSR.a0 ] ) # only in diffPSR case
        
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
                ...
                PSR.plot_trajectories(k)
                PSR.plot_trajectories(k, support=True, linewidth=2, alpha=1)     # only useful in diffPSR class
            plt.show()
            plt.pause(.1)

    # Done !
    print(time.time()-start)

savelist.extend(("LMi","mu0","GMMi_evol","a0_evol"))

if savestuff:
    print("Saving stuff")
    tosave = {k:globals()[k] for k in savelist}
    import dill
    with open(savefile, 'wb') as f:
        dill.dump(tosave, f)
        
# Wait for click
if plotstuff:
    input()

