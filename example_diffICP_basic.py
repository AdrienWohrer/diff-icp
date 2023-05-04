'''
Basic test case : LDDMM registration of a point set to a fixed GMM model
'''

import os
import time
import numpy as np

import torch
from torch.nn import Module
from torch.nn.functional import softmax, log_softmax
from torch.autograd import grad

from matplotlib import pyplot as plt
import matplotlib.cm as cm
plt.ion()

import plotly.graph_objs as go

from pykeops.torch import Vi, Vj, LazyTensor, Pm
import pykeops

#pykeops.clean_pykeops()

###################################################################
# Import from other files in this directory :

from diffICP.kernel import torchspec        # GPU/CPU
from diffICP.GMM import GaussianMixtureUnif
from diffICP.LDDMM_logdet import LDDMMModel
from diffICP.Affine_logdet import AffineModel
from diffICP.genPSR import diffPSR, affinePSR
from diffICP.visu import my_scatter, plot_shoot


###################################################################
# Saving simulation results (with dill, a generalization of pickle)
savestuff = False
import dill
savefile = "saving/test_basic.pkl"
savelist = []       # store names of variables to be saved


# Plot figures ?
plotstuff = False

# Number of global loop iterations
nIter = 20

###################################################################
### Part 1 : Synthetic data generation
###################################################################


###################################################################
### "Ground truth" GMM model

# Simpler and more reproducible : use the spiral formula to draw (deterministic) centroids for GMMg
C = 20
t = torch.linspace(0, 2 * np.pi, C + 1)[:-1].to(**torchspec)
mu0 = torch.stack((0.5 + 0.4 * (t / 7) * t.cos(), 0.5 + 0.3 * t.sin()), 1).to(**torchspec)

GMMg = GaussianMixtureUnif(mu0)
GMMg.sigma = 0.025                                          # ad hoc
GMMg.to_optimize = {'mu':False, 'sigma':False, 'w':False}   # fixed parameters

if plotstuff:
    plt.figure()
    print(GMMg)
    GMMg.plot()
    plt.show()

###################################################################
### "Ground truth" generative LDDMM model

LMg = LDDMMModel(sigma = 0.2,   # sigma of the Gaussian kernel
                 D=2,           # dimension of space
                 lambd= 1e2,    # lambda of the LDDMM regularization
                 version = "classic",
                 nt = 10)       # time discretization of interval [0,1] for ODE resolution


###################################################################
### Generate samples

# Nota: .requires_grad_ are no longer necessary, they were there only to compute derivatives of H wrt q

N = 100
x0g = GMMg.get_sample(N)            # basic GMM sample
# Random deformation moments (from LDDMM model LMg)
a0g = LMg.random_p(x0g,
##        version="svd", rcond=1/LMg.lam)       #  (Value of rcond is ad hoc)
        version="ridge", alpha=10)              # (Value of alpha is ad hoc)

shoot = LMg.Shoot(x0g, a0g)             # shooting !
x0 = shoot[-1][0]                       # arrival (deformed) points

### Variables that will be saved (also add a0g and x0g for illustration!)

savelist.extend(("GMMg","LMg","N","x0g","a0g","x0"))


###################################################################
### Check

if plotstuff:
    plt.figure()
    GMMg.plot(x0g,x0)
    my_scatter(x0)
    plot_shoot(shoot,color='b')


###################################################################
### Part 2 : Registration on fixed GMM model (new algorithm)
###################################################################

### LDDMM Hamitonian system (with logdet term)

LMi = {}
a0_evol = {}

for ver in ["logdet"]:  #"classic","logdet"]:

    LMi[ver] = LDDMMModel(sigma = 0.2,  # sigma of the Gaussian kernel
                              D=2,  # dimension of space
                              lambd= 5e2,  # lambda of the LDDMM regularization
                              version = ver,
                              computversion="keops",
                              nonsupprev = False,   # For testing. Default=False (faster on first tests)
                              nt = 10)    # time discretization of interval [0,1] for ODE resolution

    ### Resulting Point Set Registration model
    # Without support decimation (Rdecim=None) or with support decimation (Rdecim>0)
    # PSR = diffPSR(x0, GMMg, LMi[ver], Rdecim=0.7, Rcoverwarning=1)
    #PSR = diffPSR(x0, GMMg, LMi[ver], Rdecim=None)

    PSR = affinePSR(x0, GMMg, AffineModel(D=2, version = 'rigid'))
    
    # for storing results
    a0_evol[ver] = []
    
    ### Optimization loop
    
    if plotstuff:
        plt.figure()

    start = time.time()
    for it in range(nIter):
        print("ITERATION NUMBER ",it)

        ### Store stuff (for saving results to file)
        # a0_evol[version][it] = current a0 tensor(N,2)
        if isinstance(PSR, diffPSR):
            a0_evol[ver].append( PSR.a0[0].clone().detach() )

        ### E step for GMM model        
        PSR.GMM_opt()

        ###LDDMM step optimization for diffeomorphisms
        PSR.Reg_opt(tol=1e-5)

        ### Plot resulting point sets (and some trajectories)
            
        if plotstuff:
            plt.clf()
            x1 = PSR.x1[0,0]
            GMMg.plot( x0, x1 )
            my_scatter(x1, alpha=.6, color="r")
            PSR.plot_trajectories(color='red')
            PSR.plot_trajectories(color='brown', support=True, linewidth=2, alpha=1)     # only useful in diffPSR class
            plt.show()
            plt.pause(.1)

        #print(PSR.x1[0,0][:5])  # debug

    print(time.time()-start)

savelist.extend(("LMi","a0_evol"))

if savestuff:
    print("Saving stuff")
    tosave = {k:globals()[k] for k in savelist}
    import dill
    with open(savefile, 'wb') as f:
        dill.dump(tosave, f)
        
# Fini ! Wait for click
if plotstuff:
    input()
