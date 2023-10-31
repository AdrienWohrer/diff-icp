'''
Testing the full DiffICP algorithm (multiple frames + multiple structures).
TODO: this script should be reworked with a better example + better visualization.

This is an example script manipulating directly the underlying core models (GMM, LDDMM, PSR).
For a directly usable function, see api/ICP_atlas.py
'''

import copy
import pickle
import numpy as np

from matplotlib import pyplot as plt
plt.ion()

import torch

# Manual random generator seeds (to always reproduce the same point sets if required)
torch.random.manual_seed(1234)

###################################################################
# Import from diffICP module

from diffICP.core.GMM import GaussianMixtureUnif
from diffICP.core.LDDMM import LDDMMModel
from diffICP.core.PSR import DiffPSR
from diffICP.visualization.visu import my_scatter


###################################################################
### Part 1 : Synthetic data generation
###################################################################

###################################################################
### "Ground truth" generative GMM models

GMMg = [None]*3

S = 3           # number of structures
C = 20          # number of clusters per structure
t = torch.linspace(0, 2 * np.pi, C + 1)[:-1]

# first structure
mu = torch.stack((0.5 + 0.4 * (t / 7) * t.cos(), 0.5 + 0.3 * t.sin()), 1)
GMMg[0] = GaussianMixtureUnif(mu)
GMMg[0].sigma = 0.025          # ad hoc

# second structure
mu = torch.stack((1 + 0.4 * t.cos(), 0.5 + 0.4 * t.sin()), 1)
GMMg[1] = GaussianMixtureUnif(mu)
GMMg[1].sigma = 0.04          # ad hoc

# third structure
mu = torch.stack( (0.8 + 0.1 * (t - np.pi), -0.06* (t - np.pi)), 1)
GMMg[2] = GaussianMixtureUnif(mu)
GMMg[2].sigma = 0.2          # ad hoc

plt.figure(S)
for s in range(S):
    #print(GMMg[s])
    GMMg[s].plot(bounds=[0,2, -0.5,1.5], heatmap=False, color=f"C{s}")  # ugly but sufficient for the moment
    plt.show()

###################################################################
### "Ground truth" generative LDDMM model

LMg = LDDMMModel(sigma = 0.2,  # sigma of the Gaussian kernel
                 D=2,  # dimension of space
                 lambd= 1e2,  # lambda of the LDDMM regularization
                 version = "classic",
                 nt = 10)                   # time discretization of interval [0,1] for ODE resolution

###################################################################
### Generate samples

K = 10                                      # Number of frames
Nkbounds = 40,50                            # Bounds on the number of points in each point set
Nk = torch.randint(*Nkbounds, (K,S))        # (Random) number of points in each point set

x0 = [None] * K
for k in range(K):
    x0[k] = [None] * S
    for s in range(S):
        xb = GMMg[s].get_sample(Nk[k,s])   # basic GMM sample

        # Random deformation moments (from LDDMM model LMg) ( requires_grad_ is not needed anymore )
        a0b = LMg.random_p(xb,
    ##        version="svd", rcond=1/LMg.lam)   #  (Value of rcond is ad hoc)
            version="ridge", alpha=10)          # (Value of alpha is ad hoc)

        shoot = LMg.Shoot(xb, a0b)              # shooting !
        x0[k][s] = shoot[-1][0]                 # arrival (deformed) points

        # Test : what is the effect of having some empty structures ?
        # if torch.random() < 0.1 :
        #     x0[k][s] = torch.empty(0,2)
        #     Nk[k,s] = 0

print("Number of frames: ", K)
print("Number of structures: ", S)
print("Number of points in each point set:\n", Nk)

### Plot some data points (one structure = one figure) TODO maybe not ideal

for s in range(S):
    plt.figure(s)
    plt.clf()
    x0s = [ xk[s] for xk in x0 ]
    GMMg[s].plot(*x0s[:4])
    my_scatter(*x0s[:4])
plt.pause(.2)



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

PSR = DiffPSR(x0, GMMi, LMi)
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

nIter = 15

for it in range(nIter):
    print("ITERATION NUMBER ",it)

    ### Store stuff (for saving results to file)
    # GMMi_evol[version][it][s] = current GMMi objects
    # a0_evol[version][it][k] = current a0 tensor(Nk[k],2)

    GMMi_evol.append( [ copy.deepcopy( PSR.GMMi[s] ) for s in range(S) ] )
    if isinstance(PSR, DiffPSR):
        a0_evol.append( [ a0k.clone() for a0k in PSR.a0 ] )

    ### EM step for GMM model
    PSR.GMM_opt(max_iterations=10)

    ### M step optimization for diffeomorphisms (individually for each k)
    PSR.Reg_opt(tol=1e-5)

    ### Plot resulting point sets (and some trajectories)

    plt.figure(S+1)
    plt.clf()
    for s in range(S):
        x0s = [ xk[s] for xk in x0 ]
        x1s = PSR.x1[:,s]
        PSR.GMMi[s].plot(*x0s, *x1s, bounds=[0,2, -0.5,1.5], heatmap=False, color=f"C{s}")
        my_scatter(*x1s[:4], alpha=.6)
    for k in range(4):
        ... # if trajectories are too cluttered
        #PSR.plot_trajectories(k)
        #PSR.plot_trajectories(k, support=True, linewidth=2, alpha=1)     # only useful in diffPSR class
    plt.show()
    plt.pause(.1)

print("Done !")
input()

