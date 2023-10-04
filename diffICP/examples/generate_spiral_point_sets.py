'''
Generate some "spiral" point sets, using a "diffeomorphically-warped GMM" distribution.

NOTA: the "spiral" shape formula comes from a KeOps tutorial
'''

import pickle
import numpy as np
from matplotlib import pyplot as plt
plt.ion()

import torch

# Manual random generator seeds (to always reproduce the same point sets if required)
torch.random.manual_seed(1234)

# Import from diffICP module

from diffICP.core.GMM import GaussianMixtureUnif
from diffICP.core.LDDMM import LDDMMModel
from diffICP.visualization.visu import my_scatter

##########################################################################################"

def generate_spiral_point_sets(K=10, Nkbounds=(100,121), sigma_GMM=0.025, sigma_LDDMM=0.1, lambda_LDDMM=1e2):
    '''
    Generate point sets following a 2d 'spiral' geometry. (Used for tests.)

    :param K: Number of point sets to generate
    :param Nkbounds: =(Nmin,Nmax), bounds (min and max) for the random number of points in each point set
    :param sigma_GMM: Sigma of the GMM model (==added noise for points around the centroids)
    :param sigma_LDDMM:  Sigma of the LDDMM kernel (==spatial scale of deformations)
    :param lambda_LDDMM: Lambda of the LDDMM model. Higher lambda -> less deformations.
    :return: x0 (list of generated point sets), GMMg (generative GMM model), LMg (generative LDDMM model)
    '''

    ### Spiral formula to draw (deterministic) centroids for GMMg. This is fixed.
    C = 20
    t = torch.linspace(0, 2 * np.pi, C + 1)[:-1]
    mu0 = torch.stack((0.5 + 0.4 * (t / 7) * t.cos(), 0.5 + 0.3 * t.sin()), 1)

    ### Generative GMM model with centroids mu0
    GMMg = GaussianMixtureUnif(mu0)
    GMMg.sigma = sigma_GMM                  # sigma of the GMM distribution

    ### Generative LDDMM model
    LMg = LDDMMModel(sigma = sigma_LDDMM,   # sigma of the Gaussian kernel
                     D=2,                   # dimension of space
                     lambd= lambda_LDDMM,   # lambda of the LDDMM regularization
                     version = "classic",
                     nt = 10)               # time discretization of interval [0,1] for ODE resolution

    ### Generate samples
    Nk = torch.randint(*Nkbounds, (K,))         # (Random) number of points in each point set
    print("Generating 'spiral' point sets. Number of point sets: ", K)
    print("Generating 'spiral' point sets. Number of points in each point set:\n", Nk)

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

    return x0, GMMg, LMg


##########################################################################################"
### If called as a script, generate a first group of 'spiral' point sets :

if __name__ == '__main__':
    # Running as a script

    # Generate some points
    x0, GMMg, LMg = generate_spiral_point_sets(K=10, Nkbounds=(100,121), sigma_GMM=0.025, sigma_LDDMM=0.1, lambda_LDDMM=100)

    savestuff = True
    # Nota: working directory is always assumed to be the Python project home (hence, no need for ../ to return to home directory)
    # When the IDE used is Pycharm, this requires to set the default run directory, as follows:
    # Main Menu > Run > Edit Configurations > Edit Configuration templates > Python > Working directory [-> select project home dir]
    savefile = "saving/sample_spiral_points_1.pkl"
    savelist = [ "GMMg","LMg","x0" ]
    if savestuff:
        print("Saving stuff")
        tosave = {k:globals()[k] for k in savelist}
        with open(savefile, 'wb') as f:
            pickle.dump(tosave, f)

    plotstuff = True
    if plotstuff:
        GMMg.plot(*x0)
        my_scatter(*x0[:5])
        plt.pause(.2)

    input()