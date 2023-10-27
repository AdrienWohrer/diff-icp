'''
Basic example : LDDMM registration of a point set to a fixed GMM model
'''

import time
import pickle

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
plt.ion()

#pykeops.clean_pykeops()

# Manual random generator seeds (to always reproduce the same point sets if required)
#torch.random.manual_seed(1234)


###################################################################
# Import from diffICP module

from diffICP.core.LDDMM import LDDMMModel
from diffICP.core.PSR import DiffPSR, AffinePSR
from diffICP.visualization.visu import my_scatter
from diffICP.examples.generate_spiral_point_sets import generate_spiral_point_sets


###################################################################
# Saving simulation results
savestuff = False
# Nota: working directory is always assumed to be the Python project home (hence, no need for ../ to return to home directory)
# When the IDE used is Pycharm, this requires to set the default run directory, as follows:
# Main Menu > Run > Edit Configurations > Edit Configuration templates > Python > Working directory [-> select project home dir]
savefile = "saving/test_basic.pkl"
savelist = []       # store names of variables to be saved

# Plot figures ?
plotstuff = True

# Number of global loop iterations
nIter = 20

# Also optimize sigma ?
sigma_opt = True
sigma_start = 0.1   # starting sigma for GMMi (if sigma_opt=True) or FIXED value for GMMi (if sigma_opt=False)
                    # set at None to use the exact value from generating model GMMg


###################################################################
### Part 1 : Synthetic data : 'spiral' GMM + sample point set
###################################################################

reload = True
if reload:
    loadfile = "saving/sample_spiral_points_1.pkl"
    print("Loading data points from existing file : ",loadfile)
    with open(loadfile, 'rb') as f:
        yo = pickle.load(f)
    for key in ["GMMg","LMg","x0"]:
        globals()[key] = yo[key]
    x0 = x0[0]          # single point set (here id=0 -- can also test other sets from 0 to 9)
    N = x0.shape[0]     # number of points

else:
    N = 100
    x0, GMMg, LMg = generate_spiral_point_sets(K=1, Nkbounds=(N,N+1),
                                               sigma_GMM=0.025,
                                               sigma_LDDMM=0.1, lambda_LDDMM=1e2)
    x0 = x0[0]

GMMg.to_optimize = {'mu':False, 'sigma':sigma_opt, 'w':False}   # fixed parameters

### Variables that will be saved
savelist.extend(("GMMg","LMg","N","x0"))

###################################################################
### Check

if plotstuff:
    plt.figure()
    GMMg.plot(x0)
    my_scatter(x0)
#    plot_shoot(shoot,color='b')
    plt.pause(1)


###################################################################
### Part 2 : Registration on fixed GMM model (new diffICP algorithm)
###################################################################

### Point Set Registration model : diffeomorphic version

LMi = LDDMMModel(sigma = 0.2,                           # sigma of the Gaussian kernel
                          D=2,                          # dimension of space
                          lambd= 5e2,                   # lambda of the LDDMM regularization
                          version = "classic",          # "logdet", "classic" or "hybrid"
                          computversion="keops",        # "torch" or "keops"
                          scheme="Euler")               # "Euler" or "Ralston"

PSR = DiffPSR(x0, GMMg, LMi)
# Add a decimation scheme ?
PSR.set_support_scheme("grid", rho=np.sqrt(2))
# PSR.set_support_scheme("decim", rho=0.7)

### Point Set Registration model : affine version

# PSR = AffinePSR(x0, GMMg, AffineModel(D=2, version = 'similarity'))
# PSR = AffinePSR(x0, GMMg, AffineModel(D=2, version = 'euclidian', withlogdet=False))

# To change sigma also and optimize it (as in classic 2-point set probablistic ICP)
if sigma_start is not None:
    PSR.GMMi[0].sigma = sigma_start     # (else, keep true value from GMMg)

# for storing parameter evolution in time
param_evol = []

### Optimization loop

if plotstuff:
    plt.figure()

start = time.time()
for it in range(nIter):
    print("ITERATION NUMBER ",it)

    ### Store stuff (for saving results to file)
    # param_evol[version][it] = current a0 tensor(N,2)

    if isinstance(PSR, DiffPSR):
        par = {'a0':PSR.a0[0].cpu()}
    elif isinstance(PSR, AffinePSR):
        par = {'M':PSR.M[0].cpu(), 't':PSR.t[0].cpu()}
    par['sigma'] = PSR.GMMi[0].sigma
    param_evol.append(par)

    ### E step for GMM model
    PSR.GMM_opt()

    ###LDDMM step optimization for diffeomorphisms
    PSR.Reg_opt(tol=1e-5)

    if sigma_opt:
        print(f"Sigma: {PSR.GMMi[0].sigma}")

    ### Plot resulting point sets (and some trajectories)

    if plotstuff:
        plt.clf()
        x1 = PSR.x1[0,0]
        PSR.GMMi[0].plot( x0, x1 )
        my_scatter(PSR.GMMi[0].mu, alpha=.6, color="b")
        my_scatter(x1, alpha=.6, color="r")
        PSR.plot_trajectories(color='red')
        if isinstance(PSR, DiffPSR) and PSR.support_scheme:
            # also visualize support trajectories in brown
            PSR.plot_trajectories(color='brown', support=True, linewidth=2, alpha=1)
        #plt.show()
        plt.xticks(np.arange(-0.5, 1.5 + 0.1, 0.5))
        plt.yticks(np.arange(-0.5, 1.5 + 0.1, 0.5))
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))  # Ensure correctly formated ticks
        plt.gca().set_aspect('equal')
        plt.gca().autoscale(tight=True)
        plt.pause(.1)

        #print(PSR.x1[0,0][:5])  # debug

# Done.

print(f"Elapsed time : {time.time()-start} seconds")
savelist.extend(("PSR","param_evol"))

if savestuff:
    print("Saving stuff")
    tosave = {k:globals()[k] for k in savelist}
    with open(savefile, 'wb') as f:
        pickle.dump(tosave, f)

# Fini ! Wait for click
if plotstuff:
    print("Done.")
    input()
