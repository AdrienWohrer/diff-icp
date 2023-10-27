'''
Affine ICP algorithm (à la Horaud-Evangelidis) on multiple point set registration (statistical atlas).

Nota : running this file as a script provides an example usage of the function.
Else, simply import function affineICP_atlas in other scripts to use it there.
'''

import time, copy
import pickle
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import torch

#pykeops.clean_pykeops()

# Manual random generator seeds (to always reproduce the same point sets if required)
torch.random.manual_seed(1234)

###################################################################
# Import from diffICP module

from diffICP.core.GMM import GaussianMixtureUnif
from diffICP.core.affine import AffineModel
from diffICP.core.PSR import AffinePSR
from diffICP.tools.kernel import GaussKernel
from diffICP.tools.inout import read_point_sets
from diffICP.visualization.visu import my_scatter

##################################################################################
### Debug function : plot the current state of PSR model (GMM location, target points, trajectories etc.)

matplotlib.use('TkAgg')
def plot_step(PSR:AffinePSR, fig_index=None, only_GMM=False):
    plt.figure(fig_index)
    plt.clf()

    x0 = PSR.x0[:, 0]
    x1 = PSR.x1[:, 0]

    PSR.GMMi[0].plot(*x0, *x1)
    if not only_GMM:
        my_scatter(*x1[0:min(5, PSR.K)], alpha=.6)
        for k in range(min(5, PSR.K)):
            PSR.plot_trajectories(k)
            # PSR.plot_trajectories(k, support=True, linewidth=2, alpha=1)  # only useful in diffPSR class
    plt.show()
    plt.pause(.1)


##################################################################################
##################################################################################
##################################################################################

def affineICP_atlas(x0, model_parameters: dict,
                       optim_options=None, computversion='keops', plotstuff=True):
    '''
    Launch affine ICP-based atlas building. This function showcases the use of class AffinePSR.

    :param x0: input data points. Several possible formats, e.g., x0[k][s] = cloud point from frame k and structure s
    :param model_parameters: dict with main model parameters :
        model_parameters["N_components_GMM"] : number of components in the fitted GMM model ;
        model_parameters["use_outliers_GMM"] : True/False, whether to add an outlier component in the GMM model ;
        model_parameters["optimize_weights_GMM"] : True/False, whether to optimize GMM component weights ;
        model_parameters["version_affine"]: "rigid" (t+rotation), "similarity" (t+rotation+scaling), "translation" (only t), "affine" (t+general M, not implemented yet!) ;

    :param optim_options: numerical options for the optimization procedure (see code below) ;
    :param computversion: 'keops' or 'torch' (unused for the moment) ;
    :param plotstuff: True/False, whether to plot model evolution during optimization (only in 2D) ;
    :return: PSR [main output, registration object after optim], evol [evolution of selected quantities over iterations]
    '''

    ######################
    # Check parameters, set default values

    if "version_affine" not in model_parameters.keys():
        model_parameters["version_affine"] = "similarity"

    if "N_components_GMM" not in model_parameters.keys():
        model_parameters["N_components_GMM"] = 20

    if "optimize_weights_GMM" not in model_parameters.keys():
        model_parameters["optimize_weights_GMM"] = True

    if optim_options is None:
        optim_options = {'max_iterations': 25,              # Maximum number of global loop iterations
                         'convergence_tolerance': 1e-5,     # tolerance parameter (TODO differentiate between global loop and single optimizations ?)
                         'max_repeat_GMM': 10               # Maximum number of EM steps in each GMM optimization loop
                         }

    #########################

    ### Read input point sets and various dimensions. Output:
    #   x0: point sets, now cast in the format x0[k][s] ;
    #   K = number of frames
    #   S = number of structures
    #   D = dimension of space

    x0, K, S, D = read_point_sets(x0)
    if S > 1:
        raise ValueError("This function does not allow multiple structures, for the moment.")

    ### Create the AffinePSR object that will perform the registration

    # GMM model
    C = model_parameters["N_components_GMM"]
    use_outliers = model_parameters["use_outliers_GMM"]
    GMMi = GaussianMixtureUnif(torch.zeros(C,2), use_outliers=use_outliers)    # initial value for mu = whatever (will be changed by PSR algo)
    GMMi.to_optimize = {
        "mu" : True, "sigma" : True, "w" : model_parameters["optimize_weights_GMM"], "eta0" : use_outliers
    }

    # Affine model
    AffMi = AffineModel(D=D, version = model_parameters["version_affine"],
                        withlogdet=True, with_t=True)       # (withlogdet=True required to have a rigorous probabilistic interpretation)

    PSR = AffinePSR(x0, GMMi, AffMi)

    ### And optimize !

    # for storing results
    evol = {"M": [],            # evol["M"][it][k] = current registration matrix for frame k at iteration it
            "t": [],            # evol["t"][it][k] = current translation vector for frame k at iteration it
            "GMMi": []}        # evol["GMMi"][it] = current GMM model at iteration it

    tol = optim_options["convergence_tolerance"]        # for the moment, same at all levels (TODO differentiate between GMM/global ?)

    last_FE = None                                      # Previous value of free energy

    for it in range(optim_options["max_iterations"]):
        print("ITERATION NUMBER ", it)

        evol["GMMi"].append(copy.deepcopy(PSR.GMMi[0]))
        evol["M"].append([Mk.clone().detach().cpu() for Mk in PSR.M])
        evol["t"].append([tk.clone().detach().cpu() for tk in PSR.t])

        # EM step for GMM model
        PSR.GMM_opt(max_iterations=optim_options["max_repeat_GMM"], tol=tol)

        if plotstuff:
            plot_step(PSR, 2)
            plot_step(PSR, 3, only_GMM=True)

        # M step optimization for affine transformations
        PSR.Reg_opt()

        if plotstuff:
            plot_step(PSR, 2)

        if it > 1 and abs(PSR.FE-last_FE) < tol * abs(last_FE):
            print("Difference in Free Energy is below tolerance threshold : optimization is over.")
            break

        last_FE = PSR.FE

    # DONE !

    return PSR, evol


################################################################################
################################################################################
### Testing
################################################################################
################################################################################

if __name__ == '__main__':
    # Running as a script

    import os
    from diffICP.examples.generate_spiral_point_sets import generate_spiral_point_sets

    # Recover or create some sample point sets
    datafile = "saving/sample_spiral_points_1.pkl"
    if os.path.exists(datafile):
        with open(datafile, 'rb') as f:
            yo = pickle.load(f)
            x0 = yo["x0"]  # Sample point sets
    else:
        N = 100
        x0, _, _ = generate_spiral_point_sets(K=1, Nkbounds=(N, N + 1),
                                              sigma_GMM=0.025,
                                              sigma_LDDMM=0.1, lambda_LDDMM=1e2)

    # Model parameters
    model_parameters = {"N_components_GMM": 20,
                        "use_outliers_GMM": False,
                        "version_affine": "affine"}

    # Optimization options
    optim_options = {'max_iterations': 25,              # Maximm number of global loop iterations
                     'convergence_tolerance': 1e-5,     # for each optimization, including global loop itself (for the moment!)
                     'max_repeat_GMM': 25}

    # Launch
    PSR, evol = affineICP_atlas(x0, model_parameters,
                                optim_options=optim_options, computversion='keops', plotstuff=True)

    print("Final losses :")
    print(f"    registration (including logdet term): {sum(PSR.regloss)})")
    print(f"    attachment: {PSR.quadloss.sum().item()}")
    print(f"    overall loss: {PSR.FE}")

    input()