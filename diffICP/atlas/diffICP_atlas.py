'''
Test the diffICP algorithm on multiple point set registration (statistical atlas).
'''

import copy
import pickle
import matplotlib
from matplotlib import pyplot as plt
import torch

#pykeops.clean_pykeops()

# Manual random generator seeds (to always reproduce the same point sets if required)
torch.random.manual_seed(1234)

###################################################################
# Import from diffICP module

from diffICP.core.GMM import GaussianMixtureUnif
from diffICP.core.LDDMM import LDDMMModel
from diffICP.core.PSR import DiffPSR
from diffICP.tools.kernel import GaussKernel
from diffICP.tools.inout import read_point_sets
from diffICP.visualization.visu import my_scatter

##################################################################################
### Debug function : plot the current state of PSR model (GMM location, target points, trajectories etc.)

matplotlib.use('TkAgg')
def plot_step(PSR:DiffPSR, fig_index=None, only_GMM=False):
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

def diffICP_atlas(x0, model_parameters: dict,
                       support_scheme_options=None, optim_options=None, computversion='keops', plotstuff=True):
    '''
    Launch diffICP-based atlas building. This function showcases the use of class DiffPSR.

    :param x0: input data points. Several possible formats, e.g., x0[k][s] = cloud point from frame k and structure s
    :param model_parameters: dict with main model parameters :
        model_parameters["N_components_GMM"] : number of components in the fitted GMM model ;
        model_parameters["use_outliers_GMM"] : True/False, whether to add an outlier component in the GMM model ;
        model_parameters["optimize_weights_GMM"] : True/False, whether to optimize GMM component weights ;
        model_parameters["lambda_LDDMM"]: regularization term of the LDDMM framework ;
        model_parameters["sigma_LDDMM"]: spatial std of the RKHS Kernel defining LDDMM diffeomorphisms ;
        model_parameters["version_LDDMM"]: "logdet" (exact) or "hybrid" (faster) ;

    :param support_scheme_options: dict with options to define the location of LDDMM support points (see code below) ;
    :param optim_options: numerical options for the optimization procedure (see code below) ;
    :param computversion: 'keops' or 'torch' ;
    :param plotstuff: True/False, whether to plot model evolution during optimization (only in 2D) ;
    :return: PSR [main output, registration object after optim], evol [evolution of selected quantities over iterations]
    '''

    ######################
    # Check parameters, set default values

    assert {"lambda_LDDMM","sigma_LDDMM"}.issubset(model_parameters.keys()), \
        "model_parameters should at least define values of lambda_LDDMM and sigma_LDDMM"

    if "version_LDDMM" not in model_parameters.keys():
        model_parameters["version_LDDMM"] = "logdet"

    if "N_components_GMM" not in model_parameters.keys():
        model_parameters["N_components_GMM"] = 20

    if "optimize_weights_GMM" not in model_parameters.keys():
        model_parameters["optimize_weights_GMM"] = True

    if support_scheme_options is None:
        support_scheme_options = {"scheme":"grid",  # "dense", "grid" or "decim"
                                  "rho": 1.0 }      # remaining parameters to diffPSR.set_support_scheme()

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

    ### Create the DiffPSR object that will perform the registration

    # GMM model
    C = model_parameters["N_components_GMM"]
    use_outliers = model_parameters["use_outliers_GMM"]
    GMMi = GaussianMixtureUnif(torch.zeros(C,2), use_outliers=use_outliers)    # initial value for mu = whatever (will be changed by PSR algo)
    GMMi.to_optimize = {
        "mu" : True, "sigma" : True, "w" : model_parameters["optimize_weights_GMM"], "eta0" : use_outliers
    }

    # LDDMM model
    LMi = LDDMMModel(sigma = model_parameters["sigma_LDDMM"],               # sigma of the Gaussian kernel
                              D=D,                                          # dimension of space
                              lambd= model_parameters["lambda_LDDMM"],      # lambda of the LDDMM regularization
                              version = model_parameters["version_LDDMM"],  # "logdet" or "hybrid"
                              computversion=computversion,                  # "torch" or "keops"
                              scheme="Euler")                               # "Euler" (faster) or "Ralston" (more precise)

    PSR = DiffPSR(x0, GMMi, LMi)

    supp_scheme = support_scheme_options["scheme"]
    if supp_scheme != "dense":
        PSR.set_support_scheme(**support_scheme_options)

    ### And optimize !

    # for storing results
    evol = {"a0": [],        # evol["a0"][it][k] = current a0 tensor(Nk[k],2) at iteration it
            "GMMi": []}        # evol["GMMi"][it] = current GMM model at iteration it

    tol = optim_options["convergence_tolerance"]        # for the moment, same at all levels (TODO differentiate between LDDMM/GMM/global ?)

    last_FE = None                                      # Previous value of free energy

    for it in range(optim_options["max_iterations"]):
        print("ITERATION NUMBER ", it)

        evol["GMMi"].append(copy.deepcopy(PSR.GMMi[0]))
        evol["a0"].append([a0k.clone().detach().cpu() for a0k in PSR.a0])

        # EM step for GMM model
        PSR.GMM_opt(max_iterations=optim_options["max_repeat_GMM"], tol=tol)

        if plotstuff:
            plot_step(PSR, 2)
            plot_step(PSR, 3, only_GMM=True)

        # M step optimization for diffeomorphisms (individually for each k)
        PSR.Reg_opt(tol=tol, nmax=1)

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
            x0 = yo["x0"]               # Sample point sets
    else:
        N = 100
        x0, _, _ = generate_spiral_point_sets(K=1, Nkbounds=(N, N + 1),
                                              sigma_GMM=0.025,
                                              sigma_LDDMM=0.1, lambda_LDDMM=1e2)

    # Model parameters
    model_parameters = {"N_components_GMM": 20,
                        "use_outliers_GMM": False,
                        "lambda_LDDMM": 500,
                        "sigma_LDDMM": 0.2,
                        "version_LDDMM": "hybrid"}

    # Optimization options
    optim_options = {'max_iterations': 25,              # Maximm number of global loop iterations
                     'convergence_tolerance': 1e-3,     # for each optimization, including global loop itself (for the moment!)
                     'max_repeat_GMM': 25}

    # Launch
    PSR, evol = diffICP_atlas(x0, model_parameters,
                              optim_options=optim_options, computversion='keops', plotstuff=True)

    print("Final losses :")
    print(f"    regularity: {sum(PSR.regloss)})")
    print(f"    attachment: {PSR.quadloss.sum().item()}")
    print(f"    overall loss: {PSR.FE}")

    input()