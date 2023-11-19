'''
Use the ICP algorithm (diffeomorphic or affine) for multiple point set registration (statistical atlas).

Nota : running this file as a script executes an example usage of the function.
Else, simply import function ICP_atlas in other scripts to use it there.
'''

import copy, warnings
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
from diffICP.core.affine import AffineModel
from diffICP.core.PSR import MultiPSR, DiffPSR, AffinePSR
from diffICP.tools.spec import defspec
from diffICP.tools.in_out import read_point_sets
from diffICP.visualization.visu import my_scatter

##################################################################################
### 2d debug function : plot the current state of PSR model (GMM location, target points, trajectories etc.)

matplotlib.use('TkAgg')
def plot_state(PSR:MultiPSR, only_GMM=False):
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

def ICP_atlas(x0, GMM_parameters={}, registration_parameters={},
                       numerical_options={}, optim_options={}, callback_function=None):
    '''
    Launch ICP-based atlas building. This function showcases the use of class DiffPSR (resp. AffinePSR).

    :param x0: input data points. Several possible formats, e.g., x0[k][s] = cloud point from frame k and structure s

    :param GMM_parameters: dict with main model parameters for the GMM part:
        GMM_parameters["N_components"] : number of components (per structure) in the fitted GMM model ;
        GMM_parameters["optimize_weights"] : True/False, whether to optimize GMM component weights ;
        GMM_parameters["outlier_weight"] : None [no outlier component] or "optimize" [optimize weight] or float value [fixed log-odds ratio] ;
        GMM_parameters["initial_GMM"] : optional, impose a starting GMM model (overrides other parameters if present) ;

    :param registration_parameters: dict with main model parameters for the registration part:
        registration_parameters["type"] : "rigid" or "similarity" or "general_affine" or "diffeomorphic" ;
        registration_parameters["lambda_LDDMM"]: regularization term of the LDDMM framework (only used if diffeomorphic) ;
        registration_parameters["sigma_LDDMM"]: spatial std of the RKHS Kernel defining LDDMM diffeomorphisms ;

    :param numerical_options: dict with various numerical details of the algorithm:
        numerical_options["computversion"] : "keops" or "torch" ;
        numerical_options["support_LDDMM"] : dict, chosen LDDMM support scheme (see code below) ;
        numerical_options["gradcomponent_LDDMM"] : True (exact) or False (faster) ;
        etc. (see code below) ;

    :param optim_options: numerical options for the optimization procedure (see code below) ;

    :param callback_function: optional function to execute at every iteration of the optimization loop, e.g. for reporting or plotting.
        Must take the form callback_function(PSR, before_reg=False/True), with PSR the PSR object being currently optimized

    :return: PSR [main output, registration object after optim], evol [evolution of selected quantities over iterations]
    '''

    # TODO : handling of specs (gpu vs cpu) written but not tested. Expect failures when using different specs than defspec.

    ######################
    # Check mandatory model parameters (GMM and registration)

    if GMM_parameters.get("initial_GMM") is None:
        assert {"N_components", "optimize_weights"}.issubset(GMM_parameters.keys()), \
            "GMM_parameters should either\n" \
            "- define a new GMM model with values (at least) of N_components (int>0) and optimize_weights (True/False)\n" \
            "- define key 'initial_GMM' to provide an already existing GMM model."

    assert GMM_parameters.get("outlier_weight") is None or \
           GMM_parameters["outlier_weight"] == "optimize" or \
           isinstance(GMM_parameters["outlier_weight"], (int,float)), \
            "incorrect value for GMM_parameters['outlier_weight']"

    allowed_reg_types = ["rigid", "similarity", "general_affine", "diffeomorphic"]
    assert any([registration_parameters.get("type") == typ for typ in allowed_reg_types]), \
        f"registration_parameters['type'] should be one of: {allowed_reg_types}"

    is_diff = registration_parameters["type"] == "diffeomorphic"
    if is_diff:
        assert {"lambda_LDDMM","sigma_LDDMM"}.issubset(registration_parameters.keys()), \
            "if type=diffeomorphic, registration_parameters should define values of lambda_LDDMM and sigma_LDDMM"


    ######################
    # Set default values for optional arguments (numerical etc.)

    numerical_options = numerical_options.copy()
    optim_options = optim_options.copy()

    def set_default(dico, key, value):
        if dico.get(key) is None:
            dico[key] = value

    default_support_scheme = {
                "scheme": "grid",   # "dense", "grid" or "decim"
                "rho": 1.0}         # remaining parameters to diffPSR.set_support_scheme()

    set_default(numerical_options, "support_LDDMM", default_support_scheme)
    set_default(numerical_options, "computversion", "keops")
    set_default(numerical_options, "gradcomponent_LDDMM", False)           # False= approximate but generally sufficient :)
    set_default(numerical_options, "integration_scheme_LDDMM", "Euler")    # Euler (faster) vs Ralston (more precise)
    set_default(numerical_options, "integration_nt_LDDMM", 10)             # number of time steps
    set_default(numerical_options, "compspec", defspec)                    # 'compspec' = device for computations (gpu vs cpu)
    set_default(numerical_options, "dataspec", defspec)                    # 'dataspec' = device for storage (gpu vs cpu)

    set_default(optim_options, "max_iterations", 25)            # Maximum number of global loop iterations
    set_default(optim_options, "convergence_tolerance", 1e-3)   # Tolerance parameter (TODO differentiate between global loop and single optimizations ?)
    set_default(optim_options, "max_repeat_GMM", 10)            # Maximum number of EM steps in each GMM optimization loop

    compspec = numerical_options["compspec"]
    dataspec = numerical_options["dataspec"]

    #########################

    ### Read input point sets and various dimensions. Output:
    #   x0: point sets, now cast in the format x0[k][s] ;
    #   K = number of frames
    #   S = number of structures
    #   D = dimension of space

    x0, K, S, D = read_point_sets(x0)

    ### Create the MultiPSR object (Diff or Affine) that will perform the registration

    # GMM model
    is_initial_GMM = GMM_parameters.get("initial_GMM") is not None
    if is_initial_GMM:
        GMMi = copy.deepcopy(GMM_parameters["initial_GMM"])                         # (Nota: could lead to spec clashes in some weird cases)
    else:
        C = GMM_parameters["N_components"]
        use_outliers = GMM_parameters.get("outlier_weight") is not None
        GMMi = GaussianMixtureUnif(torch.zeros(C,D), use_outliers=use_outliers,     # initial value for mu = whatever (will be changed by PSR algo)
                                   spec=compspec)
        if isinstance(GMM_parameters.get("outlier_weight"), (int,float)):
            GMMi.outliers["eta0"] = GMM_parameters.get("outlier_weight")
        GMMi.to_optimize = {
            "mu" : True, "sigma" : True,
            "w" : GMM_parameters["optimize_weights"],
            "eta0" : GMM_parameters.get("outlier_weight") == "optimize"
        }

    if is_diff:
        # LDDMM registration model
        LMi = LDDMMModel(sigma= registration_parameters["sigma_LDDMM"],             # sigma of the Gaussian kernel
                        D= D,                                                       # dimension of space
                        lambd=  registration_parameters["lambda_LDDMM"],            # lambda of the LDDMM regularization
                        withlogdet= True,
                        gradcomponent = numerical_options["gradcomponent_LDDMM"],   # True or False
                        computversion = numerical_options["computversion"],         # "torch" or "keops"
                        scheme= numerical_options["integration_scheme_LDDMM"],      # "Euler" (faster) or "Ralston" (more precise)
                        spec= compspec,                                             # device and type of data
                        nt= numerical_options["integration_nt_LDDMM"])

        PSR = DiffPSR(x0, GMMi, LMi, compspec=compspec, dataspec=dataspec)

        supp_scheme = numerical_options["support_LDDMM"]["scheme"]
        if supp_scheme != "dense":
            PSR.set_support_scheme(**numerical_options["support_LDDMM"])

        # for storing results
        evol = {"a0": [],        # evol["a0"][it][k] = current a0 tensor(Nk[k],2) at iteration it
                "GMMi": []}      # evol["GMMi"][it] = current GMM model at iteration it

    else:
        # Affine registration model
        AffMi = AffineModel(D=D, version=registration_parameters["type"],
                            withlogdet=True,
                            with_t=True)

        PSR = AffinePSR(x0, GMMi, AffMi, compspec=compspec, dataspec=dataspec)

        # for storing results
        evol = {"M": [],            # evol["M"][it][k] = current registration matrix for frame k at iteration it
                "t": [],            # evol["t"][it][k] = current translation vector for frame k at iteration it
                "GMMi": []}        # evol["GMMi"][it] = current GMM model at iteration it

    #########################
    ### And optimize !

    tol = optim_options["convergence_tolerance"]        # for the moment, same at all levels (TODO differentiate between LDDMM/GMM/global ?)

    last_FE = None                                      # Previous value of free energy

    for it in range(optim_options["max_iterations"]):
        print("ITERATION NUMBER ", it)

        evol["GMMi"].append(copy.deepcopy(PSR.GMMi[0]))
        if is_diff:
            evol["a0"].append([a0k.clone().detach().cpu() for a0k in PSR.a0])
        else:
            evol["M"].append([Mk.clone().detach().cpu() for Mk in PSR.M])
            evol["t"].append([tk.clone().detach().cpu() for tk in PSR.t])

        # EM step for GMM model
        if not (is_initial_GMM and it == 0):    # (in that case, start by optimizing registrations)
            PSR.GMM_opt(max_iterations=optim_options["max_repeat_GMM"], tol=tol)

        if callback_function is not None:
            callback_function(PSR, True)

        # M step optimization for registrations (individually for each k)
        PSR.Reg_opt(tol=tol, nmax=1)

        if callback_function is not None:
            callback_function(PSR, False)

        if it > 1 and abs(PSR.FE-last_FE) < tol * abs(last_FE):
            print("Difference in Free Energy is below tolerance threshold : optimization is over.")
            break

        last_FE = PSR.FE

    # DONE !
    if it+1 == optim_options["max_iterations"]:
        print("Reached maximum number of iterations (before reaching convergence threshold).")

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
        K = 10
        x0, _, _ = generate_spiral_point_sets(K=K, Nkbounds=(N, N + 40),
                                              sigma_GMM=0.025,
                                              sigma_LDDMM=0.1, lambda_LDDMM=1e2)

    GMM_parameters = {"N_components": 20,
                      "optimize_weights": True,
                      "outlier_weight": None}

    registration_parameters = {"type": "diffeomorphic",
                               "lambda_LDDMM": 500,
                               "sigma_LDDMM": 0.2}

    # registration_parameters = {"type": "similarity"}      # affine version

    optim_options = {'max_iterations': 125,              # Maximum number of global loop iterations
                     'convergence_tolerance': 1e-3,     # for each optimization, including global loop itself (for the moment!)
                     'max_repeat_GMM': 25}

    # numerical parameters : use default for the moment

    def callback_plot(PSR, before_reg):
        plt.figure(1)
        plot_state(PSR)
        if before_reg:
            plt.figure(2)
            plot_state(PSR, only_GMM=True)

    # Launch
    PSR, evol = ICP_atlas(x0, GMM_parameters, registration_parameters,
                              optim_options=optim_options, callback_function=callback_plot)

    print("Final losses :")
    print(f"    regularity: {sum(PSR.regloss)})")
    print(f"    attachment: {PSR.quadloss.sum().item()}")
    print(f"    overall loss: {PSR.FE}")

    input()