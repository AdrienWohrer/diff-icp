'''
Test the ICP algorithm (diffeomorphic or affine) on classic ("two set") registration.

Nota : running this file as a script provides an example usage of the function.
Else, simply import function ICP_two_set in other scripts to use it there.
'''

import math, copy
import pickle
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
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
from diffICP.tools.kernel import GaussKernel
from diffICP.tools.inout import read_point_sets
from diffICP.visualization.visu import my_scatter, get_bounds, on_top
from diffICP.visualization.grid import Gridlines
from diffICP.tools.spec import defspec

##################################################################################
### Default visualization function : plot the current state of PSR model (GMM location, target points, trajectories etc.)

matplotlib.use('TkAgg')

def plot_state(PSR: MultiPSR, bounds, plot_GMM=True, plot_targets=True, plot_gridlines=True):

    plt.clf()
    x1 = PSR.get_warped_data_points()
    ### GMM Heatmap and contours
    if plot_GMM:
        PSR.GMMi[0].plot(bounds=bounds, color="#A1C8C8", cmap="RdBu", heatmap_amplification=0.7)  # https://matplotlib.org/stable/gallery/color/colormap_reference.html
    ### Association between each point and its quadratic target
    if plot_targets:
        assoc = torch.stack((x1, PSR.y[0,0]))
        for n in range(x1.shape[0]):
            plt.plot(assoc[:,n,0], assoc[:,n,1], color="purple", linewidth=0.5)
    ### Grid lines
    if plot_gridlines:
        gridlines = Gridlines(np.linspace(bounds[0],bounds[1],10), np.linspace(bounds[2],bounds[3],10))
        # gridlines.plot(color='gray', linewidth=1, linestyle='dotted')
        reglines = gridlines.register(PSR.Registration())
        reglines.plot(color=(0.8,0.5,0.5), linewidth=1)
    ### Two point sets
    my_scatter(PSR.GMMi[0].mu, alpha=1, color="b")
    my_scatter(x1, alpha=1, color="red")

    plt.xticks(np.arange(math.floor(bounds[0]), math.ceil(bounds[1]) + 0.1, 0.5))
    plt.yticks(np.arange(math.floor(bounds[2]), math.ceil(bounds[3]) + 0.1, 0.5))
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))  # Ensure correctly formated ticks
    plt.gca().set_aspect('equal')
    plt.xlim(bounds[:2])
    plt.ylim(bounds[2:])
    on_top(plt.gcf())        # from diffICP.visu
    # plt.gca().autoscale(tight=True)
    plt.pause(.1)

##################################################################################
##################################################################################
##################################################################################

def ICP_two_set(xA, xB, GMM_parameters: dict, registration_parameters: dict,
                       numerical_options={}, optim_options={}, plotstuff=True):
    '''
    Launch ICP-based two-set registration. This function showcases the use of class DiffPSR (resp. AffinePSR).

    :param xA: first point set ("data", to register).
    :param xB: second point set ("template", serves as centroids of a GMM model).

    :param GMM_parameters: dict with main model parameters for the GMM part :
        GMM_parameters["sigma"] (float) : initial value of GMM sigma parameter ;
        GMM_parameters["optimize_sigma"] : True/False, whether to optimize GMM sigma parameter ;
        GMM_parameters["outlier_weight"] : None [no outlier component] or "optimize" [optimize weight] or float value [fixed log-odds ratio] ;

    :param registration_parameters: dict with main model parameters for the registration part :
        registration_parameters["type"] : "rigid" or "similarity" or "general_affine" or "diffeomorphic" ;
        registration_parameters["lambda_LDDMM"]: regularization term of the LDDMM framework (only used if diffeomorphic) ;
        registration_parameters["sigma_LDDMM"]: spatial std of the RKHS Kernel defining LDDMM diffeomorphisms ;

    :param numerical_options: dict with various numerical details of the algorithm :
        numerical_options["computversion"] : "keops" or "torch" ;
        numerical_options["support_LDDMM"] : dict, chosen LDDMM support scheme (see code below) ;
        numerical_options["gradcomponent_LDDMM"] : True (exact) or False (faster) ;
        etc. (see code below) ;

    :param optim_options: numerical options for the optimization procedure (see code below) ;

    :param plotstuff: True/False, whether to plot model evolution during optimization (only in 2D) ;
    :return: PSR [main output, registration object after optim], evol [evolution of selected quantities over iterations]
    '''

    # TODO transform plotstuff into callback function if necessary
    # TODO handle specs if necessary

    ######################
    # Check mandatory model parameters (GMM and registration)

    allowed_reg_types = ["rigid", "similarity", "general_affine", "diffeomorphic"]
    assert any([registration_parameters["type"] == typ for typ in allowed_reg_types]), \
        f"registration_parameters['type'] should be one of: {allowed_reg_types}"

    is_diff = registration_parameters["type"] == "diffeomorphic"
    if is_diff:
        assert {"lambda_LDDMM","sigma_LDDMM"}.issubset(registration_parameters.keys()), \
            "if type=diffeomorphic, registration_parameters should define values of lambda_LDDMM and sigma_LDDMM"

    assert {"optimize_sigma", "sigma"}.issubset(GMM_parameters.keys()), \
        "GMM_parameters should at least define values of sigma (float>0) and optimize_sigma (True/False)"

    assert GMM_parameters["outlier_weight"] is None or \
           GMM_parameters["outlier_weight"] == "optimize" or \
           isinstance(GMM_parameters["outlier_weight"], (int,float)), \
            "incorrect value for GMM_parameters['outlier_weight']"

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

    set_default(optim_options, "max_iterations", 25)            # Maximum number of global loop iterations
    set_default(optim_options, "convergence_tolerance", 1e-3)   # Tolerance parameter (TODO differentiate between global loop and single optimizations ?)
    set_default(optim_options, "max_repeat_GMM", 10)            # Maximum number of EM steps in each GMM optimization loop

    #########################

    ### Read input point sets and various dimensions.

    D, DB = xA.shape[1], xB.shape[1]
    assert D == DB, "point sets xA and xB should have same vector dimension (dim 1)"

    bounds = get_bounds(xA, xB, relmargin=0.1)

    ### Create the MultiPSR object (Diff or Affine) that will perform the registration

    # GMM model
    use_outliers = GMM_parameters.get("outlier_weight") is not None
    GMMi = GaussianMixtureUnif(xB, use_outliers=use_outliers)
    GMMi.sigma = GMM_parameters["sigma"]
    if isinstance(GMM_parameters.get("outlier_weight"), (int, float)):
        GMMi.outliers["eta0"] = GMM_parameters["outlier_weight"]
    GMMi.to_optimize = {
        "mu" : False,
        "sigma" : GMM_parameters["optimize_sigma"],
        "w" : False,
        "eta0" : GMM_parameters.get("outlier_weight") == "optimize"
    }

    if is_diff:
        # LDDMM registration model
        LMi = LDDMMModel(sigma= registration_parameters["sigma_LDDMM"],             # sigma of the Gaussian kernel
                        D= D,                                                       # dimension of space
                        lambd=  registration_parameters["lambda_LDDMM"],            # lambda of the LDDMM regularization
                        withlogdet= True,
                        computversion = numerical_options["computversion"],         # "torch" or "keops"
                        scheme= numerical_options["integration_scheme_LDDMM"],      # "Euler" (faster) or "Ralston" (more precise)
                        nt= numerical_options["integration_nt_LDDMM"])

        PSR = DiffPSR(xA, GMMi, LMi)

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

        PSR = AffinePSR(xA, GMMi, AffMi)

        # for storing results
        evol = {"M": [],            # evol["M"][it][k] = current registration matrix for frame k at iteration it
                "t": [],            # evol["t"][it][k] = current translation vector for frame k at iteration it
                "GMMi": []}        # evol["GMMi"][it] = current GMM model at iteration it

    #########################

    if plotstuff:
        plt.figure()    # Figure 1 : basic point sets
        plot_state(PSR, bounds, plot_GMM=False, plot_targets=False, plot_gridlines=False)
        # if savefigs:      # TODO update if required
        #     savefigs_path = os.path.dirname(os.path.realpath(savefigs_name))
        #     os.makedirs(savefigs_path, exist_ok=True)
        #     plt.savefig(f"{savefigs_name}_data.{savefigs_format}", format=savefigs_format, bbox_inches='tight')

        plt.figure()    # Figure 2 : algorithm evolution

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
        PSR.GMM_opt(max_iterations=optim_options["max_repeat_GMM"], tol=tol)

        if plotstuff:
            plot_state(PSR, bounds)

        # M step optimization for diffeomorphisms (individually for each k)
        PSR.Reg_opt(tol=tol, nmax=1)

        if plotstuff:
            plot_state(PSR, bounds)

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

    import scipy.io

    chui_dataset = 3  # (1 to 5)
    yo = scipy.io.loadmat(f"diffICP/examples/chui-data/demodata_ex{chui_dataset}.mat")
    x_name = f'x{[1, 2, 3, 1, 1][chui_dataset - 1]}'  # name of the variables in the chui file (no general rule :))
    y_name = f'y{[1, 2, 3, "2a", "2a"][chui_dataset - 1]}'  # name of the variables in the chui file (no general rule :))
    xA = torch.tensor(yo[x_name], **defspec).contiguous()  # xA will correspond to fixed GMM centroids
    xB = torch.tensor(yo[y_name], **defspec).contiguous()  # xB will be the registered point set
    if False:  # try the reversed version
        xB, xA = xA, xB

    # GMM parameters
    GMM_parameters = {"sigma": 0.1,
                      "optimize_sigma": True,
                      "outlier_weight": None}

    # Registration parameters
    registration_parameters = {"type": "diffeomorphic",     # or "similarity", ...
                        "lambda_LDDMM": 200,        # (only used in "diffeomorphic" case, but can be left anyway)
                        "sigma_LDDMM": 0.2}         # (idem)

    # registration_parameters = {"type": "similarity"}      # affine version

    # Numerical parameters : use default for the moment

    # Optimization options
    optim_options = {'max_iterations': 50,              # Maximm number of global loop iterations
                     'convergence_tolerance': 1e-5,     # for each optimization, including global loop itself (for the moment!)
                     'max_repeat_GMM': 25}

    # Launch
    PSR, evol = ICP_two_set(xA, xB, GMM_parameters, registration_parameters,
                              optim_options=optim_options, plotstuff=True)

    print("Final losses :")
    print(f"    regularity: {sum(PSR.regloss)})")
    print(f"    attachment: {PSR.quadloss.sum().item()}")
    print(f"    overall loss: {PSR.FE}")

    input()