'''
"Standard" diffeomorphic point-set-matching algorithm from Glaunès et al 04, with a personal re-implementation.

Results should be compared to our own method for two-set registration (diffICP).

Nota : running this file as a script provides an example usage of the function.
Else, simply import function standard_atlas in other scripts to use it there.
'''

import math, copy
import pickle

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import torch

from diffICP.core.LDDMM import LDDMMModel
from diffICP.core.affine import AffineModel
from diffICP.core.GMM import GaussianMixtureUnif

from diffICP.tools.spec import defspec
from diffICP.tools.kernel import GaussKernel
from diffICP.visualization.visu import my_scatter, get_bounds, on_top
from diffICP.visualization.grid import Gridlines

from diffICP.core.PSR_standard import MultiPSR_std, DiffPSR_std, AffinePSR_std

##################################################################################
### Default visualization function : plot the current state of PSR model (template location, target points, trajectories etc.)

matplotlib.use('TkAgg')

def plot_state(PSR: MultiPSR_std, bounds, plot_gridlines=True):

    plt.clf()

    if plot_gridlines:
        gridlines = Gridlines(np.linspace(bounds[0], bounds[1], 10), np.linspace(bounds[2], bounds[3], 10))
        # gridlines.plot(color='gray', linewidth=1, linestyle='dotted')
        reglines = gridlines.register(PSR.Registration())
        reglines.plot(color=(0.8, 0.5, 0.5), linewidth=1)
    ### Two point sets
    my_scatter(PSR.get_data_points(), alpha=1, color="b")
    my_scatter(PSR.get_warped_template(), alpha=1, color="red")

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

def standard_two_set(xA, xB, model_parameters: dict,
                       numerical_options={}, optim_options={}, plotstuff=True, printstuff=True):
    '''
    Launch standard LDDMM two-set registration, using personal reimplementation in PSR_standard.py.

    :param xA: first point set ("template", will be deformed to fit the data).
    :param xB: second point set ("data", fixed).

    :param model_parameters: dict with main model parameters :
        model_parameters["type"] : "rigid" or "similarity" or "general_affine" or "diffeomorphic" ;
        model_parameters["sigma_data"]: spatial std of the RKHS Kernel used to define the data distance, (K(x)=exp(-x^2/2*sigma^2)) ;
        model_parameters["noise_std"]: scaling parameter of the data loss term, so Loss = \sum_s rkhs_distance(s) / noise_std[s]^2 ;
        model_parameters["sigma_LDDMM"]: spatial std of the RKHS Kernel defining LDDMM diffeomorphisms (only used when type='diffeomorphic') ;

    :param numerical_options: dict with various numerical details of the algorithm :
        numerical_options["computversion"] : "keops" or "torch" ;
        numerical_options["support_LDDMM"] : dict, chosen LDDMM support scheme (see code below) ;
        etc. (see code below) ;

    :param optim_options: numerical options for the optimization procedure (see code below) ;

    :param plotstuff: True/False, whether to plot model evolution during optimization (only in 2D) ;
    :param printstuff: True/False, whether to print evolution information during optimization ;
    :return: PSR [main output, registration object after optim], evol [evolution of selected quantities over iterations]
    '''

    # TODO transform plotstuff into callback function if necessary
    # TODO handle specs if necessary

    ######################
    # Check mandatory model parameters

    is_diff = model_parameters["type"] == "diffeomorphic"
    if is_diff:
        assert {"sigma_data", "noise_std", "sigma_LDDMM"}.issubset(model_parameters.keys()), \
            "if type=diffeomorphic, model_parameters should at least define values of sigma_data, noise_std and sigma_LDDMM"
    else:
        assert {"type", "sigma_data"}.issubset(model_parameters.keys()), \
            "model_parameters should at least define values of 'type' and 'sigma_data' (and more if type=diffeomorphic)"

    ######################
    # Set default values for optional arguments (numerical etc.)

    model_parameters = model_parameters.copy()
    numerical_options = numerical_options.copy()
    optim_options = optim_options.copy()

    def set_default(dico, key, value):
        if dico.get(key) is None:
            dico[key] = value

    set_default(model_parameters, "noise_std", 1)   # (in case of affine registration)

    default_support_scheme = {
                "scheme": "grid",   # "dense", "grid" or "decim"
                "rho": 1.0}         # remaining parameters to diffPSR.set_support_scheme()
    set_default(numerical_options, "support_LDDMM", default_support_scheme)
    set_default(numerical_options, "computversion", "keops")
    set_default(numerical_options, "integration_scheme_LDDMM", "Euler")    # Euler (faster) vs Ralston (more precise)
    set_default(numerical_options, "integration_nt_LDDMM", 10)             # number of time steps

    set_default(optim_options, "max_iterations", 25)            # Number of global loop iterations (fixed, for the moment !)
    set_default(optim_options, "convergence_tolerance", 1e-3)   # Tolerance parameter (TODO differentiate between global loop and single optimizations ?)
    set_default(optim_options, "nmax_per_iter", 10)             # Max number of iterations for Reg_opt, inside *each* iteration.

    #########################

    ### Read input point sets and various dimensions.

    D, DB = xA.shape[1], xB.shape[1]
    assert D == DB, "point sets xA and xB should have same vector dimension (dim 1)"

    bounds = get_bounds(xA, xB, relmargin=0.1)

    ### Create the DiffPSR_std object that will perform the registration

    DataKernel = GaussKernel(model_parameters["sigma_data"], D=D)

    if is_diff:

        LMi = LDDMMModel(sigma=model_parameters["sigma_LDDMM"],         # sigma of the Gaussian kernel
                         D=D,                       # dimension of space
                         lambd=2.0,                 # Always 2 to match the "standard" definition in deformetrica.
                         version="classic",
                         computversion=numerical_options["computversion"],      # "torch" or "keops"
                         scheme=numerical_options["integration_scheme_LDDMM"],  # "Euler" (faster) or "Ralston" (more precise)
                         nt=numerical_options["integration_nt_LDDMM"])

        PSR = DiffPSR_std(xB, xA,
                          model_parameters["noise_std"], LMi, DataKernel, template_weights=False)

        supp_scheme = numerical_options["support_LDDMM"]["scheme"]
        if supp_scheme != "dense":
            PSR.set_support_scheme(**numerical_options["support_LDDMM"])

        # for storing results
        evol = {"a0": [],        # evol["a0"][it][k] = current a0 tensor(Nk[k],2) at iteration it
                "y0": []}        # evol["y0"][it] = current template point set PSR.y0 at iteration it

    else:

        # Affine registration model
        AffMi = AffineModel(D=D, version=model_parameters["type"],
                            withlogdet=False,
                            with_t=True)
        PSR = AffinePSR_std(xB, xA,
                            model_parameters["noise_std"], AffMi, DataKernel, template_weights=False)

        # for storing results
        evol = {"M": [],            # evol["M"][it][k] = current registration matrix for frame k at iteration it
                "t": [],            # evol["t"][it][k] = current translation vector for frame k at iteration it
                "y0": []}        # evol["y0"][it] = current template point set PSR.y0 at iteration it

    ########################

    if plotstuff:
        plt.figure()    # Figure 1 : basic point sets
        plot_state(PSR, bounds, plot_gridlines=False)
        # if savefigs:      # TODO update if required
        #     savefigs_path = os.path.dirname(os.path.realpath(savefigs_name))
        #     os.makedirs(savefigs_path, exist_ok=True)
        #     plt.savefig(f"{savefigs_name}_data.{savefigs_format}", format=savefigs_format, bbox_inches='tight')

        plt.figure()    # Figure 2 : algorithm evolution

    #########################
    ### And optimize !

    tol = optim_options["convergence_tolerance"]

    last_E = None                                      # Previous value of energy

    for it in range(optim_options["max_iterations"]):
        if printstuff:
            print("ITERATION NUMBER ", it)

        evol["y0"].append(copy.deepcopy(PSR.y0))
        if is_diff:
            evol["a0"].append([a0k.clone().detach().cpu() for a0k in PSR.a0])
        else:
            evol["M"].append([Mk.clone().detach().cpu() for Mk in PSR.M])
            evol["t"].append([tk.clone().detach().cpu() for tk in PSR.t])

        if plotstuff:
            plot_state(PSR, bounds)

        ### One optimization step (to view evolution of optimizer)
        PSR.Reg_opt(nmax=optim_options["nmax_per_iter"], tol=tol)

        if it > 1 and abs(PSR.E-last_E) < tol * abs(last_E):
            if printstuff:
                print("Difference in energy is below tolerance threshold : optimization is over.")
            break

        last_E = PSR.E

    # DONE !
    if printstuff and it+1 == optim_options["max_iterations"]:
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

    # Model parameters
    model_parameters = {"type": "similarity",
                        "sigma_data": 0.1,
                        "noise_std": 0.2,
                        "sigma_LDDMM": 0.2
                        }

    # Numerical parameters : use default for the moment

    # Optimization options
    optim_options = {'max_iterations': 15,              # Number of global loop iterations (fixed, for the moment)
                     'convergence_tolerance': 1e-7,     # for each optimization in the global loop (for the moment)
                     'nmax_per_iter': 2
                     }

    PSR, evol = \
        standard_two_set(xA, xB,
                    model_parameters,
                    optim_options=optim_options)

    print("Final losses :")
    print(f"    regularity: {sum(PSR.regloss)}")
    print(f"    attachment: {PSR.dataloss.sum().item()}")
    print(f"    overall loss: {PSR.E}")

    input()