'''-
Test the "standard" atlas construction with LDDMM (Glaunès et al. 04), with a personal re-implementation.

Results should be compared to our own method for atlas construction (diffICP).

Nota : running this file as a script provides an example usage of the function.
Else, simply import function standard_atlas in other scripts to use it there.
'''

import copy, warnings
import pickle

import matplotlib
import matplotlib.pyplot as plt
import torch

from diffICP.core.LDDMM import LDDMMModel
from diffICP.core.GMM import GaussianMixtureUnif

from diffICP.tools.spec import defspec
from diffICP.tools.in_out import read_point_sets
from diffICP.tools.kernel import GaussKernel
from diffICP.tools.point_sets import intrinsic_scale

from diffICP.visualization.visu import my_scatter

from diffICP.core.PSR_standard import DiffPSR_std
import diffICP.core.calibration as calibration

##################################################################################
### Debug function : plot the current state of PSR model (template location, target points, trajectories etc.)

def plot_state(PSR:DiffPSR_std, sigma=None, only_template=False):
    plt.clf()
    y1 = PSR.y1[:, 0]   # warped templates
    x = PSR.x[:, 0]    # data points
    # utilise un GMM uniquement pour la représentation de y_template
    GMMr = GaussianMixtureUnif(PSR.y0[0])
    if sigma is not None:
        GMMr.sigma = sigma
    else:
        # GMMr.sigma = PSR.DataKernel.sigma               # choix logique, mais pas optimal pour visualiser lorsque sigma est grand
        GMMr.sigma = intrinsic_scale(PSR.y0[0])     # ad hoc but better visualization

    GMMr.plot(*x, *y1)
    #        my_scatter(*y1[0:min(5, PSR.K)], alpha=.6)
    if not only_template:
        my_scatter(*x[0:min(5, PSR.K)], alpha=.6)
        for k in range(min(5, PSR.K)):
            PSR.plot_trajectories(k)
            # PSR.plot_trajectories(k, support=True, linewidth=2, alpha=1)  # only useful in DiffPSR class
    plt.show()
    plt.pause(.1)


##################################################################################
##################################################################################
##################################################################################

def standard_atlas(x, initial_template=0,
                       model_parameters={},
                       numerical_options={}, optim_options={}, callback_function=None, printstuff=True):
    '''
    Launch standard LDDMM atlas building, using personal reimplementation in PSR_standard.py.

    :param x: input data points. Several possible formats, e.g., x[k][s] = cloud point from frame k and structure s

    :param initial_template: initial location of the template points. Can be either :
        a torch 2d tensor : custom location of the template points ;
        an index i : use point set x[i] as initial template ;

    :param model_parameters: dict with main model parameters :
        model_parameters["sigma_data"]: spatial std of the RKHS Kernel used to define the data distance, (K(x)=exp(-x^2/2*sigma^2)).
            If None, automatically use a reference value based on the 'intrinsic scale' of the point sets ;
        model_parameters["noise_std"]: scaling parameter of the data loss term, so Loss = \sum_s rkhs_distance(s) / noise_std[s]^2.
            Set at "auto" for ad hoc calibration (experimental!) ;
        model_parameters["sigma_LDDMM"]: spatial std of the RKHS Kernel defining LDDMM diffeomorphisms ;
        model_parameters["use_template_weights"]: associate inhomogeneous scalar weights to the template points, and optimize them too ;

    :param numerical_options: dict with various numerical details of the algorithm :
        numerical_options["computversion"] : "keops" or "torch" ;
        numerical_options["support_LDDMM"] : dict, chosen LDDMM support scheme (see code below) ;
        etc. (see code below) ;

    :param optim_options: numerical options for the optimization procedure (see code below) ;

    :param callback_function: optional function to execute at every iteration of the optimization loop, e.g. for reporting or plotting.
        Must take the form callback_function(PSR, before_reg=False/True), with PSR the PSR object being currently optimized

    :param printstuff: True/False, whether to print evolution information during optimization ;

    :return: PSR [main output, registration object after optim], evol [evolution of selected quantities over iterations]
    '''

    # TODO : handling of specs (gpu vs cpu) written but not tested. Expect failures when using different specs than defspec.

    ######################
    # Check mandatory model parameters

    assert {"sigma_data","noise_std","sigma_LDDMM"}.issubset(model_parameters.keys()), \
        "model_parameters should at least define values of sigma_data, noise_std and sigma_LDDMM"

    ######################
    # Set default values for optional arguments (numerical etc.)

    model_parameters = model_parameters.copy()
    numerical_options = numerical_options.copy()
    optim_options = optim_options.copy()

    def set_default(dico, key, value):
        if dico.get(key) is None:
            dico[key] = value

    set_default(model_parameters, "use_template_weights", False)

    default_support_scheme = {
                "scheme": "grid",   # "dense", "grid" or "decim"
                "rho": 1.0}         # remaining parameters to diffPSR.set_support_scheme()
    set_default(numerical_options, "support_LDDMM", default_support_scheme)
    set_default(numerical_options, "computversion", "keops")
    set_default(numerical_options, "integration_scheme_LDDMM", "Euler")    # Euler (faster) vs Ralston (more precise)
    set_default(numerical_options, "integration_nt_LDDMM", 10)             # number of time steps
    set_default(numerical_options, "compspec", defspec)                    # 'compspec' = device for computations (gpu vs cpu)
    set_default(numerical_options, "dataspec", defspec)                    # 'dataspec' = device for storage (gpu vs cpu)

    set_default(optim_options, "max_iterations", 25)            # Maximum number of global loop iterations
    set_default(optim_options, "convergence_tolerance", 1e-3)   # Tolerance parameter (TODO differentiate between global loop and single optimizations ?)
    set_default(optim_options, "start_by_template_opt", False)  # can drastically change the resulting convergence !

    compspec = numerical_options["compspec"]
    dataspec = numerical_options["dataspec"]

    #########################

    ### Read input point sets and various dimensions. Output:
    #   x: point sets, now cast in the format x[k][s] ;
    #   K = number of frames
    #   S = number of structures
    #   D = dimension of space

    x, K, S, D = read_point_sets(x)

    # Special code for sigma_data : compute from input point sets if necessary
    if model_parameters["sigma_data"] is None:
        model_parameters["sigma_data"] = sum(intrinsic_scale(x[k][s]) for k in range(K) for s in range(S)) / (K*S)

    ### Create the DiffPSR_std object that will perform the registration

    if type(initial_template) is int:
        initial_template = x[initial_template]

    DataKernel = GaussKernel(model_parameters["sigma_data"], D=D, spec=compspec)

    noise_std = model_parameters["noise_std"]
    sig = model_parameters["sigma_LDDMM"]

    # Special code for automatic calibration of noise_std (EXPERIMENTAL!)
    if noise_std == "auto":
        if printstuff:
            print("--------------------\nAutomatic calibration of noise_std (warning: this is ad hoc!) ...")
        # Calibrate lambda from repeated two_set registrations of one (arbitrary) point set on another
        N_pairs = min(K - 1, 10)
        noise_stds = torch.tensor(
            [calibration.calibrate_noise_std(x[i][0], x[i + 1][0], sig) for i in range(N_pairs)])
        noise_stds = noise_stds[torch.logical_not(noise_stds.isnan())]
        # Harmonic mean seems more appropriate given that noise_std is an "inverse deformation".
        noise_std = 1 / ((1 / noise_stds).mean())
        if printstuff:
            print(f"    noise_std = {noise_std}\n--------------------")

    LMi = LDDMMModel(sigma=sig,                 # sigma of the Gaussian kernel
                     D=D,                       # dimension of space
                     lambd=2.0,                 # Always 2 to match the "standard" definition in deformetrica.
                     version="classic",
                     computversion=numerical_options["computversion"],      # "torch" or "keops"
                     scheme=numerical_options["integration_scheme_LDDMM"],  # "Euler" (faster) or "Ralston" (more precise)
                     spec= compspec,                                        # device and type of data
                     nt=numerical_options["integration_nt_LDDMM"])

    PSR = DiffPSR_std(x, initial_template, noise_std, LMi, DataKernel,
                      compspec=compspec, dataspec=dataspec,
                      template_weights=model_parameters["use_template_weights"])

    supp_scheme = numerical_options["support_LDDMM"]["scheme"]
    if supp_scheme != "dense":
        PSR.set_support_scheme(**numerical_options["support_LDDMM"])

    #########################
    ### And optimize !

    # for storing results
    evol = {"a0": [],        # evol["a0"][it][k] = current a0 tensor(Nk[k],2) at iteration it
            "y0": []}        # evol["y0"][it] = current template point set PSR.y0 at iteration it
    if model_parameters["use_template_weights"]:
        evol["w0"] = []      # evol["w0"][it] = current template weights PSR.w0 at iteration it

    tol = optim_options["convergence_tolerance"]

    last_E = None                                      # Previous value of energy

    for it in range(optim_options["max_iterations"]):   # TODO add break condition on global loop
        print("ITERATION NUMBER ", it)

        evol["y0"].append(copy.deepcopy(PSR.y0))
        evol["a0"].append([a0k.clone().detach().cpu() for a0k in PSR.a0])
        if model_parameters["use_template_weights"]:
            evol["w0"].append(copy.deepcopy(PSR.w0))

        if callback_function is not None:
            callback_function(PSR, True)

        if not (it == 1 and optim_options["start_by_template_opt"]):
            print("Updating diffeomorphisms (individually for each frame k).")
            PSR.Reg_opt(nmax=1)

        if callback_function is not None:
            callback_function(PSR, False)

        print("Updating (common) template.")
        PSR.Template_opt(nmax=1)

        if it > 1 and abs(PSR.E-last_E) < tol * abs(last_E):
            print("Difference in energy is below tolerance threshold : optimization is over.")
            break

        last_E = PSR.E

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
        K = 10
        x0, _, _ = generate_spiral_point_sets(K=K, Nkbounds=(N, N + 40),
                                              sigma_GMM=0.025,
                                              sigma_LDDMM=0.1, lambda_LDDMM=1e2)

    # Template point set. Recommended initialization : use one of the datasets
    initial_template = 0        # use point set 0
    # Alternatively : take Ntemplate random points from all point sets (harder : test efficiency of optimization procedure)
    # Ntemplate = 50
    # allpoints = torch.cat(tuple(x0), dim=0)
    # initial_template = allpoints[torch.randperm(allpoints.shape[0])[:Ntemplate], :]

    model_parameters = {"sigma_data": 0.1,
                        "noise_std": 0.2,
                        "sigma_LDDMM": 0.2,
                        "use_template_weights": False
                        }

    optim_options = {'max_iterations': 15,              # Maximum number of global loop iterations
                     'convergence_tolerance': 1e-3,     # Relative tolerance parameter for convergence
                     'start_by_template_opt': False     # can drastically change the resulting convergence !
                     }

    # numerical parameters : use default for the moment

    def callback_plot(PSR, before_reg):
        plt.figure(1)
        plot_state(PSR)
        if before_reg:
            plt.figure(2)
            plot_state(PSR, only_template=True)

    # Launch
    PSR, evol = \
        standard_atlas(x0, initial_template, model_parameters,
                    optim_options=optim_options, callback_function=callback_plot)

    def plot_with_template(template):
        plt.figure()
        # utilise un GMM pour la représentation du template
        GMMr = GaussianMixtureUnif(template)
        # GMMr.sigma = template_specifications["pointset"]["kernel_width"]        # choix logique, mais pas optimal pour visualiser lorsque sigma est grand
        GMMr.sigma = 0.025                                                        # ad hoc but better visualization

        my_scatter(template, color='green')
#        GMMr.plot(*x0, template)
        #        my_scatter(*y1[0:min(5, PSR.K)], alpha=.6)
        my_scatter(*x0[3:4], alpha=.6)
        for k in range(3,4):
            PSR.plot_trajectories(k)
        plt.show()
        plt.pause(.1)

    print("Final losses :")
    print(f"    regularity: {sum(PSR.regloss)}")
    print(f"    attachment: {PSR.dataloss.sum().item()}")
    print(f"    overall loss: {PSR.E}")

    # Figure with final_template
    plot_with_template(PSR.get_template())

    # Figure with initial_template (for comparison)
    plot_with_template(initial_template)

    # Small test...
    with open("saving/test_standard_atlas.pkl", 'wb') as f:
        pickle.dump(PSR.get_template(), f)

    input()