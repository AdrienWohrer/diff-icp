'''-
Test the "standard" atlas construction with LDDMM, as provided by library Deformetrica :
    - Prepare data of the multiple point sets under the suitable format (vtk files)
    - Launch Deformetrica on these data, to build the "standard" atlas of the point sets
    - Retrieve the output of deformetrica in a PSR_std object

Results should be compared to our own method diffICP atlas construction.

Nota : running this file as a script provides an example usage of the function.
Else, simply import function deformetrica_atlas in other scripts to use it there.
'''

import copy
import os
import pickle
import shutil

import numpy as np
import matplotlib.pyplot as plt
import torch
import pyvista as pv

# imports from deformetrica
import importlib.util
if importlib.util.find_spec("deformetrica") is None:
    raise ModuleNotFoundError("The deformetrica module should be installed to use this function.")

from deformetrica import deformetrica as dfca
from deformetrica.deformetrica.core.observations import Landmark

from diffICP.core.LDDMM import LDDMMModel
from diffICP.core.GMM import GaussianMixtureUnif

from diffICP.tools.inout import read_point_sets
from diffICP.tools.kernel import GaussKernel
from diffICP.visualization.visu import my_scatter

from diffICP.core.PSR_standard import DiffPSR_std

################################################################################

def vtk2torch(vtkfile):
    '''
    Load a VTK point cloud saved by Deformetrica (in a .vtk file) into a torch Tensor.

    :param vtkfile: path to the vtk file
    :return: a 2d torch tensor with point set coordinates
    '''

    # We use pyvista to read the vtk file (as meshio does not seem to recognize this vtk format)
    # [:,:2] to switch back to 2d (deformetrica outputs as 3d by default)
    return torch.tensor(pv.PointSet(pv.read(vtkfile)).points[:, :2])


##################################################################################

def deformetrica_atlas(x, initial_template:torch.Tensor, model_parameters,
                       numerical_options={}, estimator_options={}):
    '''
    Launch standard LDDMM atlas building, using deformetrica api.

    :param x: input data points. Several possible formats, e.g., x[k][s] = cloud point from frame k and structure s;
    :param initial_template: initial location of the template points ;

    :param model_parameters: main model parameters :
        model_parameters["sigma_data"]: spatial std. of the RKHS Kernel used to define the data distance, (K(x)=exp(-x^2/2*sigma^2)) ;
        model_parameters["noise_std"]: scaling parameter of the data loss term, so Loss = \sum_s rkhs_distance(s) / noise_std[s]^2 ;
        model_parameters["sigma_LDDMM"]: spatial std of the RKHS Kernel defining LDDMM diffeomorphisms ;

    :param numerical_options: dict with various numerical details of the algorithm :
        numerical_options["computversion"] : "keops" or "torch" ;
        numerical_options["dense_mode"] : True -> control points = template points, False -> control points on a grid ;

    :param estimator_options: numerical options for the optimization procedure (see code) ;
    :return: PSR [main output, registration object after optim], shoot_defo, iter_status [extra info produced by deformetrica]
    '''

    ######################
    # Check mandatory model parameters

    assert {"sigma_data","noise_std","sigma_LDDMM"}.issubset(model_parameters.keys()), \
        "model_parameters should define values of sigma_data, noise_std and sigma_LDDMM"

    ######################
    # Set default values for optional arguments (numerical etc.)

    numerical_options = numerical_options.copy()
    estimator_options = estimator_options.copy()

    def set_default(dico, key, value):
        if dico.get(key) is None:
            dico[key] = value

    default_support_scheme = {
                "scheme": "grid",   # "dense", "grid" or "decim"
                "rho": 1.0}         # remaining parameters to diffPSR.set_support_scheme()

    set_default(numerical_options, "dense_mode", False)
    set_default(numerical_options, "computversion", "keops")

    set_default(estimator_options, 'optimization_method_type', 'GradientAscent')  # 'GradientAscent' or 'ScipyLBFGS' (better but less stable)
    set_default(estimator_options, 'max_iterations', 500)
    set_default(estimator_options, 'convergence_tolerance', 1e-7)
    set_default(estimator_options, 'initial_step_size', 1e-6)

    #########################
    # Temporary input/output storage for deformetrica

    tmpdir = f"saving/last_deformetrica_tmp"
    datadir = f"{tmpdir}/data"
    outdir = f"{tmpdir}/out"
    os.makedirs(datadir, exist_ok=True)

    #########################
    # Read input point sets and various dimensions. Output:
    #   x: point sets, now cast in the format x[k][s] ;
    #   K = number of frames
    #   S = number of structures
    #   D = dimension of space

    x, K, S, D = read_point_sets(x)

    #########################
    # Sources for using deformetrica API :
    # https://colab.research.google.com/drive/1ZYArpukrdp_SsRh-cW6PXJNaLEt1Tafr
    # https://medium.com/miccai-educational-initiative/a-beginners-guide-to-shape-analysis-using-deformetrica-fa9e346357b7#a80c

    ### PREPARE DATASET IN SUITABLE FORM
    # TODO: generalize when there are multiple structures s

    for k in range(K):
        lmark = Landmark(x[k][0].numpy())
        lmark.write(datadir, f"point_set_{k}.vtk")  # write to vtk format

    dataset_specifications = {
        'dataset_filenames': [[{'pointset': f"{datadir}/point_set_{k}.vtk"}] for k in range(K)],
        'subject_ids': [str(i) for i in range(K)]
    }

    ### PREPARE TEMPLATE SPECIFICATIONS IN SUITABLE FORM

    # TODO: maybe allow here some different 'default' options for template initialization ?
    Landmark(initial_template.numpy()).write(datadir, f"initial_template.vtk")      # write to vtk format

    # template_specifications required by deformetrica
    ### TODO : Generalize to the case when there are multiple structures s !
    template_specifications = {
        'pointset' : {'deformable_object_type': 'pointcloud',
                     'kernel_type': numerical_options["computversion"],
                     'kernel_width': model_parameters["sigma_data"] * np.sqrt(2),
                     'noise_std': model_parameters["noise_std"],
                     'filename': f"{datadir}/initial_template.vtk"}
    }

    ### Add callback to estimator options
    # (Nota : status_dict does not hold enough information to be used for online plotting as in my own atlas methods)

    iter_status = []
    def estimator_callback(status_dict):
        iter_status.append(status_dict)
        return True
    estimator_options["callback"] = estimator_callback

    ### PREPARE MODEL_OPTIONS IN SUITABLE FORM

    model_options = {'dimension': D,
                     'deformation_kernel_type': numerical_options["computversion"],  # 'torch' or 'keops'
                     'deformation_kernel_width': model_parameters["sigma_LDDMM"] * np.sqrt(2),   # sigma of LDDMM kernel
                     'dense_mode': numerical_options["dense_mode"],     # True -> one control point per template point. False -> control points on a regular grid
                     'number_of_timepoints': 11,                        # Not sure this is even taken into account ???
                     'dtype': 'float32',
                     'gpu_mode': dfca.GpuMode.NONE}

    ######################################################
    ### Done with specifications ! Launch analysis with Deformetrica API

    deformetrica = dfca.Deformetrica(output_dir=outdir, verbosity='INFO')

    # And LAUNCH !
    deformetrica.estimate_deterministic_atlas(
        template_specifications,
        dataset_specifications,
        estimator_options=estimator_options,
        model_options=model_options)

    ######################################################
    ### Recover simulation results : read in vtk files produced by deformetrica.
    # It is convenient to store all this information in my handmade `diffPSR_std` object

    # Template after optimization (can then be recovered as PSR.get_template())
    final_template = vtk2torch(f'{outdir}/DeterministicAtlas__EstimatedParameters__Template_pointset.vtk')

    # LDDMM diffeomorphisms after optimization : control points q0 and momenta a0[k] for each patient

    q0 = torch.tensor(np.loadtxt(f'{outdir}/DeterministicAtlas__EstimatedParameters__ControlPoints.txt'))
    Nq = q0.shape[0]

    momentafile = f'{outdir}/DeterministicAtlas__EstimatedParameters__Momenta.txt'
    with open(momentafile) as f:
        lines = [line for n,line in enumerate(f) if n>1]
    a0 = torch.tensor(np.loadtxt(lines))
    a0 = [a0[Nq*i:Nq*(i+1)] for i in range(K)]     # to format a0[k] = torch tensor = momenta for frame k

    # Store all information in PSR_std object

    LMi = LDDMMModel(model_parameters["sigma_LDDMM"], D=D, lambd=2.0, version="classic", scheme="Euler")
    DataKernel = GaussKernel(model_parameters["sigma_data"], D)
    PSR = DiffPSR_std(x, final_template, model_parameters["noise_std"], LMi, DataKernel)
    PSR.set_support_scheme("custom", q0=q0)
    PSR.a0 = [ a.to(**PSR.compspec).contiguous() for a in a0 ]
    PSR.update_state()

    # Also recover the shooting of the different point sets by deformetrica. Store this in our handmade `shoot` format.
    # This is only useful for debug : we can check afterwards that our handamade PSR object computes the same trajectories,
    # as it should since it has the same support points and the same initial momenta.

    shoot_defo = [[ (vtk2torch(f"{outdir}/DeterministicAtlas__flow__pointset__subject_{k}__tp_{t}.vtk"),)
               for t in range(model_options["number_of_timepoints"]) ] for k in range(PSR.K) ]

    # Remove temporary directory
    shutil.rmtree(tmpdir)

    return PSR, iter_status, shoot_defo



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
    initial_template = x0[0]

    # Model parameters
    model_parameters = {"sigma_data": 0.1,
                        "noise_std": 0.2,
                        "sigma_LDDMM": 0.2 }

    # Optimization options
    estimator_options = {'optimization_method_type': 'ScipyLBFGS',  # 'GradientAscent' or 'ScipyLBFGS' (better but less stable)
                         'max_iterations': 200,
                         'convergence_tolerance': 1e-6,
                         'initial_step_size': 1e-6}

    PSR, iter_status, shoot_defo = \
        deformetrica_atlas(x0, initial_template, model_parameters,
                           estimator_options=estimator_options)

    import matplotlib
    matplotlib.use('TkAgg')

    def plot_with_template(template):
        plt.figure()
        # utilise un GMM pour la représentation du template
        GMMr = GaussianMixtureUnif(template)
        # GMMr.sigma = template_specifications["pointset"]["kernel_width"]        # choix logique, mais pas optimal pour visualiser lorsque sigma est grand
        GMMr.sigma = 0.025                                                        # ad hoc but better visualization

        my_scatter(template, color='green')
        GMMr.plot(*x0, template)
        #        my_scatter(*y1[0:min(5, PSR.K)], alpha=.6)
        my_scatter(*x0[3:4], alpha=.6)
        for k in range(3,4):
            PSR.plot_trajectories(k, color='red', linewidth=4)
            # CHECK THAT THE RECOMPUTED SHOOTING BY `PSR' CORRESPONDS TO THE SHOOT VARIABLE PRODUCED BY DEFORMETRICA :
            # YES ! The match is absolutely perfect.
            PSR.plot_trajectories(k, shoot=shoot_defo[k], color='black')
        plt.show()
        plt.pause(.1)

    print("Final log-likelihoods in deformetrica:")
    print({k:iter_status[-1][k] for k in ('current_log_likelihood','current_attachment','current_regularity') if k in iter_status[-1]})

    print("Corresponding losses computed from handmade PSR object (should correspond) :")
    print(f"    regularity: {sum(PSR.regloss)})")
    print(f"    attachment: {PSR.dataloss.sum().item()}")
    print(f"    overall loss: {PSR.E}")

    # Figure with final_template
    plot_with_template(PSR.get_template())

    # Figure with initial_template (for comparison)
    plot_with_template(initial_template)

    input()

    ### Old piece of code to view the evolution of likelihood, in the case of deformetrica ; adapt if necessary

    # x = range(len(iteration_status_dictionaries))
    # # plot log-likelihood
    # matplotlib.use('TkAgg')
    # plt.figure()
    # plt.plot(x, [it_data['current_log_likelihood'] for it_data in iteration_status_dictionaries],
    #          label='log_likelihood')
    # # plt.plot(x, [it_data['current_attachment'] for it_data in iteration_status_dictionaries], label='attachment')
    # # plt.plot(x, [it_data['current_regularity'] for it_data in iteration_status_dictionaries], label='regularity')
    # plt.xticks(x)
    # plt.style.use('default')
    # plt.legend()