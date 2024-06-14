'''
Implement ad hoc "calibration" procedures for the tradeoff parameter of the algorithms
    - standard algorithm : parameter noise_std
    - diffICP algorithm : parameter lambda_LDDMM
'''

import math
import torch

from diffICP.core.GMM import GaussianMixtureUnif
from diffICP.core.LDDMM import LDDMMModel
from diffICP.core.affine import AffineModel
from diffICP.core.PSR_standard import data_distance
from diffICP.tools.point_sets import intrinsic_scale

# "import xxxx" avoids circular imports
import diffICP.api.ICP_two_set as ICP_two_set
import diffICP.api.standard_two_set as standard_two_set


######################################################################################
# Calibration for the diffICP algorithm
######################################################################################

def calibrate_lambda_LDDMM(x: torch.Tensor, x2: torch.Tensor, sigma_LDDMM):
    '''
    Calibrate lambda_LDDMM for diffICP registration of point set x on point set x2
    :param x:
    :param x2:
    :param sigma_LDDMM:
    :return: predicted lambda_LDDMM value
    '''

    ### AFFINE REGISTRATION of x on x2

    registration_parameters = {"type": "general_affine"}
    GMM_parameters = {"sigma": None,
                      "optimize_sigma": True,
                      "outlier_weight": None}
    optim_options = {'max_iterations': 30,              # (needs not be very precise)
                     'convergence_tolerance': 1e-4,     # (needs not be very precise)
                     'max_repeat_GMM': 25}
    # Register !
    PSR, _ = ICP_two_set.ICP_two_set(x, x2, GMM_parameters, registration_parameters, optim_options=optim_options, plotstuff=False, printstuff=False)
    y = PSR.y[0,0]                  # 'quadratic targets' of point set x in point set x2
    sigref = PSR.GMMi[0].sigma      # sigma parameter (typical distance between each x_n and its quadratic target y_n)
    # sigref = intrinsic_scale(y)     # TODO rather use sigma = intrinsic_scale(y) ? Not sure it's a good idea...
    Lref = (((PSR.x1[0,0]-y) **2).sum() / (2 * sigref**2) ).detach().item()   # 'reference' value for quadloss when registering point set x to point set x2

    ### LAUNCH AD HOC LDDMM OPTIMIZATION :
    #       min_{a0}  K * exp( quadloss(a0)/Lref ) + 0.5 * ||a0||_{rkhs}^2
    # which represents a relaxed version of the following constrained optimization :
    #       min_{a0}  ||a0||_{rkhs}^2    subject to quadloss(a0) <= Lref
    # For constant K, use a "reference" value of ||a0||_{rkhs}^2, before optimization

    # TODO: this is very unstable, as exp(quadloss(a0)/Lref) is very prone to numeric overflow
    # TODO: must find another procedure, more stable, to realize the "relaxed constrained optimization" required

    LM = LDDMMModel(sigma=sigma_LDDMM, D=x.shape[1], lambd=1, version="classic",
                    computversion="keops", scheme="Ralston")

    # (default) dense scheme : support_points q = data_points x. TODO: allow to define a support scheme if dense version is too slow ?

    a0 = LM.v2p(x, y - x, rcond=1e-2)   # initialize at 'target' speeds
    H0_ref = LM.Hamiltonian(x,a0)       # reference value for the regularization term

    def expLossFunc(x):
        L = ((x - y) ** 2).sum() / (2 * sigref**2)
        return H0_ref * (L/Lref).exp()

    a0, _, _, _, _, _ = LM.Optimize(expLossFunc, x, a0, tol=1e-3, nmax=20)
    deformation = LM.Hamiltonian(x,a0)

    # 'Optimal' lambda_LDDMM should lead to an approximate 1-1 balance of dataloss and regularity in the atlas building, i.e.
    #       typical_quadloss ~= lambda_LDDMM * typical_LDDMM_deformation
    # Thus, we predict:

    lambda_LDDMM = Lref / deformation
    return lambda_LDDMM


######################################################################################
# Calibration for the standard algorithm
######################################################################################

def calibrate_noise_std(x: torch.Tensor, x2: torch.Tensor, sigma_LDDMM):
    '''
    Calibrate noise_std for "standard' diffeomorphic registration of point set x on point set x2
    :param x: "template" point set that will be deformed
    :param x2: "data" point set that will be fixed
    :param sigma_LDDMM:
    :return: predicted noise_std value
    '''

    ### AFFINE REGISTRATION of x on x2

    model_parameters = {"type": "general_affine",
                        "sigma_data": intrinsic_scale(x2)}  # only logical choice :)
    optim_options = {'max_iterations': 30,              # (needs not be very precise)
                     'convergence_tolerance': 1e-4,     # (needs not be very precise)
                     'max_repeat_GMM': 25}
    # Register !
    PSR, _ = standard_two_set.standard_two_set(x, x2, model_parameters, optim_options=optim_options, plotstuff=False, printstuff=False)
    Lref = PSR.E    # (consits only of dataloss in this "affine" case)

    ### LAUNCH AD HOC LDDMM OPTIMIZATION :
    #       min_{a0}  K * exp( dataloss(a0)/Lref ) + 0.5 * ||a0||_{rkhs}^2
    # which represents a smoothed version of the following constrained optimization :
    #       min_{a0}  ||a0||_{rkhs}^2    subject to dataloss(a0) <= Lref
    # For constant K, use a "reference" value of ||a0||_{rkhs}^2, before optimization

    LM = LDDMMModel(sigma=sigma_LDDMM, D=x.shape[1], lambd=1, version="classic",
                     computversion="keops", scheme="Euler")

    # (default) dense scheme : support_points q = data_points x. TODO: allow to define a support scheme if dense version is too slow ?

    Tx = PSR.Registration().apply(x)
    a0 = LM.v2p(x, Tx - x, rcond=1e-2)   # initialize at 'target' speeds
    H0_ref = LM.Hamiltonian(x,a0)       # reference value for the regularization term

    def expLossFunc(q):
        L = data_distance(PSR.DataKernel, q, x2)
        return H0_ref * (L/Lref).exp()

    a0, _, _, _, _, _ = LM.Optimize(expLossFunc, x, a0, tol=1e-3, nmax=20)
    regloss = LM.Hamiltonian(x,a0)

    # 'Optimal' noise_std should lead to an approximate 1-1 balance of dataloss and regularity in the atlas building, i.e.
    #       typical_quadloss / noise_std**2 ~= typical_LDDMM_deformation
    # Thus, we predict:

    noise_std = math.sqrt(Lref / regloss)
    return noise_std