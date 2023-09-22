'''
"Registration" objects providing a simple interface to perform actual registration of points.
The underlying registration logic is stored elsewhere, in LDDMMModel_logdet.py (for LDDMM registration)
and Affine_logdet.py (for affine registration)
'''

# Standard imports

import os, time, math, copy
import warnings
import numpy as np
from matplotlib import pyplot as plt
import torch

# Imports from diffICP module

from diffICP.LDDMM_logdet import LDDMMModel
from diffICP.Affine_logdet import AffineModel


#################################################
# Common (informal) interface to LDDMM and Affine registrations

class Registration:

    # ---------------------
    def apply(self, X:torch.Tensor):
        '''Compute the registration of an external point set X.
        Y = apply(X)
        with X(N,D) some input points, and Y(N,D) their registered version'''
        pass

    # ---------------------
    def backward(self, Y:torch.Tensor):
        '''Compute the BACKWARD registration of an external point set Y.
        X = backward(Y)
        produces X such that Y = apply(X)
        A shoot variable corresponding to a previous forward registration can be provided, to gain time (minimal!)'''

    # ---------------------
    def shoot(self, X:torch.Tensor, backward=False):
        '''Compute the 'shoot' variable on external point set X. (See LDDMM_logdet.py and Affine_logdet.py)'''
        pass



#################################################
# LDDMM version

class LDDMMRegistration(Registration):

    # ---------------------
    def __init__(self, LMi:LDDMMModel, q0:torch.Tensor, a0:torch.Tensor):
        self.LMi = LMi
        self.q0 = q0
        self.a0 = a0

    # ---------------------
    def shoot(self, X:torch.Tensor, backward=False, previous_forwardshoot=None):
        '''Compute the LDDMM "shooting" on external point set X. Return a "shoot" variable (see LDDMM_logdet.py)
        When backward=True, uses the inverse diffeomorphism. In that case, a shoot variable corresponding to a previous
        forward registration can be provided, if available, to gain a little time (meh...)'''

        if not backward:
            if previous_forwardshoot is not None:
                warnings.warn("variable 'previous_forwardshoot' is useless when backward=False [default]", RuntimeWarning)
            return self.LMi.Shoot(self.q0, self.a0, X)
        else:
            if previous_forwardshoot is None:
                previous_forwardshoot = self.shoot(None)
            q1k, a1k = previous_forwardshoot[-1][0], previous_forwardshoot[-1][1]   # arrival positions and momenta
            return self.LMi.Shoot(q1k, -a1k, X)

    # ---------------------
    def apply(self, X:torch.Tensor):
        '''Compute the registration of an external point set X.
        Y = apply(X)
        with X(N,D) some input points, and Y(N,D) their registered version'''

        return self.shoot(X)[-1][3]

    # ---------------------
    def backward(self, Y:torch.Tensor, previous_forwardshoot=None):
        '''Compute the BACKWARD registration of an external point set Y.
        X = backward(Y)
        produces X such that Y = apply(X)
        If available, a shoot variable corresponding to a previous forward registration can be provided, to gain a little time'''

        backshoot = self.shoot(Y, backward=True, previous_forwardshoot=previous_forwardshoot)
        return backshoot[-1][3]


#################################################
# affine version

class AffineRegistration(Registration):

    # ---------------------
    def __init__(self, AffMi:AffineModel, M:torch.Tensor, t:torch.Tensor):
        self.AffMi = AffMi
        self.M = M
        self.t = t

    # ---------------------
    def shoot(self, X:torch.Tensor):
        '''Compute the Affine "shooting" on external point set X. Return a "shoot" variable (see Affine_logdet.py)'''
        # TODO: backward version not implemented

        return self.AffMi.Shoot(self.M, self.t, X)

    # ---------------------
    def apply(self, X:torch.Tensor):
        '''Compute the registration of an external point set X.
        Y = apply(X)
        with X(N,D) some input points, and Y(N,D) their registered version'''

        return X @ self.M.t() + self.t[None,:]

    # ---------------------
    def backward(self, Y:torch.Tensor):
        '''Compute the BACKWARD registration of an external point set Y.
        X = backward(Y)
        produces X such that Y = apply(X)'''

        return torch.linalg.solve(self.M.t(), Y - self.t[None,:], left=False)


