'''
TODO : Make a class implementing the affine model of registration, including logdet term
'''

import os, time, math
import copy
import warnings

import numpy as np

from matplotlib import pyplot as plt

import torch

from scipy.linalg import expm,logm    # matrix exponential and logarithm (to compute affine shooting)

# Torch tensor specifications (GPU vs CPU) : defined in diffICP.kernel
from diffICP.kernel import torchspec

#####################################################################################################################
# Encapsulate all the "affine registration logic" in a class
#####################################################################################################################

# T(X) = X * M' + t'      with
# X(N,d): input data points
# t(d,1): translation vector
# M(d,d): linear deformation matrix

class AffineModel:

    #############################################################################################
    ### Constructor

    # version = "rigid" (M=rotation), "similarity" (M=rotation+scaling), "affine" (general affine), "translation" (M=Id)
    # with_t = True/False. Also optimize a translation term (True or False)
    # withlogdet = True/False. Add to the registration energy the logdet of the matrix ("backward" model of registration)

    # TODO !

    def __init__(self, D, version="rigid", withlogdet=True, with_t=True, nt=10):

        self.D = D
        self.version = version
        self.withlogdet = withlogdet
        self.with_t = with_t
        self.nt = nt

    ##################################################################################################
    # Shooting and optimization


    ### Simulate a "shooting" trajectory for the affine transformation of parameters (M,t), on sample points X (N*D torch tensor).
    #
    # Let p be the invariant point of T(x)=Mx+t : T(x) = p + M(x-p) ==> p = (Id-M)^{-1} * t
    # Then, at time u, we define : shoot(u,x) = p + M(u)*(x-p)  with  M(u) := exp(u*log(M))
    #
    # For compatibility with the LDDMM case, trajectories are returned in a "list containing a tuple", that is
    # x = list of size nt, and x[u] = ( X_u, ) with X_u the image of point set X at time u

    def Shoot(self, M, t, X):

        # to be sure
        M,t,X = M.detach(), t.detach(), X.detach()

        Ts = np.linspace(0, 1, self.nt)
        if M.equal(torch.eye(self.D)):
            # only special case that I handle explicitly (for other non-invertibility cases, pray that I'm lucky :))
            return [ ( X + u * t[None, :], ) for u in Ts ]                                  # (tuple format for compatibility with LDDMM)
        else:
            # Normal, invertible situation
            Pk = torch.inverse(torch.eye(self.D) - M) @ t
            logM = logm(M, disp=False)[0].real       # ensure real to remove warning
            return [( Pk[None,:] + (X-Pk[None,:]) @ torch.tensor(expm(u * logM).T,**torchspec) ,) for u in Ts ]     # (tuple format for compatibility with LDDMM)


    ### Regularisation energy associated to a given affine transformation (M,t) on point set X
    # In the case of affine registration, this is only the "logdet term", if present.

    def regloss(self, M, w):

        if self.withlogdet:
            return - w.sum() * torch.logdet(M)
        else:
            return 0

    ### Optimization of full registration energy (quadratic datacost + logdet regularisation if present)
    #   for given X (data points), Y (GMM quadratic targets), data weights z_n, and logdet weights w_n :
    #   E(M,t) = \sum_n z_n |M*x_n+t - y_n|^2 - \sum_n w_n log |M|
    #
    # return the optimal M (given required constraints) and t (if with_t=True)
    # TODO switch to KeOps if useful (faster?). For the moment, plain pytorch
    # TODO CHECK
    # TODO implement missing version : affine with logdet

    def Optimize(self, X, Y, z, w=None):

        if w is None:
            w = torch.ones(X.shape[0]).to(**torchspec)

        # Just to be sure
        X, Y, z, w = X.detach(), Y.detach(), z.detach(), w.detach()

        if self.with_t:
            # Center the points
            Xm = (X * z[:,None]).sum(dim=0) / z.sum()
            Ym = (Y * z[:,None]).sum(dim=0) / z.sum()
            Xc, Yc = X - Xm, Y - Ym
        else:
            Xc, Yc = X, Y

        # E = Tr(A*M'*M) - 2* Tr(B'*M) - c* log|M| + C
        # A = Xc.t() @ (z[:,None] * Xc)     # A_ij = \sum_n z_n x_n[i] x_n[j] (not necessary to compute in general)
        B = Yc.t() @ (z[:,None] * Xc)
        c = w.sum()

        if self.version == 'rigid' or self.version == 'similarity':
            # Constraint M'*M=Id (rigid) or lam^2*Id (similarity) ---> SVD solution (see my notes)
            # U, _, V = torch.svd(B, some=False)             # Pytorch 1.7 version
            # Vh = V.t()                                     # transpose (CAUTION when changing Pytorch version!)
            U, _, Vh = torch.linalg.svd(B)                  # Pytorch 2.0 version
            print(f"SVD check : {torch.norm(B - U@torch.diag(_)@Vh)}")
            D = torch.eye(self.D)
            D[-1,-1] = torch.linalg.det(U) * torch.linalg.det(Vh)        # ensure that det(R)=1 (rotation)
            R = U @ D @ Vh

        if self.version == 'rigid':
            M = R

        elif self.version == 'similarity':
            # also optimize scaling factor
            trA = ((Xc**2).sum(-1) * z).sum()
            trBR = (B * R).sum()
            if self.withlogdet:
                lam = (trBR + torch.sqrt(trBR**2 + 2*c*self.D*trA)) / (2*trA)
            else:
                lam = trBR / trA
            M = lam * R

        elif self.version == 'affine':
            if not self.withlogdet:
                # M = B / (X'*X) : classic (unconstrained) least square
                A = Xc.t() @ (z[:,None] * Xc)
                # M = B @ torch.inverse(A)         # Pytorch 1.7 (=old) version
                M = torch.linalg.solve(A,B,left=False)         # Pytorch 2.0 (=new) version (not tested)
            else:
                raise NotImplementedError("General affine registration with term '-c*logdet(M)' not implemented.\n"
                                          "Should find M to minimize E(M) = Tr(A*M*M') - 2*Tr(B'*M) - c*logdet(M)")

        else:   # version = 'translation'
            M = torch.eye(self.D, **torchspec)

        if self.with_t:
            t = Ym - M @ Xm
        else:
            t = 0

        # Also compute and return dataloss (quadratic) and regloss (logdet if present)
        TX = X @ M.t() + t[None,:]
        datal = (((Y-TX)**2).sum(-1) * z).sum()
        regl = self.regloss(M,w)
        return M, t, TX, datal.item(), regl.item()


##############################################################
### Test

if True:
    print("YOOO")
    AM = AffineModel(3, version="rigid", withlogdet=True, with_t=True, nt=10)
    print("YOOO")
    X = torch.randn(100,3).to(**torchspec)
    Y = torch.randn(100,3).to(**torchspec)
    z = torch.ones(100).to(**torchspec)
    M,t,TX,dl,rl = AM.Optimize(X,Y,z)
    print(TX, M, t, dl, rl)
#    print(AM.Shoot(M, t, X))