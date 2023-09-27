'''
Kernel and kernel reductions required by the LDDMM framework (only choice for the moment: Gaussian kernel)
'''
import warnings

# A Wohrer, 2023

# Standard imports
import numpy as np
import torch

import logging
logging.basicConfig(level=logging.INFO)

import importlib.util

# Look for keops and use it if possible
use_keops = importlib.util.find_spec("pykeops") is not None
if use_keops:
    from pykeops.torch import Vi, Vj, Pm
else:
    print("Warning: pykeops not installed. Consider installing it, if you are on linux")

from diffICP.tools.spec import defspec, getspec


############################################################################################################
############################################################################################################


def SVDpow(M, alpha, rcond=None):
    '''
     Helper function : SVD-based (pseudo-)power of a hermitian matrix.
    :param M:  PyTorch hermitian matrix
    :param alpha: sought power, i.e., return M**alpha
    :param rcond: relative cut-off on SVD (warning: big influence on result when alpha<0)
    :return: M**alpha as a Pytorch tensor
    '''
    U, S, Vh = torch.linalg.svd(M)      # torch 2.0
    if rcond is not None:
        keep = S > rcond * S[0]
    else:
        keep = range(len(S))
    return U[:,keep] @ torch.diag(S[keep] ** alpha) @ Vh[keep,:]

# Test
if False:
    L = torch.randn((10, 300), **defspec)
    K = L @ L.t()
    M = SVDpow(K, -0.5, rcond=None)
    print(M @ K @ M)
    exit()

###############################################################################################################
# Base class "GenKernel" (virtual, as K_keops, K_torch and some other functions are not implemented)
###############################################################################################################

class GenKernel:

    ###############
    ### Some "basic" torch and keops functions, used in the reductions below

    # USAGE : GK.K_torch(x,y)	    --> Pytorch tensor of size (M,N) with values K(x_i-y_j)
    def K_torch(self,x,y):
        raise NotImplementedError()

    # USAGE : GK.GradK_torch(x,y)	    --> Pytorch tensor of size (M,N) with values (\nabla K)(x_i-y_j)
    def GradK_torch(self, x, y):
        raise NotImplementedError()

    # USAGE : GK.LapK_torch(x,y)    --> Pytorch tensor of size (M,N) with values -(Delta K)(x_i-y_j)
    def LapK_torch(self, x, y):
        raise NotImplementedError()

    if use_keops:

        # USAGE : GK.K_keops(x_,y_)	    --> LazyTensor symbolic matrix of size (M,N) with values K(x_i-y_j)
        def K_keops(self, x_, y_):
            raise NotImplementedError()

        # Nota: the symbolic computation of Laplacian (based on Keops formula for the kernel) is too slow
        # So the hard-coded formula should be written for each type of Kernel

        # USAGE : GK.LapK_keops(x_,y_)	--> LazyTensor symbolic matrix of size (M,N) with values -Delta K)(x_i-y_j)
        def LapK_keops(self, x_, y_):
            raise NotImplementedError()

    ###############
    ### Constructor

    # (computversion argument leaves us the possibility to use "torch" computations even when keops is available)

    def __init__(self, D, computversion="keops"):

        self.computversion = None
        self.KBase, self.KRed, self.KRedScal, self.GradKRed, self.DDKRed,\
            self.GenDKRed, self.HessKRed, self.LapKRed, self.GradLapKRed, self.GradKRed_rev \
            = None,None,None,None,None,None,None,None,None,None

        ########
        ### Various generic KeOps-based reductions

        if use_keops:

            x, y, b, c = Vi(0,D), Vj(1,D), Vj(2,D), Vi(3,D)      # Symbolic argument passing (faster)
            K = self.K_keops(x, y)

            # USAGE : GK.KBase(x,y)    --> PyTorch tensor of size (M,)    --> X(i) = \sum_j K(x_i-y_j)
            self.KBase_keops = K.sum_reduction(axis=1)

            # USAGE : GK.KRedScal(x,y,d)    --> PyTorch tensor of size (M,)    --> X(i) = \sum_j K(x_i-y_j)d_j
            d = Vj(2, 1)
            self.KRedScal_keops = (K * d).sum_reduction(axis=1)

            # USAGE : GK.KRed(x,y,b)    --> PyTorch tensor of size (M,D)    --> X(i,d) = \sum_j K(x_i-y_j)b_j^d
            self.KRed_keops = (K * b).sum_reduction(axis=1)

            # "Gradient of K" reduction
            # USAGE : GK.GradKRed(x,y)    --> PyTorch tensor of size (M,D)  --> X(i,d) = \sum_j (\partial_d K)(x_i-y_j)
            self.GradKRed_keops = K.grad(x, 1).sum_reduction(axis=1)

            # "Reversed summation" version of the preceding
            # USAGE : GK.GradKRed_rev(x,y,d)    --> PyTorch tensor of size (N,1)  --> Y(i) = \sum_i\sum_d (\partial_d K)(x_i-y_j)d_i^d
            d = Vi(2, D)
            self.GradKRed_rev_keops = (K.grad(x,1)*d).sum(-1).sum_reduction(axis=0)

            # "Diagonal Differential" of Kred
            # USAGE : GK.DDKRed(x,y,b)    --> PyTorch tensor of size (M,D)  --> X(i,d) = \sum_j (\partial_d K)(x_i-y_j)b_j^d
            self.DDKRed_keops = (K.grad(x, 1)*b).sum_reduction(axis=1)

            # Reduction used for gradient of H
            # USAGE : GK.GenDKRed(x,y,b,c)    --> PyTorch tensor of size (M,D)  --> X(i,d) = \sum_j (\partial_d K)(x_i-y_j)(c_i^t b_j)
            self.GenDKRed_keops = (K.grad(x, 1)*(b*c).sum(-1)).sum_reduction(axis=1)

            # Reduction used for gradient of H
            # USAGE : GK.HessKRed(x,y,b,c)    --> PyTorch tensor of size (M,D)  --> X(i,d) = \sum_j (\partial^{(2)}_{de} K)(x_i-y_j)(c_i^e - b_j^e)
            # Nota: dans la commande T.grad(x,v), ce sont T et v qui doivent avoir la même dimension vectorielle
            self.HessKRed_keops = K.grad(x,1).grad(x,c-b).sum_reduction(axis=1)

            # "Laplacian of K" reduction
            # USAGE : GK.LapKRed(x,y)    --> PyTorch tensor of size (M,1)   --> \sum_j (\Delta K)(x_i-y_j)
            self.LapKRed_keops = self.LapK_keops(x,y).sum_reduction(axis=1)

            # "Gradient of Laplacian of K" reduction (like GradKRed, but on (Delta K) instead of K)
            # USAGE : GK.GradLapKRed(x,y)    --> PyTorch tensor of size (M,D)   --> X(i,d) = \sum_j (\partial_d \Delta K)(x_i-y_j)
            self.GradLapKRed_keops = self.LapK_keops(x,y).grad(x,1).sum_reduction(axis=1)

        self.set_computversion(computversion)


    #########################
    ### PyTorch versions of the Reductions

    # USAGE : GK.KBase(x,y)    --> PyTorch tensor of size (M,)    --> X(i) = \sum_j K(x_i-y_j)
    def KBase_torch(self, x, y):
        return torch.sum(self.K_torch(x,y), 1)

    # USAGE : GK.KRedScal(x,y,d)    --> PyTorch tensor of size (M,)    --> X(i) = \sum_j K(x_i-y_j)d_j
    def KRedScal_torch(self, x, y, d):
        return torch.sum(self.K_torch(x, y) * d[None, :], 1)

    # USAGE : GK.KRed(x,y,b)    --> PyTorch tensor of size (M,D)    --> X(i,d) = \sum_j K(x_i-y_j)b_j^d
    def KRed_torch(self, x, y, b):
        return torch.sum(self.K_torch(x,y)[:, :, None] * b[None,:, :], 1)

    # USAGE : GK.GradKRed(x,y)    --> PyTorch tensor of size (M,D)  --> X(i,d) = \sum_j (\partial_d K)(x_i-y_j)
    def GradKRed_torch(self, x, y):
        return torch.sum(self.GradK_torch(x,y), 1)

    # USAGE : GK.GradKRed_rev(x,y,d)    --> PyTorch tensor of size (N,1)  --> Y(i) = \sum_i\sum_d (\partial_d K)(x_i-y_j)d_i^d
    def GradKRed_rev_torch(self, x, y, d):
        return torch.sum( (self.GradK_torch(x,y) * d[:,None,:]).sum(-1), 0)

    # USAGE : GK.DDKRed(x,y,b)    --> PyTorch tensor of size (M,D)  --> X(i,d) = \sum_j (\partial_d K)(x_i-y_j)b_j^d
    def DDKRed_torch(self, x, y, b):
        return torch.sum(self.GradK_torch(x,y) * b[None, :, :], 1)

    # USAGE : GK.GenDKRed(x,y,b,c)    --> PyTorch tensor of size (M,D)  --> X(i,d) = \sum_j (\partial_d K)(x_i-y_j)(c_i^t b_j)
    def GenDKRed_torch(self,x,y,b,c):
        return torch.sum( self.GradK_torch(x,y) * (b[None,:,:]*c[:,None,:]).sum(-1)[:,:,None] ,1)

    # USAGE : GK.LapKRed(x,y)    --> PyTorch tensor of size (M,1)   --> \sum_j (\Delta K)(x_i-y_j)
    def LapKRed_torch(self, x, y):
        return torch.sum(self.LapK_torch(x,y), 1)

    # USAGE : GK.HessKRed(x,y,b,c)    --> PyTorch tensor of size (M,D)  --> X(i,d) = \sum_j (\partial^{(2)}_{de} K)(x_i-y_j)(c_i^e - b_j^e)
    def HessKRed_torch(self,x,y,b,c):
        raise NotImplementedError()     # should be implemented directly depending on the kernel

    # USAGE : GK.GradLapKRed(x,y)    --> PyTorch tensor of size (M,D)   --> X(i,d) = \sum_j (\partial_d \Delta K)(x_i-y_j)
    def GradLapKRed_torch(self, x, y):
        raise NotImplementedError()     # should be implemented directly depending on the kernel


    #########
    ### Some methods to (pseudo-) solve the linear system based on K
    #
    # USAGE : GK.KxxxSolve(x,v) --> PyTorch tensor of size (M,D)
    #       --> (b_j) such that v_i = \sum_j K(x_i-x_j)b_j    (or best approximation in some sense)

    # Inverse problem is ill-conditioned in general, so we test two standard methods :
    # (1) lstsq/pseudo-inverse based on SVD, or (2) add a small ridge term: K + alpha*Id

    def KpinvSolve(self, x, v, rcond=None):
        K_xx = self.K_torch(x,x)
        return torch.from_numpy(
            # Use Numpy's lstsq. Updated version sending the tensor back to the same device as x and v (not tested)
            np.linalg.lstsq(K_xx.numpy(force=True), v.numpy(force=True), rcond=rcond)[0]
        ).to(**getspec(x,v))

    def KridgeSolve_torch(self, x, v, alpha=1e-4):
        # use PyTorch instead (can also be veeery long, so...)
        K_xx = self.K_torch(x,x)
        return torch.linalg.solve(K_xx + alpha* torch.eye(K_xx.shape[0]), v)    # Newer torch version. (TODO Not checked on Gpu)

    if use_keops:
        def KridgeSolve_keops(self, x, v, alpha=1e-4):
            # KeOps one-liner, but problematic (very long when alpha is small / N is big). (TODO Not checked on Gpu)
            return self.K_keops(Vi(x), Vj(x)).solve(Vi(v), alpha=alpha)

    ###############
    ### Set computation version ('keops' or 'pytorch')

    def set_computversion(self, version):

        if version == "keops" and not use_keops:
            warnings.warn(
                "Asked for keops kernel, but keops is not available on this machine. Switching to torch kernel.")
            version = "torch"
        # Aliases for the reductions (keops or torch version) :
        if version == 'keops':
            # KeOps versions : work even for large datasets
            self.KBase, self.KRed, self.KRedScal, self.GradKRed, self.DDKRed, self.GenDKRed, self.HessKRed, self.LapKRed, self.GradLapKRed, self.GradKRed_rev \
                = self.KBase_keops, self.KRed_keops, self.KRedScal_keops, self.GradKRed_keops, self.DDKRed_keops, self.GenDKRed_keops, self.HessKRed_keops, \
                self.LapKRed_keops, self.GradLapKRed_keops, self.GradKRed_rev_keops
        elif version == "torch":
            # PyTorch versions : faster on CPU + small datasets ; crash on large datasets
            self.KBase, self.KRed, self.KRedScal, self.GradKRed, self.DDKRed, self.GenDKRed, self.HessKRed, self.LapKRed, self.GradLapKRed, self.GradKRed_rev \
                = self.KBase_torch, self.KRed_torch, self.KRedScal_torch, self.GradKRed_torch, self.DDKRed_torch, self.GenDKRed_torch, \
                self.HessKRed_torch, self.LapKRed_torch, self.GradLapKRed_torch, self.GradKRed_rev_torch
        else:
            raise ValueError(f"unkown computversion : {version}. Choices are 'keops' or 'torch'")
        self.computversion = version


###############################################################################################################
# Gaussian kernel (only kernel implemented so far)
###############################################################################################################

#   K(z) = \exp(-\|z\|^2 / 2sig^2)
#   (\nabla K)(z) = -z.K(z)/sig^2
#   (Hess K)(z) = (zz'/sig^4 - I/sig^2).K(z)
#   (\delta K)(z) = (|z|^2/sig^4 - d/sig^2).K(z)
#   (\nabla\delta K)(z) = -(z/sig^2).(\delta K)(z) + 2z/sig^4 .K(z) = (|z|^2/sig^6 - (d+2)/sig^4) * (-z.K(z))

class GaussKernel(GenKernel):

    #######################
    ### Actual kernel (and corresponding gradient + Laplacian kernel) formulas : torch versions

    def K_torch(self, x, y):
        return (-(x[:, None, :] - y[None, :, :]) ** 2 / (2 * self.sigma ** 2)).sum(-1).exp()

    def GradK_torch(self, x, y):
        return self.K_torch(x, y)[:,:,None] * (y[None,:,:]-x[:,None,:]) / self.sigma**2

    def LapK_torch(self, x, y):
        D2 = torch.sum((x[:,None,:] - y[None,:,:]) ** 2, -1)
        return torch.exp(-D2 / (2 * self.sigma ** 2)) * (D2 / self.sigma ** 4 - self.D / self.sigma ** 2)

    #######################
    ### Actual kernel (and corresponding Laplacian kernel) formulas : KeOps versions (symbolic lazytensor)

    if use_keops:

        def K_keops(self, x_, y_):
            return (-(x_.sqdist(y_)) / (2 * self.sigma_k ** 2)).exp()

        def LapK_keops(self, x_, y_):
            return self.K_keops(x_,y_) * ( x_.sqdist(y_) / self.sigma_k ** 4 - self.D_k / self.sigma_k ** 2 )

    ### Some PyTorch reductions that remained to implement directly

    # USAGE : GK.HessKRed(x,y,b,c)    --> PyTorch tensor of size (M,D)  --> X(i,d) = \sum_j (\partial^{(2)}_{de} K)(x_i-y_j)(c_i^e - b_j^e)
    #               X(i,d) = \sum_j ( [(xi-yj)^T(ci-bj)](xi-yj)^d - sig**2.(ci-bj)^d ) K(xi-yj) /sig**4
    def HessKRed_torch(self,x,y,b,c):
        yo = ( (x[:,None,:]-y[None,:,:])*(c[:,None,:]-b[None,:,:]) ).sum(-1)[:,:,None] * (x[:,None,:]-y[None,:,:])
        return torch.sum( ( yo/self.sigma**4 - (c[:,None,:]-b[None,:,:])/self.sigma**2 )*self.K_torch(x,y)[:,:,None], 1)

    # USAGE : GK.GradLapKRed(x,y)    --> PyTorch tensor of size (M,D)   --> X(i,d) = \sum_j (\partial_d \Delta K)(x_i-y_j)
    def GradLapKRed_torch(self, x, y):
        D2 = torch.sum((x[:,None,:] - y[None,:,:]) ** 2, -1)[:,:,None]
        return torch.sum( torch.exp(-D2 /(2*self.sigma**2)) * (y[None,:,:]-x[:,None,:])
                          * (D2 / self.sigma ** 6 - (self.D+2) / self.sigma ** 4)  , 1)

    ###############
    ### Constructor

    # (computversion argument leaves us the possibility to use "torch" computations even when keops is available)

    def __init__(self, sigma, D, computversion="keops", spec=defspec):

        self.sigma = sigma
        self.D = D
        self.spec = spec

        # A current limitation of KeOps SYMBOLIC lazytensors is that parameters (e.g., for the Gaussian kernel,
        # dimension D and scale parameter sigma) are treated as CPU objects unless explicitly precised otherwise,
        # leading to a possible 'CPU/GPU clash'. See question posted here: https://github.com/getkeops/keops/issues/306
        # Thus, we make torch copies of these parameters with imposed spec (and this is the only role of attribute self.spec):
        if use_keops:
            self.sigma_k = Pm(torch.tensor(sigma, **spec))
            self.D_k = Pm(torch.tensor(D, **spec))

        super().__init__(D, computversion)

        # Normally unnecessary : hard-coded formula for gradient
        # self.GradKRed_keops = (-K * (x - y) / sigma ** 2).sum_reduction(axis=1)


    ###############
    ### Helper method : check which points X are "covered" by the kernels centered in Y, i.e., at a distance less than
    # Rthreshold * sigma to one of the points in Y.
    # TODO if necessary: switch to a more general definition applicable to any kernel (not only Gaussian)

    def check_coverage(self, X, Y, Rthreshold):
        if use_keops:
            return Vi(X).sqdist(Vj(Y)).min(axis=1) > (Rthreshold * self.sigma) ** 2
        else:
            return ((X[:,None,:]-Y[None,:,:])**2).sum(-1).min(dim=1) > (Rthreshold * self.sigma) ** 2
            # TODO not tested
    
    ###############
    # Hack to ensure a correct value of spec when Unpickling. See diffICP.spec.CPU_Unpickler and
    # https://docs.python.org/3/library/pickle.html#handling-stateful-objects
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.spec = defspec


######################################################################################
#
# Testing
#
######################################################################################


if __name__ == '__main__':
    # Running as a script

    ### Test reductions :
    if True:

        M, N, D, sig = 100, 1000, 2, 2.0
        xt = torch.randn(M, D).to(**defspec)
        yt = torch.randn(N, D).to(**defspec)
        bt = torch.randn(N, D).to(**defspec)
        vt = torch.randn(M, D).to(**defspec)

        GK = GaussKernel(sig, D, spec=defspec)

        ### Test all reductions (KeOps vs Pytorch versions)

        print(GK.KRed_torch(xt, yt, bt)[:5])  # version PyTorch
        print(GK.KRed_keops(xt, yt, bt)[:5])  # version KeOps

        print(GK.GradKRed_keops(xt, yt)[:5])  # version KeOps
        print(GK.GradKRed_torch(xt, yt)[:5])  # version PyTorch

        print(GK.LapKRed_keops(xt, yt)[:5])  # version KeOps
        print(GK.LapKRed_torch(xt, yt)[:5])  # version PyTorch

        print(GK.DDKRed_keops(xt, yt, bt)[:5])  # version KeOps
        print(GK.DDKRed_torch(xt, yt, bt)[:5])  # version PyTorch

        print(GK.GenDKRed_keops(xt, yt, bt, vt)[:5])  # version KeOps
        print(GK.GenDKRed_torch(xt, yt, bt, vt)[:5])  # version PyTorch

        print(GK.HessKRed_keops(xt, yt, bt, vt)[:5])  # version KeOps
        print(GK.HessKRed_torch(xt, yt, bt, vt)[:5])  # version PyTorch

        print(GK.GradLapKRed_keops(xt, yt)[:5])  # version KeOps
        print(GK.GradLapKRed_torch(xt, yt)[:5])  # version PyTorch

        ### Test "reversed" gradient sum reduction

        print((vt*GK.GradKRed_keops(xt, yt)).sum())  # version KeOps
        print(GK.GradKRed_rev_keops(xt, yt, vt).sum())  # version KeOps (reversed)
        print(GK.GradKRed_rev_torch(xt, yt, vt).sum())  # version pytorch (reversed)

        ### Some more reductions

        for version in ["torch","keops"]:
            GK.set_computversion(version)
            print(GK.KBase(xt,yt)[:5])

        for version in ["torch", "keops"]:
            GK.set_computversion(version)
            print(GK.KRedScal(xt, yt, bt[:,0])[:5])

        exit()

        ### Test pseudo-inverses

        yo = GK.KpinvSolve(xt, vt, rcond=1e-6)
        print(yo)

        vback = GK.KRed(xt, xt, yo)
        print(vt)
        print(vback)  # different than vt, because matrix K is ill-conditioned

    # OK!

    ### Plot some vector fields (for personal slide show)
    if False:

        import matplotlib.pyplot as plt
        plt.ion()
        savefigs = False
        savefigs_name = 'example_vector_field'
        format = 'png'
        bounds = (-0.5,1.5,-0.5,1.5)

        from matplotlib.ticker import FormatStrFormatter

        GK = GaussKernel(0.3, 2)
        N = 3                  # number of support points

        fig = plt.figure()
        xvals = np.linspace(*bounds[:2], 20)
        yvals = np.linspace(*bounds[2:], 20)
        intersec = np.stack(np.meshgrid(xvals, yvals), axis=2)                               # grid intersection (shape (Nx,Ny,2))
        intersec = torch.tensor(intersec.reshape((-1, 2), order='F'), **defspec).contiguous()  # convert to torch tensor (shape (Nx*Ny,2))

        colors = ['red','green']
        for i in range(2):
            q = torch.rand((N,2))           # random support points
            p = torch.randn((N,2))          # random momenta
            vf = GK.KRed(intersec,q,p)      # vector field at intersections
            print(f"v_{i}, vector norm : {(p*GK.KRed(q,q,p)).sum()}")

            plt.quiver(intersec[:,0], intersec[:,1], vf[:,0], vf[:,1], scale=20, color=colors[i], width=0.005)
            plt.xlim(*bounds[:2])
            plt.ylim(*bounds[2:])
            plt.xticks(np.arange(-10, 10, 0.5))
            plt.yticks(np.arange(-10, 10, 0.5))
            plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))    # Ensure correctly formated ticks
            plt.gca().set_aspect('equal')
            plt.gca().autoscale(tight=True)
            plt.pause(.1)
            if i==0:
                prev_q = q
                prev_p = p
            else:
                print(f"cross scalar product : {(prev_p * GK.KRed(prev_q,q,p)).sum()}")
            if savefigs:
                plt.savefig(f"figs/{savefigs_name}_{i}.{format}", format=format, bbox_inches='tight')
        input()
