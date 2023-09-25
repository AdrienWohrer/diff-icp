'''
LDDMM model for point sets (classic or with logdet term)
'''

# Adapted from the LDDMM demo script of library KeOps :
# https://www.kernel-operations.io/keops/_auto_tutorials/surface_registration/plot_LDDMM_Surface.html
# A Wohrer, 2023


# Standard imports

import math

import torch

### Import kernel reductions, from other file in this directory
from diffICP.tools.kernel import SVDpow, GaussKernel, GenKernel
from diffICP.tools.spec import defspec, getspec
from diffICP.tools.optim import LBFGS_optimization
from diffICP.tools.integrators import EulerIntegrator, RalstonIntegrator

#####################################################################################################################
# Encapsulate all the "LDDMM logic" in a class
#####################################################################################################################

# LDDMMM based on vector fields of the form
#           v(x) = \sum_j [p_j K(x-q_j) - eta*(nablaK)(x-q_j) ]
# with eta=0 [if gradcomponent=False]  or  eta=1/lambda [if gradcomponent=True]

class LDDMMModel:

    #############################################################################################
    ### Constructor

    def __init__(self, sigma=1, D=2, lambd=1,
                 spec=defspec, gradcomponent=True, withlogdet=True, version=None,
                 usetrajcost=True, computversion="keops", scheme="Ralston", nonsupprev=False, nt=10):

        # Gaussian kernel: only choice so far
        self.Kernel = GaussKernel(sigma, D, computversion=computversion, spec=spec)     # (variable spec is only used here, to transfer to the kernel)
        self.D = D
        self.lam = lambd
        self.nt = nt

        # Shortcut "version" keyword overrides the precise model used
        if version=="classic":
            gradcomponent, withlogdet, usetrajcost = False, False, False
        elif version=="logdet":
            gradcomponent, withlogdet, usetrajcost = True, True, True
        elif version=="hybrid":         # optimize a "classic" LDMMM vector field (without gradcomponent) but with a logdet energy term
            gradcomponent, withlogdet, usetrajcost = False, True, True

        self.withlogdet = withlogdet
        self.gradcomponent = gradcomponent
        if gradcomponent:
            self.eta = 1.0 / lambd
        else:
            self.eta = 0

        # Computational detail : compute hamiltonian and divcost separately, or jointly ("trajcost") ?
        self.usetrajcost = usetrajcost
        # Computational detail : use reversed reductions for non-support points (x). A priori, not interesting (slower) but kept for testing on larger datasets
        self.nonsupprev = nonsupprev
        # Which integration scheme ?
        self.scheme, self.Integrator = None, None
        self.set_integration_scheme(scheme)


    def set_integration_scheme(self, scheme: str):
        '''
        Set integration scheme for underlying ODE.
        :param scheme: "Euler" or "Ralston" (only schemes implemented so far)
        '''
        self.scheme = scheme
        if scheme == "Euler":   # faster but (much) less stable in case of large deformations !
            self.Integrator = EulerIntegrator
        elif scheme == "Ralston":
            self.Integrator = RalstonIntegrator
        else:
            raise ValueError(f"Unkown numerical scheme : {scheme}")


    # "Post-creation" function when Unpickling with pickle/dill. See https://docs.python.org/3/library/pickle.html#handling-stateful-objects
    def __setstate__(self, state):
        self.__dict__.update(state)
        # use local spec (type/device) for torch tensors
        self.Kernel = GaussKernel(self.Kernel.sigma, self.Kernel.D, self.Kernel.computversion, spec=defspec)

    #####################################################################################################################
    # LDDMM Hamiltonian functions

    # Hamiltonian system based on a Hamiltonian of the form (with nabla=gradient operator, Delta=Laplacian operator)
    # H(q,p) = 1/2* \sum_{ij} [(pi^t.pj)K(qi-qj) - eta.(pi-pj)^t.(nablaK)(qi-qj) - eta**2.(DeltaK)(qi-qj) ]

    # The heavy computations are those made by KeOps during calls to the various reductions KRed(...), gradKRed(...), etc.
    # Thus, the implementation should make sure to call these functions only the minimal number of times required.

    ### Evaluate v at given points x, given vector field parameters (q,p)

    def v(self, x, q, p):
        spec = getspec(x,q,p)
        if x.numel() == 0:  # skip computation if empty
            return torch.empty(x.shape, **spec)
        if self.eta == 0:   # accelerate computation in this case
            return self.Kernel.KRed(x, q, p)
        else:
            return self.Kernel.KRed(x, q, p) - self.eta * self.Kernel.GradKRed(x, q)

    ### Evaluate the sum of -div(v) at given points x.

    def mdivsum(self, x, q, p, rev=False):
        spec = getspec(x,q,p)
        if x.numel() == 0:  # skip computation if empty
            return torch.tensor([0.0], **spec)
        if self.eta == 0:   # accelerate computation in this case
            if rev: return self.Kernel.GradKRed_rev(q,x,p).sum()  # (use reverse summation order)
            else: return (p * self.Kernel.GradKRed(q, x)).sum()
        else:               # mdivsum = Bsum + 2*eta* Csum
            if rev: return self.Kernel.GradKRed_rev(q,x,p).sum() + self.eta * self.Kernel.LapKRed(x, q).sum()  # (use reverse summation order)
            else: return (p * self.Kernel.GradKRed(q, x)).sum() + self.eta * self.Kernel.LapKRed(q, x).sum()

    ### Return the system Hamitonian

    def Hamiltonian(self, q, p):
        getspec(q,p)        # just checking coherent specs (cleaner)
        if self.eta == 0:   # accelerate computation in this case
            H = 0.5 * (p * self.Kernel.KRed(q, q, p)).sum()
        else:               # H = Asum - eta*Bsum - eta^2 *Csum
            H = 0.5 * (p * self.Kernel.KRed(q, q, p)).sum() \
                - self.eta * (p * self.Kernel.GradKRed(q, q)).sum() \
                - 0.5 * self.eta ** 2 * self.Kernel.LapKRed(q, q).sum()
        # Remark: Hamiltonian system ODE could be then derived from autograd-computed partial derivatives of H, e.g.,
        # vq, Gq = grad(H, (p), create_graph=True), grad(H, (q), create_graph=True)
        # However, this is markedly slower (~= 2 times) than the hard-coded reductions used in function ODE below
        return H

    ### Alternatively: evaluate the "trajcost derivative" (lambda*Hamiltonian(q,p) + mdivsum(q,q,p)) for given values of (q,p)

    def dtrajcost(self, q, p):
        getspec(q,p)            # just checking coherent specs (cleaner)
        if self.eta == 0:       # accelerate computation in this case
            return self.lam * 0.5 * (p * self.Kernel.KRed(q, q, p)).sum()
        else:                   # dtrajcost = lambda*Asum + eta*Csum
            return self.lam * 0.5 * (p * self.Kernel.KRed(q, q, p)).sum() + 0.5 * self.eta * self.Kernel.LapKRed(q,q).sum()

    ### Hamiltonian system ODE function, such that d/dt(q,p,cost,x) = ODE(q,p,cost,x)
    # (New version with additional non-support points x)

    def ODE(self, q, p, cost, x):
        spec = getspec(q,p,cost,x)
        # Speeds (support points q + non-support points x)
        vq = self.v(q, q, p)
        vx = self.v(x, q, p)
        # Evolution of momentum p (at support points q) : dp/dt = -partial(H,q)
        # Hard-coded formula is faster than pytorch autograd (see remark in function Hamiltonian above)
        if self.eta == 0:       # accelerate computation in this case
            Gq = self.Kernel.GenDKRed(q, q, p, p)
        else:
            Gq = self.Kernel.GenDKRed(q, q, p, p) \
                 - self.eta * self.Kernel.HessKRed(q, q, p, p) \
                 - self.eta ** 2 * self.Kernel.GradLapKRed(q, q)
        # Divergence term : sum of -div(v) at all points (support and non-support)
        if self.withlogdet:
            if self.usetrajcost:
                dcost = self.dtrajcost(q, p) + self.mdivsum(x, q, p, self.nonsupprev)  # "trajcost" version : slightly faster
            else:
                dcost = self.mdivsum(q, q, p) + self.mdivsum(x, q, p, self.nonsupprev)  # "divcost" version : slightly clearer
        else:
            if self.usetrajcost:
                dcost = self.dtrajcost(q, p)
            else:
                dcost = torch.tensor([0.0], **spec)
        return vq, -Gq, dcost, vx


    ############################# ########################################################################
    ### Conversions between lagrangian speeds (v) and hamiltonian moments (p) based on kernel K(q,q)
    #       v_i = \sum_j K(q_i-q_j)p_j - eta * \sum_j (nablaK)(q_i-q_j)

    # WARNING : ill-posed inverse problem in general. Use with caution !

    def v2p(self, q, v, rcond=1e-3, alpha=1e-4, version='pinv'):
        getspec(q,v)      # just checking coherent specs (cleaner)
        if version == 'pinv':
            return self.Kernel.KpinvSolve(q, v + self.eta * self.Kernel.GradKRed(q, q), rcond)
        elif version == 'ridge_keops':
            return self.Kernel.KridgeSolve_keops(q, v + self.eta * self.Kernel.GradKRed(q, q), alpha)
        elif version == 'ridge_pytorch':
            return self.Kernel.KridgeSolve_pytorch(q, v + self.eta * self.Kernel.GradKRed(q, q), alpha)
        else:
            raise ValueError("unknown version")

    ### Random generation of momentums p at some fixed positions q,
    ### following the associated Bayesian prior : P(p) ~ exp( -lambda * H(q,p) )
    # (Warning: ill-posed in general ; in case of trouble, probably use lower rcond or higher alpha)

    def random_p(self, q, rcond=1e-3, alpha=1e-4, version='svd'):
        spec = getspec(q)
        if self.eta != 0:
            raise ValueError("random_p not implemented yet when gradcomponent=True. (But it shouldn't be too hard!) ")

        K = (-(q[:, None, :] - q[None, :, :]) ** 2 / (2 * self.Kernel.sigma ** 2)).sum(-1).exp()
        zeta = torch.randn(q.shape, **spec) / math.sqrt(self.lam)  # zeta ~ N(0,1/lam)

        if version == 'svd':
            return (SVDpow(K, -0.5, rcond) @ zeta).contiguous()
        elif version == 'ridge':
            # return torch.linalg.torch.solve(zeta, torch.cholesky(
            #     K + alpha * torch.eye(K.shape[0]))).solution.contiguous()  # Pytorch 1.7 (=old) commands
            return torch.linalg.solve(torch.linalg.cholesky(K + alpha* torch.eye(K.shape[0],**spec)), zeta).contiguous()  # torch 2.0
        else:
            raise ValueError("Unknown version")


    ##################################################################################################
    # Geodesic shooting and optimization

    ### Simulate the geodesic ODE with initial condition (q0,p0)

    def Shoot(self, q0, p0, x0=None) :
        spec = getspec(q0,p0,x0)
        if x0 is None:
            x0 = torch.empty(0,self.D, **spec)
        cost0 = torch.tensor([0.0], **spec)
        return self.Integrator(self.ODE, (q0, p0, cost0, x0), self.nt)

    ### "Basic" loss function for LDDMM landmarks (shown here, as an option by default)
    # - each yn is the target of corresponding xn

    def QuadLoss(self, y, cmul=1):
        y = y.detach()      # (to be sure)
        def dataloss(q,x):  # (x is unused here)
            return ((q - y) ** 2).sum() * cmul / 2

        return dataloss

    ### LDDMM trajectory energy (i.e., WITHOUT the datacost) associated to a given geodesic.
    # Assumes that the shooting has already been done, and stored in variable "shoot" (as returned by self.Shoot)

    def trajloss(self, shoot):
        _,_,cost,_ = shoot[-1]      # full cost computed during the shooting, detailed below
        if self.usetrajcost:
            # "trajcost" computation version : cost = lambda*Hamiltonian(q,p) + divcost -- marginally faster
            L = cost
        else:
            # standard "divcost" computation version : cost = divcost
            q0,p0,_,_ = shoot[0]    # trick (kind of useless, we could also provide q0 and p0 explicitly)
            L = self.lam * self.Hamiltonian(q0,p0) + cost
        return L


    ### Optimization of E(p0) (trajectory energy for geodesic shooting with initial momenta p0)

    def Optimize(self, dataloss, q0, p0, x0=None, nmax=10, tol=1e-3, errthresh=1e8):

        spec = getspec(q0, p0, x0)

        if x0 is None:
            x0 = torch.empty(0, self.D, **spec)

        # Make sure not to accumulate computation graph across successive calls to Optimize
        q0 = q0.detach()
        x0 = x0.detach()

        # Loss function (total registration cost function = data loss + LDDMM traj cost)
        def lossfunc(p0):
            shoot = self.Shoot(q0, p0, x0)
            q, _, _, x = shoot[-1]
            L = self.trajloss(shoot) + dataloss(q, x)
            return L

        # Optimize lossfunc w.r.t. p0 (LBFGS algorithm, now externalized to helper function)
        p0, nsteps, change = LBFGS_optimization([p0], lossfunc, nmax=nmax, tol=tol, errthresh=errthresh)
        p0 = p0[0]

        # TODO: various reset strategies when optimization failed
        # Found no better value than p0prev. Try a "compromise" between two imperfect reset strategies:
        # Option 1 : revert to previous parameters p0prev ---> but then global optimization loop might get stuck
        # Option 2 : reset speeds at 0 --> but then all previous work in the global optimization loops is lost
        # p0 = 0.9*p0prev + 0.1*self.v2p(q0, torch.zeros(q0.shape, **spec))
        # print("Exiting current optimization of p0. Multiplicative modification of p0 towards zero speeds.")
        # Option 3 : add some noise to (initial speeds encoded by) momentum p0prev
        # rmod = 0.01
        # p0 = p0prev + rmod * p0prev.std() * self.v2p(q0, torch.randn(q0.shape, **spec))
        # print(
        #     f"Exiting current optimization of p0. Trying a random perturbation of p0 from its current value, with relative strength {rmod}.")
        
        # One last shoot, just to compute variables of interest while they are easily accesible
        shoot = self.Shoot(q0, p0, x0)
        trajl = self.trajloss(shoot).item()
        q, _, _, x = shoot[-1]
        datal = dataloss(q, x).item()

        # print("optim steps ", nsteps, " loss =", trajl+datal, " change =", change)
        return p0, shoot, trajl, datal, nsteps, change


    ### Optimization of E(p0) (trajectory energy for geodesic shooting with initial momenta p0)
    ### Legacy version, should be removed soon

    def Optimize_legacy(self, dataloss, q0, p0, x0=None, nmax=10, tol=1e-3, errthresh=1e8, **kwargs):

        spec = getspec(q0,p0,x0)

        if x0 is None:
            x0 = torch.empty(0,self.D, **spec)

        # Necessary to not accumulate computation graph across successive calls to Optimize
        p0 = p0.detach().requires_grad_(True)
        q0 = q0.detach()
        x0 = x0.detach()
        # Warning: any external pytorch tensor serving as parameters in dataloss(q,x) should also be detached (see example in QuadLoss above)

        # optimizer = torch.optim.LBFGS([p0], max_eval=10, max_iter=10, history_size=100, **kwargs) # original values
        optimizer = torch.optim.LBFGS([p0], max_iter=20, max_eval=100, history_size=100, line_search_fn="strong_wolfe", **kwargs)     # try stuff
        # optimizer = torch.optim.Adam([p0], lr=0.001)     # try stuff... meh
        # optimizer = torch.optim.SGD([p0], lr=0.00001)     # try stuff... crap
        # optimizer = torch.optim.RMSprop([p0])     # try stuff... meh

        # Keep some track of the optimizer's repeated function evaluations during this step (for manual debug and handling of exceptions)
        iter_L, best_L, best_p0 = [], None, None

        def closure(opt=True):
            if opt:
                optimizer.zero_grad()       # reset grads to 0 between every call to function .backward()
            shoot = self.Shoot(q0, p0, x0)
            q,_,_,x = shoot[-1]
            L = self.trajloss(shoot) + dataloss(q,x)
            if opt:
                L.backward()                # Update gradients of all the chain of computations from p0 to L
                Ld = L.detach().item()      # manual tracking of function evaluations and parameters
                iter_L.append(Ld)           # list of all values of L encountered during current optimizer step
                nonlocal best_L, best_p0    # https://stackoverflow.com/questions/64323757/why-does-python-3-8-0-allow-to-change-mutable-types-from-enclosing-function-scop
                if Ld < best_L :
                    best_L = Ld
                    best_p0 = p0.clone().detach()
            return L

        i, keepOn = 0, True
        while i < nmax and keepOn:
            i += 1
            p0prev = p0.clone().detach()
            iter_L, best_L, best_p0 = [], math.inf, None            # keep track of evaluations during optimizer.step
            # For closure-based optimizer (like LBFGS)
            optimizer.step(closure)                                 # The long line !
            Lprev, L = iter_L[0], iter_L[-1]                        # value of L before / after optimizer step
            # For single-evaluation optimizers (like Adam, SGD...)
            # L = closure()
            # optimizer.step()

            if L > Lprev or L > errthresh or math.isnan(L):
                # Detect some form of divergent behavior!

                # Print some debug information
                if math.isnan(L):
                    print("WARNING: NaN value for loss L during LDDMM optimization.")
                elif L > errthresh:
                    print("WARNING: Aberrantly large value for loss L during LDDMM optimization.")
                elif L > Lprev:
                    print("WARNING: increase of loss L during L-BGFS optimization of LDDMM cost function.")
                print(f"iter {i} , all L values during iteration : {iter_L}")
                print(f"iter {i} , best L value during iteration : {best_L}")
                print(f"iter {i} , last L value during iteration : {L}")

                # Use some fallback value for p0
                if best_L < Lprev :
                    # Some better p0 value than p0prev has been encoutered during the optimizer step --> use it.
                    p0 = best_p0
                    print("Exiting current optimization of p0. Found an intermediate 'best_p0' value to use instead.")
                else:
                    # Found no better value than p0prev. Try a "compromise" between two imperfect reset strategies:
                    # Option 1 : revert to previous parameters p0prev ---> but then global optimization loop might get stuck
                    # Option 2 : reset speeds at 0 --> but then all previous work in the global optimization loops is lost
                    # p0 = 0.9*p0prev + 0.1*self.v2p(q0, torch.zeros(q0.shape, **spec))
                    # print("Exiting current optimization of p0. Multiplicative modification of p0 towards zero speeds.")
                    # Option 3 : add some noise to (initial speeds encoded by) momentum p0prev
                    rmod = 0.01
                    p0 = p0prev + rmod * p0prev.std() * self.v2p(q0, torch.randn(q0.shape, **spec))
                    print(f"Exiting current optimization of p0. Trying a random perturbation of p0 from its current value, with relative strength {rmod}.")

                change = "STOP"
                keepOn = False                                                      # exit the iterations (optimization has failed)

            else:  # normal behavior
                change = ((p0 - p0prev) ** 2).mean().sqrt().detach().cpu().numpy()  # change in parameter value
                keepOn = change > tol

        # Done ! Return p0 as "inert" parameter, without gradient attached (important!)
        p0 = p0.detach()
        # One last shoot, just to compute variables of interest while they are easily accesible
        shoot = self.Shoot(q0, p0, x0)
        trajl = self.trajloss(shoot).item()
        q,_,_,x = shoot[-1]
        datal = dataloss(q,x).item()

        #print("optim steps ", i, " loss =", trajl+datal, " change =", change)
        return p0, shoot, trajl, datal, i, change



###########################################################
### Testing
###########################################################


if __name__ == '__main__':
    # Running as a script

    M, D, sig, lam = 10, 2, 2.0, 100.0
    xt = torch.randn(M, D, **defspec)
    bt = torch.randn(M, D, **defspec)

    LDDMM = LDDMMModel(sig, D, lambd=lam, version="classic")

    vt = LDDMM.v(xt,xt,bt)
    pt = LDDMM.v2p(xt,vt, rcond=None)
    vback = LDDMM.v(xt,xt,pt)
    print(vt)
    print(vback)    # ok, vt == vback
    print("yo")
    print(bt)
    print(pt)       # but bt != pt, because system has mutiple solutions, and pt has smaller norm than bt

    print('Yo')
    print(LDDMM.random_p(xt))
    ##    print("yeah")
    ##    print(LDDMM.random_p(xt, version='ridge'))       # fail when N big (matrix singular up to numeric precision)

