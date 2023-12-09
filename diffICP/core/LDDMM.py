'''
LDDMM model for point sets (classic or with logdet term)
'''

# Adapted from the LDDMM demo script of library KeOps :
# https://www.kernel-operations.io/keops/_auto_tutorials/surface_registration/plot_LDDMM_Surface.html
# A Wohrer, 2023


# Standard imports
import math
import torch

### Import from other diffICP tool files : kernel reductions, optimizers, integrators, etc.
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
    ### Constructor + various initialization functions

    def __init__(self, sigma=1.0, D=2, lambd=2.0,
                 spec=defspec, gradcomponent=True, withlogdet=True, version=None,
                 computversion="keops", scheme="Ralston", nonsupprev=False, nt=10):

        # Gaussian kernel: only choice so far
        self.Kernel = GaussKernel(sigma, D, computversion=computversion, spec=spec)     # (variable spec is only used here, to transfer to the kernel)
        self.D = D
        self.lam = lambd
        self.nt = nt

        # Shortcut "version" keyword overrides the precise model used
        if version=="classic":
            gradcomponent, withlogdet = False, False
        elif version=="logdet":
            gradcomponent, withlogdet = True, True
        elif version=="hybrid":         # optimize a "classic" LDMMM vector field (without gradcomponent) but with a logdet energy term
            gradcomponent, withlogdet = False, True

        self.withlogdet = withlogdet
        self.gradcomponent = gradcomponent
        if gradcomponent:
            self.eta = 1.0 / lambd
        else:
            self.eta = 0
        # Computational detail : use reversed reductions for non-support points (x). A priori, not interesting (slower) but kept for testing on larger datasets
        self.nonsupprev = nonsupprev
        # Which integration scheme ?
        self.scheme, self.Integrator = None, None
        self.set_integration_scheme(scheme)

        # Computational detail : use optimization available in some cases ("trajcost" = joint computation of hamiltonian and divcost) ?
        # Cannot be modified in __init()__, modify externally if required (e.g. for testing)
        self.try_trajcost_optim = False         # (it seems safer to always use the same computation)

    ################################

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

    ##################################

    # "Post-creation" function when Unpickling. See https://docs.python.org/3/library/pickle.html#handling-stateful-objects
    def __setstate__(self, state):
        self.__dict__.update(state)
        # use local spec (type/device) for torch tensors
        self.Kernel = GaussKernel(self.Kernel.sigma, self.Kernel.D, self.Kernel.computversion, spec=defspec)


    #####################################################################################################################
    ### LDDMM Hamiltonian functions

    # Hamiltonian system based on a Hamiltonian of the form (with nabla=gradient operator, Delta=Laplacian operator)
    # H(q,p) = 1/2* \sum_{ij} [(pi^t.pj)K(qi-qj) - eta.(pi-pj)^t.(nablaK)(qi-qj) - eta**2.(DeltaK)(qi-qj) ]

    # The heavy computations are those made by KeOps during calls to the various reductions KRed(...), gradKRed(...), etc.
    # Thus, the implementation should make sure to call these functions only the minimal number of times required.

    def v(self, x, q, p):
        '''
        Evaluate RKHS vector field v at given points x, given support parameters (q,p) :
            v(x) = \sum_j [p_j K(x-q_j) - eta*(nablaK)(x-q_j) ]

        :param x: torch.tensor of size (Nx,D) : points where evaluation should be made
        :param q: torch.tensor of size (M,D) : support points of the RKHS vector field
        :param p: torch.tensor of size (M,D) : vector weights (momenta) at the support points q
        :return: torch.tensor of size (Nx,D) corresponding to v(x)
        '''
        spec = getspec(x,q,p)
        if x.numel() == 0:  # skip computation if empty
            return torch.empty(x.shape, **spec)
        if self.gradcomponent:
            return self.Kernel.KRed(x, q, p) - self.eta * self.Kernel.GradKRed(x, q)
        else:           # accelerate computation in this case
            return self.Kernel.KRed(x, q, p)

    #####################################

    def mdivsum(self, x, q, p, rev=False):
        '''
        Evaluate the sum of -div(v) at given points x.
        :param x: see self.v
        :param q: see self.v
        :param p: see self.v
        :param rev: use a computational scheme with reverse summation order (generally not useful)
        :return: torch.tensor of size (1,) corresponding to -\sum_k div(v)(x_k)
        '''

        spec = getspec(x,q,p)
        if x.numel() == 0:  # skip computation if empty
            return torch.tensor([0.0], **spec)
        if self.gradcomponent:      # mdivsum = Bsum + 2*eta* Csum
            if rev: return self.Kernel.GradKRed_rev(q,x,p).sum() + self.eta * self.Kernel.LapKRed(x, q).sum()  # (use reverse summation order)
            else: return (p * self.Kernel.GradKRed(q, x)).sum() + self.eta * self.Kernel.LapKRed(q, x).sum()
        else:                       # accelerate computation in this case
            if rev: return self.Kernel.GradKRed_rev(q,x,p).sum()  # (use reverse summation order)
            else: return (p * self.Kernel.GradKRed(q, x)).sum()

    ###########################

    def Hamiltonian(self, q, p):
        '''
        Value of the system Hamiltonian given RKHS support parameters (q,p)
        :param q: see self.v
        :param p: see self.v
        :return: H(q,p) := 1/2* \sum_{ij} [(pi^t.pj)K(qi-qj) - eta.(pi-pj)^t.(nablaK)(qi-qj) - eta**2.(DeltaK)(qi-qj) ]
        '''
        getspec(q,p)        # just checking coherent specs (cleaner)
        if self.gradcomponent:          # H = Asum - eta*Bsum - eta^2 *Csum
            H = 0.5 * (p * self.Kernel.KRed(q, q, p)).sum() \
                - self.eta * (p * self.Kernel.GradKRed(q, q)).sum() \
                - 0.5 * self.eta ** 2 * self.Kernel.LapKRed(q, q).sum()
        else:                           # accelerate computation in this case
            H = 0.5 * (p * self.Kernel.KRed(q, q, p)).sum()
        # Remark: Hamiltonian system ODE could be then derived from autograd-computed partial derivatives of H, e.g.,
        # vq, Gq = grad(H, (p), create_graph=True), grad(H, (q), create_graph=True)
        # However, this is markedly slower (~= 2 times) than the hard-coded reductions used in function ODE below
        return H

    ############################

    def dtrajcost(self, q, p):
        '''
        "Trajcost derivative" (only used in alternative computational scheme, when self.try_trajcost_optim=True)
        :param q: see self.v
        :param p: see self.v
        :return: (dL/dt)(q,p) := lambda*Hamiltonian(q,p) + mdivsum(q,q,p)
        '''
        getspec(q,p)                # checking coherent specs (cleaner)
        # dtrajcost = lambda*Asum + eta*Csum
        return 0.5 * self.lam * (p * self.Kernel.KRed(q, q, p)).sum() + 0.5 * self.eta * self.Kernel.LapKRed(q,q).sum()

    ############################

    def ODE(self, q, p, cost, x=None):
        '''
        Hamiltonian system ODE function, such that d/dt(q,p,cost,x) = ODE(q,p,cost,x).
        :param q: support points
        :param p: support momenta
        :param x: external "data point set" evolving under the ODE, or None [in which case the "data point set" is q].
        :param cost: value of some relevant cost function along the trajectory :
            - if not self.withlogdet :
                    cost = 0 ;
            - if self.withlogdet and self.gradcomponent and x==None and self.try_trajcost_optim :
                    cost = trajcost     [numerical speed-up in that specific case] ;
            - if self.withlogdet and any other configuration :
                    cost = divcost (integral from t =0 to 1 of mdivsum([q or x],q,p)) ;
        :return: time derivatives of (q,p,cost[,x]), as a tuple    [if x==None, no 4th element is output]
        '''

        spec = getspec(q,p,cost,x)
        # Speeds  at support points q
        vq = self.v(q, q, p)

        # Evolution of momentum p (at support points q) : dp/dt = -partial(H,q)
        # Hard-coded formula is faster than pytorch autograd (see remark in function Hamiltonian above)
        if self.eta == 0:       # accelerate computation in this case
            Gq = self.Kernel.GenDKRed(q, q, p, p)
        else:
            Gq = self.Kernel.GenDKRed(q, q, p, p) \
                 - self.eta * self.Kernel.HessKRed(q, q, p, p) \
                 - self.eta ** 2 * self.Kernel.GradLapKRed(q, q)

        # Divergence term : sum of -div(v) at data points (x if given, else q)

        if x is None:
            # data points = support points = q

            if self.withlogdet:
                if self.gradcomponent and self.try_trajcost_optim:
                    dcost = self.dtrajcost(q, p)    # = lambda*Hamiltonian(q,p) + mdivsum(q,q,p) , special speed-up ;
                else:
                    dcost = self.mdivsum(q, q, p)
            else:
                dcost = torch.tensor([0.0], **spec)
            return vq, -Gq, dcost

        else:
            # data points = external point set = x

            if self.withlogdet:
                dcost = self.mdivsum(x, q, p, self.nonsupprev)
            else:
                dcost = torch.tensor([0.0], **spec)
            vx = self.v(x, q, p)
            return vq, -Gq, dcost, vx

    ######################################################################################################
    ### Conversions between lagrangian speeds (v) and hamiltonian moments (p) based on kernel K(q,q)
    #       v_i = \sum_j K(q_i-q_j)p_j - eta * \sum_j (nablaK)(q_i-q_j)

    # WARNING : ill-posed inverse problem in general. Use with caution !

    def v2p(self, q, v, rcond=1e-3, alpha=1e-4, version='pinv'):
        '''
        Inverse problem : estimate momenta p such that self.v(q,q,p) ~= v
        :param q: torch.tensor of size (M,D) : support points ;
        :param v: torch.tensor of size (M,D) : target speeds at the support points ;
        :param rcond: smoothing parameter in the pseudo-inverse inversion method ;
        :param alpha: smoothing parameter in the ridge inversion method ;
        :param version: inversion method : 'pinv' or 'ridge_keops' or 'ridge_pytorch' ;
        :return: torch.tensor of size (M,D) : estimated momenta p (at points q)
        '''
        getspec(q,v)      # just checking coherent specs (cleaner)
        if version == 'pinv':
            return self.Kernel.KpinvSolve(q, v + self.eta * self.Kernel.GradKRed(q, q), rcond)
        elif version == 'ridge_keops':
            return self.Kernel.KridgeSolve_keops(q, v + self.eta * self.Kernel.GradKRed(q, q), alpha)
        elif version == 'ridge_pytorch':
            return self.Kernel.KridgeSolve_pytorch(q, v + self.eta * self.Kernel.GradKRed(q, q), alpha)
        else:
            raise ValueError("unknown version")

    ############################################################

    def random_p(self, q, rcond=1e-3, alpha=1e-4, version='svd'):
        '''
        Random generation of momentums p at some fixed positions q, following the associated Bayesian prior :
            P(p) ~ exp( -lambda * H(q,p) )
            Warning: ill-posed in general ; in case of trouble, probably use lower rcond or higher alpha.
        :param q: torch.tensor of size (M,D) : support points ;
        :param rcond: smoothing parameter in the svd inversion method ;
        :param alpha: smoothing parameter in the ridge inversion method ;
        :param version: inversion method : 'svd' or 'ridge'
        :return: torch.tensor of size (M,D) : generated momenta p (at points q)
        '''
        spec = getspec(q)
        if self.eta != 0:
            raise ValueError("random_p not implemented yet when gradcomponent=True. (But it shouldn't be too hard!) ")

        K = (-(q[:, None, :] - q[None, :, :]) ** 2 / (2 * self.Kernel.sigma ** 2)).sum(-1).exp()
        zeta = torch.randn(q.shape, **spec) / math.sqrt(self.lam)  # zeta ~ N(0,1/lam)

        if version == 'svd':
            return (SVDpow(K, -0.5, rcond) @ zeta).contiguous()
        elif version == 'ridge':
            return torch.linalg.solve(torch.linalg.cholesky(K + alpha* torch.eye(K.shape[0],**spec)), zeta).contiguous()  # torch >= 2.0
        else:
            raise ValueError("Unknown version")


    ##################################################################################################
    ### Geodesic shooting and optimization

    def Shoot(self, q0, p0, x0=None) :
        '''
        Simulate the geodesic ODE with initial condition (q0,p0).
        :param q0: torch.tensor of size (M,D) : initial value of support points ;
        :param p0: torch.tensor of size (M,D) : initial value of momenta ;
        :param x0: torch.tensor of size (Nx,D) : external "data point set" undergoing the ODE, or None [in which case "data point set" is q]
        :return: a "shoot" variable : list of (q,p,cost[,x]) at all integration times t
        '''
        spec = getspec(q0,p0,x0)
        cost0 = torch.tensor([0.0], **spec)
        if x0 is None:
            return self.Integrator(self.ODE, (q0, p0, cost0), self.nt)
        else:
            return self.Integrator(self.ODE, (q0, p0, cost0, x0), self.nt)

    #############################

    def BasicQuadLossFunctor(self, y, cmul=1):
        '''
        Functor returning a "basic" loss function, for LDDMM landmarks. (Added here to provide an example of what to use in self.Optimize.)
        -
        :param y: each yn is the target of corresponding warped support point qn ;
        :param cmul: possibly add an overall multiplicative factor ;
        :return: a function dataloss, such that x->dataloss(x) can be used as input to self.Optimize().
        '''
        y = y.detach()      # (to be sure)
        def dataloss(x):
            return ((x - y) ** 2).sum() * cmul / 2
        return dataloss

    ############################

    def trajloss(self, shoot):
        '''
        LDDMM trajectory energy (i.e., WITHOUT the datacost) associated to a given geodesic.
        :param shoot: "shoot" variable containg the whole trajectory, as produced by self.Shoot.
        :return: associated trajloss (a single number)
        '''
        arrival = shoot[-1]
        cost = arrival[2]
        is_x = len(arrival) == 4    # (ugly detection of external data point set)
        if not is_x and self.withlogdet and self.gradcomponent and self.try_trajcost_optim:
            # "trajcost" optimization : cost = lambda*Hamiltonian(q,p) + divcost [marginally faster in that situation]
            L = cost
        else:
            # normal computation version : cost = divcost
            q0,p0 = shoot[0][:2]    # hack
            L = self.lam * self.Hamiltonian(q0,p0) + cost
        return L

    #############################

    def Optimize(self, dataloss, q0, p0, x0=None, nmax=10, tol=1e-3, errthresh=1e8):
        '''
        Optimization of E(p0) (trajectory energy for geodesic shooting with initial momenta p0). Solve
            min_{p0} trajcost(p0) + dataloss(q1(q0,p0)) [resp. dataloss(x1(q0,p0,x0))]
            where q1 [resp x1] are the arrival points (at time t=1) of the LDDMM geodesic of initial momenta p0.

        :param dataloss: dataloss function, should take the form dataloss(x) where x are the (warped) data points. See self.BasicQuadLossFunctor for an example.
        :param q0: location of the support points q at time t=0 (this is fixed).
        :param p0: initial value for momenta p at time t=0 (this is the quantity being optimized).
        :param x0: location of "data points" x at time t=0 (this is fixed). If x0=None, the "data points" are assumed to be x=q.
        :param nmax: maximum number of calls to the underlying optimization function (see optim.py)
        :param tol: tolerance of the underlying optimization function (see optim.py)
        :param errthresh: error threshold of the underlying optimization function (see optim.py)
        :return: optimized value for parameter p0.
        '''

        getspec(q0, p0, x0)
        is_x = x0 is not None

        # Make sure not to accumulate computation graph across successive calls to Optimize
        q0 = q0.detach()
        if is_x:
            x0 = x0.detach()

        # Loss function (total registration cost function = data loss + LDDMM traj cost)
        def lossfunc(p0):
            shoot = self.Shoot(q0, p0, x0)
            if is_x:
                x = shoot[-1][-1]
                L = self.trajloss(shoot) + dataloss(x)
            else:
                q = shoot[-1][0]
                L = self.trajloss(shoot) + dataloss(q)
            return L

        # Optimize lossfunc w.r.t. p0 (LBFGS algorithm, now externalized to helper function)
        p0, _, nsteps, change = LBFGS_optimization([p0], lossfunc, nmax=nmax, tol=tol, errthresh=errthresh)
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
        if is_x:
            datal = dataloss(shoot[-1][-1]).item()
        else:
            datal = dataloss(shoot[-1][0]).item()
        # print(p0, shoot, trajl, datal, nsteps, change)

        return p0, shoot, trajl, datal, nsteps, change




#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

# OLD VERSION -- DO NOT DELETE YET


class LDDMMModel_old:

    #############################################################################################
    ### Constructor + various initialization functions

    def __init__(self, sigma=1.0, D=2, lambd=2.0,
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

    ################################

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

    ##################################

    # "Post-creation" function when Unpickling. See https://docs.python.org/3/library/pickle.html#handling-stateful-objects
    def __setstate__(self, state):
        self.__dict__.update(state)
        # use local spec (type/device) for torch tensors
        self.Kernel = GaussKernel(self.Kernel.sigma, self.Kernel.D, self.Kernel.computversion, spec=defspec)


    #####################################################################################################################
    ### LDDMM Hamiltonian functions

    # Hamiltonian system based on a Hamiltonian of the form (with nabla=gradient operator, Delta=Laplacian operator)
    # H(q,p) = 1/2* \sum_{ij} [(pi^t.pj)K(qi-qj) - eta.(pi-pj)^t.(nablaK)(qi-qj) - eta**2.(DeltaK)(qi-qj) ]

    # The heavy computations are those made by KeOps during calls to the various reductions KRed(...), gradKRed(...), etc.
    # Thus, the implementation should make sure to call these functions only the minimal number of times required.

    def v(self, x, q, p):
        '''
        Evaluate RKHS vector field v at given points x, given support parameters (q,p) :
            v(x) = \sum_j [p_j K(x-q_j) - eta*(nablaK)(x-q_j) ]

        :param x: torch.tensor of size (Nx,D) : points where evaluation should be made
        :param q: torch.tensor of size (M,D) : support points of the RKHS vector field
        :param p: torch.tensor of size (M,D) : vector weights (momenta) at the support points q
        :return: torch.tensor of size (Nx,D) corresponding to v(x)
        '''
        spec = getspec(x,q,p)
        if x.numel() == 0:  # skip computation if empty
            return torch.empty(x.shape, **spec)
        if self.eta == 0:   # accelerate computation in this case
            return self.Kernel.KRed(x, q, p)
        else:
            return self.Kernel.KRed(x, q, p) - self.eta * self.Kernel.GradKRed(x, q)

    #####################################

    def mdivsum(self, x, q, p, rev=False):
        '''
        Evaluate the sum of -div(v) at given points x.
        :param x: see self.v
        :param q: see self.v
        :param p: see self.v
        :param rev: use a computational scheme with reverse summation order (generally not useful)
        :return: torch.tensor of size (1,) corresponding to -\sum_k div(v)(x_k)
        '''

        spec = getspec(x,q,p)
        if x.numel() == 0:  # skip computation if empty
            return torch.tensor([0.0], **spec)
        if self.eta == 0:   # accelerate computation in this case
            if rev: return self.Kernel.GradKRed_rev(q,x,p).sum()  # (use reverse summation order)
            else: return (p * self.Kernel.GradKRed(q, x)).sum()
        else:               # mdivsum = Bsum + 2*eta* Csum
            if rev: return self.Kernel.GradKRed_rev(q,x,p).sum() + self.eta * self.Kernel.LapKRed(x, q).sum()  # (use reverse summation order)
            else: return (p * self.Kernel.GradKRed(q, x)).sum() + self.eta * self.Kernel.LapKRed(q, x).sum()

    ###########################

    def Hamiltonian(self, q, p):
        '''
        Value of the system Hamiltonian given RKHS support parameters (q,p)
        :param q: see self.v
        :param p: see self.v
        :return: H(q,p) := 1/2* \sum_{ij} [(pi^t.pj)K(qi-qj) - eta.(pi-pj)^t.(nablaK)(qi-qj) - eta**2.(DeltaK)(qi-qj) ]
        '''
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

    ############################

    def dtrajcost(self, q, p):
        '''
        "Trajcost derivative" (used in alternative computational scheme, when self.try_trajcost_optim=True)
        :param q: see self.v
        :param p: see self.v
        :return: (dL/dt)(q,p) := lambda*Hamiltonian(q,p) + mdivsum(q,q,p)
        '''

        # TODO : which of q vs x enters the trajcost ? NOT CLEAR IN GENERAL (depends on problem being solved) ! MUST RETHINK ORGANIZATION !
        # THis function is only valid when q corresponds to datapoints being warped (and thus, associated to a logdet term).
        # By contrast, "pure support" points q are not associated to a logdet loss term !
        # Also see TODO in function ODE just below

        getspec(q,p)            # just checking coherent specs (cleaner)
        if self.eta == 0:       # accelerate computation in this case
            return self.lam * 0.5 * (p * self.Kernel.KRed(q, q, p)).sum()
        else:                   # dtrajcost = lambda*Asum + eta*Csum
            return self.lam * 0.5 * (p * self.Kernel.KRed(q, q, p)).sum() + 0.5 * self.eta * self.Kernel.LapKRed(q,q).sum()

    ############################

    def ODE(self, q, p, cost, x):
        '''
        Hamiltonian system ODE function, such that d/dt(q,p,cost,x) = ODE(q,p,cost,x).
        :param q: support points
        :param p: support momenta
        :param cost: value of cost along the trajectory (trajcost or divcost, depending on self.try_trajcost_optim=True/False)
        :param x: external point set evolving under the ODE [optional]
        :return: time derivatives of (q,p,cost,x) [as a tuple]
        '''

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
                # TODO ! Usage of mdivsum is inconstitant when support points q are NOT the data points x being warped,
                # because then mdivsum(q) should not enter the energy functional ! (Only mdivsum(x)).
                # TODO Must reorganize/rethink handling of q vs. x, and correct this true bug !!
                dcost = self.dtrajcost(q, p) + self.mdivsum(x, q, p, self.nonsupprev)  # "trajcost" version : slightly faster
            else:
                dcost = self.mdivsum(q, q, p) + self.mdivsum(x, q, p, self.nonsupprev)  # "divcost" version : slightly clearer
        else:
            if self.usetrajcost:
                dcost = self.dtrajcost(q, p)
            else:
                dcost = torch.tensor([0.0], **spec)
        return vq, -Gq, dcost, vx


    ######################################################################################################
    ### Conversions between lagrangian speeds (v) and hamiltonian moments (p) based on kernel K(q,q)
    #       v_i = \sum_j K(q_i-q_j)p_j - eta * \sum_j (nablaK)(q_i-q_j)

    # WARNING : ill-posed inverse problem in general. Use with caution !

    def v2p(self, q, v, rcond=1e-3, alpha=1e-4, version='pinv'):
        '''
        Inverse problem : estimate momenta p such that self.v(q,q,p) ~= v
        :param q: torch.tensor of size (M,D) : support points ;
        :param v: torch.tensor of size (M,D) : target speeds at the support points ;
        :param rcond: smoothing parameter in the pseudo-inverse inversion method ;
        :param alpha: smoothing parameter in the ridge inversion method ;
        :param version: inversion method : 'pinv' or 'ridge_keops' or 'ridge_pytorch' ;
        :return: torch.tensor of size (M,D) : estimated momenta p (at points q)
        '''
        getspec(q,v)      # just checking coherent specs (cleaner)
        if version == 'pinv':
            return self.Kernel.KpinvSolve(q, v + self.eta * self.Kernel.GradKRed(q, q), rcond)
        elif version == 'ridge_keops':
            return self.Kernel.KridgeSolve_keops(q, v + self.eta * self.Kernel.GradKRed(q, q), alpha)
        elif version == 'ridge_pytorch':
            return self.Kernel.KridgeSolve_pytorch(q, v + self.eta * self.Kernel.GradKRed(q, q), alpha)
        else:
            raise ValueError("unknown version")

    ############################################################

    def random_p(self, q, rcond=1e-3, alpha=1e-4, version='svd'):
        '''
        Random generation of momentums p at some fixed positions q, following the associated Bayesian prior :
            P(p) ~ exp( -lambda * H(q,p) )
            Warning: ill-posed in general ; in case of trouble, probably use lower rcond or higher alpha.
        :param q: torch.tensor of size (M,D) : support points ;
        :param rcond: smoothing parameter in the svd inversion method ;
        :param alpha: smoothing parameter in the ridge inversion method ;
        :param version: inversion method : 'svd' or 'ridge'
        :return: torch.tensor of size (M,D) : generated momenta p (at points q)
        '''
        spec = getspec(q)
        if self.eta != 0:
            raise ValueError("random_p not implemented yet when gradcomponent=True. (But it shouldn't be too hard!) ")

        K = (-(q[:, None, :] - q[None, :, :]) ** 2 / (2 * self.Kernel.sigma ** 2)).sum(-1).exp()
        zeta = torch.randn(q.shape, **spec) / math.sqrt(self.lam)  # zeta ~ N(0,1/lam)

        if version == 'svd':
            return (SVDpow(K, -0.5, rcond) @ zeta).contiguous()
        elif version == 'ridge':
            return torch.linalg.solve(torch.linalg.cholesky(K + alpha* torch.eye(K.shape[0],**spec)), zeta).contiguous()  # torch >= 2.0
        else:
            raise ValueError("Unknown version")


    ##################################################################################################
    ### Geodesic shooting and optimization

    def Shoot(self, q0, p0, x0=None) :
        '''
        Simulate the geodesic ODE with initial condition (q0,p0).
        :param q0: torch.tensor of size (M,D) : initial value of support points ;
        :param p0: torch.tensor of size (M,D) : initial value of momenta ;
        :param x0: torch.tensor of size (Nx,D) : external points undergoing the ODE (optional)
        :return: a "shoot" variable : list of (q,p,cost,x) at all integration times t
        '''
        spec = getspec(q0,p0,x0)
        if x0 is None:
            x0 = torch.empty(0,self.D, **spec)
        cost0 = torch.tensor([0.0], **spec)
        return self.Integrator(self.ODE, (q0, p0, cost0, x0), self.nt)

    #############################

    def BasicQuadLossFunctor(self, y, cmul=1):
        '''
        Functor returning a "basic" loss function, for LDDMM landmarks. (Added here to provide an example of what to use in self.Optimize.)
        -
        :param y: each yn is the target of corresponding warped support point qn ;
        :param cmul: possibly add an overall multiplicative factor ;
        :return: a function dataloss, such that (q,x)->dataloss(q,x) can be used as input to self.Optimize(). Parameter
        x is unused here, but must still be present as a second argument to function dataloss.
        '''
        y = y.detach()      # (to be sure)
        def dataloss(q,x):  # (x is unused here)
            return ((q - y) ** 2).sum() * cmul / 2
        return dataloss

    ############################

    def trajloss(self, shoot):
        '''
        LDDMM trajectory energy (i.e., WITHOUT the datacost) associated to a given geodesic.
        :param shoot: "shoot" variable containg the whole trajectory, as produced by self.Shoot.
        :return: associated trajloss (a single number)
        '''
        _,_,cost,_ = shoot[-1]      # full cost computed during the shooting, detailed below
        if self.usetrajcost:
            # "trajcost" computation version : cost = lambda*Hamiltonian(q,p) + divcost -- marginally faster
            L = cost
        else:
            # standard "divcost" computation version : cost = divcost
            q0,p0,_,_ = shoot[0]    # trick (kind of useless, we could also provide q0 and p0 explicitly)
            L = self.lam * self.Hamiltonian(q0,p0) + cost
        return L

    #############################

    def Optimize(self, dataloss, q0, p0, x0=None, nmax=10, tol=1e-3, errthresh=1e8):
        '''
        Optimization of E(p0) (trajectory energy for geodesic shooting with initial momenta p0). Solve
            min_{p0} trajcost(p0) + dataloss(q1(p0), x1(p0))
            where q1, x1 are the arrival points (at time t=1) of the LDDMM geodesic of initial parameters p0.

        :param dataloss: dataloss function, should take the form dataloss(q,x) where q are the (warped) support points,
            and x the (warped) external points. See self.BasicQuadLossFunctor for an example.
        :param q0: location of the support points q at time t=0 (this is fixed).
        :param p0: initial value for momenta p at time t=0 (this is the quantity being optimized).
        :param x0: location of external points x at time t=0. Leave at None if there are no external points.
        :param nmax: maximum number of calls to the underlying optimization function (see optim.py)
        :param tol: tolerance of the underlying optimization function (see optim.py)
        :param errthresh: error threshold of the underlying optimization function (see optim.py)
        :return: optimized value for parameter p0.
        '''

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
        p0, _, nsteps, change = LBFGS_optimization([p0], lossfunc, nmax=nmax, tol=tol, errthresh=errthresh)
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

        # print(p0, shoot, trajl, datal, nsteps, change)

        return p0, shoot, trajl, datal, nsteps, change


###########################################################
### Testing
###########################################################

if __name__ == '__main__':
    # Running as a script


    M, D, sig, lam = 10, 2, 2.0, 100.0
    xt = torch.randn(M, D, **defspec)
    qt = torch.randn(M, D, **defspec)
    bt = torch.randn(M, D, **defspec)

    if True:
        # TODO transform this part in a real test of the functions !

        # withlogdet=True, gradcomponent=True: ok
        # withlogdet=True, gradcomponent=False: ok
        # withlogdet=False, gradcomponent=True: ok
        # withlogdet=False, gradcomponent=False: ok
        # HUMMM !?

        withlogdet = True
        gradcomponent = True
        for usetrajcost in [True,False]:
            print(f"withlogdet : {withlogdet}")
            print(f"gradcomponent : {gradcomponent}")
            print(f"try_trajcost_optim : {usetrajcost}")
            LDDMM = LDDMMModel(sig, D, lambd=lam, gradcomponent=gradcomponent, withlogdet=withlogdet,
                               scheme="Ralston", nt=15)
            LDDMM.try_trajcost_optim = usetrajcost
            # print(LDDMM.ODE(qt,bt,torch.Tensor([0]),xt))    # --> OK, TESTED
            print(LDDMM.trajloss(LDDMM.Shoot(qt,bt,xt)))    # --> OK, TESTED
        exit()

    ########

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

