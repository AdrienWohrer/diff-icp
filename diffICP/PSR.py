'''
Function implementing the "general" version of diffICP algorithm
'''

import os, time, math
import copy
import warnings

import numpy as np

from matplotlib import pyplot as plt

plt.ion()

import torch

# keops imports
from pykeops.torch import Vi, Vj

#######################################################################
# Import from other files in this directory :

from diffICP.GMM import GaussianMixtureUnif
from diffICP.LDDMM_logdet import LDDMMModel
from diffICP.Affine_logdet import AffineModel
from diffICP.visu import my_scatter, plot_shoot
from diffICP.decimate import decimate
from diffICP.spec import defspec, getspec


#######################################################################
###
### Base class multiPSR : multiple Point Set Registration algorithms
###
#######################################################################

# This is a virtual class, that will be specified into "linear" and "diffeomorphic" versions of the algorithm

# Typical optimization loop  (to be called externally) :
#
#    for it in range(50):
#        print("ITERATION NUMBER ",it)
#        PSR.GMM_opt(repeat=10)
#        PSR.Reg_opt(tol=1e-5)

class multiPSR:

    ###################################################################
    # Initialization
    #
    # x = list of input point sets. Three possible formats:
    #  - x = torch tensor of size (N, D) : single point set
    #  - x[k] = torch tensor of size (N[k], D) : point set from frame k
    #  - x[k][s] = torch tensor of size (N[k,s], D) : point set in structure s from frame k
    #
    # GMMi = GMM model (and parameters) used in the algorithm. Two possible formats:
    #  - GMMi = single GMM model (for a single structure)
    #  - GMMi[s] = GMM model for structure s
    # In any case, the GMMs given as input will be *copied* in the multiPSR object

    # - dataspec : spec (dtype+device) under which all point sets are stored (see diffICP/spec.py)
    # - compspec : spec (dtype+device) under which the actual computations (GMM and registrations) will be performed
    # These two concepts are kept separate in case one wishes to use Gpu for the computations (use compspec["device"]='cuda:0')
    # but there are too many frames/point sets to store them all on the Gpu (use dataspec["device"]='cpu')
    # Of course, ideally, everything should be kept on the same device to avoid copies between devices

    def __init__(self, x, GMMi, dataspec=defspec, compspec=defspec):

        self.dataspec, self.compspec = dataspec, compspec

        ### Check input format and return various dimensions

        if isinstance(x, torch.FloatTensor) or isinstance(x, torch.cuda.FloatTensor):
            # single point set (single frame and structure)
            x = [[x]]
        elif isinstance(x, list):
            if isinstance(x[0], torch.FloatTensor) or isinstance(x[0], torch.cuda.FloatTensor):
                # multiple frames / single structure
                x = [[xk] for xk in x]
            else:
                x = [xk.copy() for xk in x]   # copy x as a list of lists (does not copy the data point sets)
        else:
            raise ValueError("Wrong format for input x")

        # Number of frames
        self.K = len(x)

        # Number of structures
        allSs = list(set([len(xk) for xk in x]))
        if len(allSs) > 1:
            raise ValueError("All frames should have same number of structures")
        self.S = allSs[0]

        # Point set dimension
        allDs = list(set([xks.shape[1] for xk in x for xks in xk]))
        if len(allDs) > 1:
            raise ValueError("All point sets should have same axis-1 dimension")
        self.D = allDs[0]

        ### Use np.arrays with dtype=object to store the point sets (allows simpler indexing than "list of list of list")
        # These point sets are stored on the device given by self.dataspec

        # This must be made carefully to avoid unwanted conversions from pytorch to numpy, see
        # https://github.com/pytorch/pytorch/issues/85606
        # https://stackoverflow.com/questions/33983053/how-to-create-a-numpy-array-of-lists

        # self.x0[k,s] : unregistered point sets
        # self.x1[k,s] : registered (warped) point sets
        # self.y[k,s] : quadratic (GMM E-step) targets for each point

        self.x0 = np.empty((self.K,self.S), dtype=object)
        self.x1 = np.empty((self.K,self.S), dtype=object)
        self.y = np.empty((self.K,self.S), dtype=object)
        for k in range(self.K):
            for s in range(self.S):
                self.x0[k,s] = x[k][s].contiguous().detach().to(**self.dataspec)
                self.x1[k,s] = self.x0[k,s].clone()
                self.y[k,s] = self.x0[k,s].clone()

        # self.N[k,s] = number of points in point set (k,s)
        self.N = np.array([[self.x0[k,s].shape[0] for s in range(self.S)] for k in range(self.K)])


        ### GMM model for inference (one per structure s)

        if GMMi.spec != compspec:
            raise ValueError("Spec (dtype+device) error : GMM 'spec' and diffPSR 'compspec' attributes should be the same")

        if isinstance(GMMi, GaussianMixtureUnif):
            self.GMMi = [copy.deepcopy(GMMi) for s in range(self.S)]
        else:
            if not isinstance(GMMi, list) or len(GMMi) != self.S:
                raise ValueError("GMMi should be a single GMM model, or a list with S GMM models")
            self.GMMi = [copy.deepcopy(gmm) for gmm in GMMi]

        for s in range(self.S):
            # all (unwarped) points associated to structure s:
            allx0s = torch.cat(tuple(self.x0[:,s]), dim=0)
            if self.GMMi[s].to_optimize["mu"]:
                # initial centroids = close to center of mass of all (unwarped) points
                self.GMMi[s].mu = allx0s.mean(dim=0) + 0.05 * allx0s.std() * torch.randn(self.GMMi[s].C, self.D, **self.dataspec)
            if self.GMMi[s].to_optimize["sigma"]:
                self.GMMi[s].sigma = 0.25 * allx0s.std()  # ad hoc



        # The FULL EM free energy writes
        #   F = \sum_{k,s} quadloss[k,s] + \sum_k regloss[k] + \sum_s Cfe[s],
        # with Cfe the "free energy offset term" associated to each GMM model's current state (see GMM.py)

        self.Cfe = [None] * self.S                              # updated after GMM_opt
        self.regloss = [0] * self.K                             # updated after Reg_opt
        self.quadloss = np.zeros((self.K,self.S))               # updated both after GMM_opt and Reg_opt
        self.last_FE = None                                     # keep a trace of previous value of free energy (for debug)

        # Store last shoot for each frame (for plotting, etc.)
        self.shoot = [None] * self.K


    # Hack to ensure a correct value of spec when Unpickling. See diffICP.spec.CPU_Unpickler and
    # https://docs.python.org/3/library/pickle.html#handling-stateful-objects
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.dataspec = defspec
        self.compspec = defspec


    ################################################################
    ### Update quadratic error between point sets x1[k,s] (warped points) and y[k,s] (GMM target points)

    def update_quadloss(self, k, s):
        self.quadloss[k,s] = ((self.x1[k,s]-self.y[k,s])**2).sum() / (2 * self.GMMi[s].sigma ** 2)


    ################################################################
    ### Recompute and store current value of free energy

    def update_FE(self, message=""):
        FE = sum(self.Cfe) + sum(self.regloss) + self.quadloss.sum().item()
        print(message.ljust(70)+f"Total free energy = {FE:.8}")
        if self.last_FE is not None and FE > self.last_FE:
            print("WARNING: measured increase in free energy ! Should not happen.")
        self.last_FE = FE


    ################################################################
    ### Partial optimization function, GMM part (for each structure s in turn)

    def GMM_opt(self, repeat=1):

        for s in range(self.S):
            allx1s = torch.cat(tuple(self.x1[:,s]), dim=0).to(**self.compspec)

            for i in range(repeat):
                # Possibly repeat several GMM loops on every iteration (since it's faster than LDDMM)
                # TODO : use convergence criterion instead of hard coded repeat number
                allys, self.Cfe[s] = self.GMMi[s].EM_step(allx1s)

            # re-assign targets to corresponding frames
            last = 0
            for k in range(self.K):
                first, last = last, last + self.N[k,s]
                self.y[k,s] = allys[first:last].to(**self.dataspec)
                #print(f"GMM check : {(self.x1[k,s]-allx1s[first:last]).norm()}")   # small_tests correct indexing
                # update quadratic losses
                self.update_quadloss(k,s)

        # nota: self.y is guaranteed to have no gradient attached (important for LDDMM QuadLoss below)

        # keep track of full free energy (to check that it only decreases!)
        self.update_FE(message = f"EM optimisation ({repeat} repetition(s)).")


    ################################################################
    ### Partial optimization, registration part (for each frame k in turn)
    # This is a virtual function, to be redefined in each derived class

    def Reg_opt(self, tol=1e-5):
        raise NotImplementedError("function Reg_opt must be written in derived classes.")



    ################################################################
    ### For convenience : return a "shoot" variable for frame k.
    # - if params is None, return the PSR model's current shoot variable for frame k : self.shoot[k]
    # - if params is not None, compute the shoot variable that *would* be associated to registration parameters "param"
    # (for LDDMM : initial momentum a0k ; for Affine : transformation parameters (M,t) as a tuple). Nota: this computation
    # remains "external", i.e., it does not affect the current state of the PSR (self.shoot, self.x1, self.a0, etc).

    def get_shoot(self, k=0, params=None):
        if params is None:
            return self.shoot[k]
        elif isinstance(self, diffPSR):
            a0k = params
            return self.LMi.Shoot(self.q0[k], a0k, self.remx0[k])
        elif isinstance(self, affinePSR):
            M,t = params
            Xk = torch.cat(tuple(self.x0[k, :]), dim=0)
            return self.AffMi.Shoot(M, t, Xk)


    ################################################################
    ### For convenience : visualization of trajectories for frame k
    # support=True only used in derived class diffPSR, to visualize trajectories of support points only
    # shoot = shoot variable containing the trajectories. By default, self.shoot[k]
    # kwargs = plotting arguments

    def plot_trajectories(self, k=0, support=False, shoot=None, **kwargs):

        # Some default plotting values
        if "alpha" not in kwargs.keys():
            kwargs["alpha"] = .5    # default transparency
        if "color" not in kwargs.keys():
            kwargs["color"] = "C" + str(k)

        # index of point set to represent inside tuple shoot (only used in class diffPSR)
        if isinstance(self, diffPSR):
            if support:
                i = 0
            else:
                i = 3
        else:
            if support:
                return      # do nothing in this case
            else:
                i = 0

        if shoot is None:
            shoot = self.shoot[k]

        x = [tup[i] for tup in shoot]
        for n in range(x[0].shape[0]):
            xnt = np.array([xt[n, :].cpu().tolist() for xt in x])
            plt.plot(xnt[:, 0], xnt[:, 1], **kwargs)

    ################################################################
    ### For convenience : compute the registration of an external point set X.
    # Y = register(X,k)
    # with X(N,D) some input points, and Y(N,D) their registered version, using parameters from frame k.
    # Remains "external", i.e., it does not affect the current state of the PSR (self.shoot, self.x1, self.a0, etc).

    def register(self, X, k=0):
        if isinstance(self, diffPSR):
            shoot = self.LMi.Shoot(self.q0[k], self.a0[k], X)
            Y = shoot[-1][3]
        elif isinstance(self, affinePSR):
            Y = X @ self.M[k].t() + self.t[k][None,:]
        return Y


#######################################################################
###
### Derived class diffPSR : multiPSR with diffeomorphic (LDDMM) registrations
###
#######################################################################

class diffPSR(multiPSR):

    ###################################################################
    # Initialization
    #
    # x = list of input point sets, as in multiPSR
    # GMMi = GMM model (and parameters) used in the algorithm, as in multiPSR
    #
    # LMi = LLDDMM model (and parameters) used in the algorithm, as provided in class LDDMMModel
    #
    # Possible decimation to derive LDDMM support points. A smaller support point set (larger Rdecim) allows to accelerate
    # the LDDMM optimization. (Note that the registration + GMM optimization still apply to the full sets of points.)
    #   - Rdecim : pick LDDMM support points out of the full point sets, through a greedy decimation algorithm,
    #     ensuring that the support points cover the full point sets with a covering radius of Rdecim*LMi.Kernel.sigma
    #   - A warning is issued if some non-support point ends up at a distance > Rcoverwarning*LMi.Kernel.sigma
    #     from all support points, at any time during the shooting procedure (unlike Rdecim which only concerns time t=0)

    def __init__(self, x, GMMi, LMi, dataspec=defspec, compspec=defspec, Rdecim=None, Rcoverwarning=None):

        # Initialize common features of the algorithm (class multiPSR)
        super().__init__(x, GMMi, dataspec=dataspec, compspec=compspec)

        # LDDMM Hamitonian system for inference. Also has a "spec" attribute, although almost useless (see kernel.py)
        if LMi.spec != compspec:
            raise ValueError("Spec (dtype+device) error : LDDMMmodel 'spec' and diffPSR 'compspec' attributes should be the same")
        self.LMi = LMi

        ### Segregate between support and non-support points for LDDMM shooting (if Rdecim is defined)
        #   supp_ids[k,s,0] = ids of support points in original point set x[k,s]
        #   supp_ids[k,s,1] = remaining ids (non-support points) in original point set x[k,s]
        # Nota : if Rdecim = None, supp_ids[k,s,0] = range(N[k,s]) and supp_ids[k,s,1] = range(0)

        self.Rdecim = Rdecim
        self.supp_ids = np.array([[[None]*2]*self.S]*self.K, dtype=object)
        if Rdecim is not None:
            for k in range(self.K):
                for s in range(self.S):
                    self.supp_ids[k,s,0], self.supp_ids[k,s,1] = decimate(self.x0[k,s], Rdecim*LMi.Kernel.sigma)
                # Report amount of decimation for each frame k (across all structures s)
                Ndecim = [ sum([ len(self.supp_ids[k,s,u]) for s in range(self.S) ]) for u in range(2) ]
                Pdecim = Ndecim[0] / (Ndecim[0]+Ndecim[1])
                print(f"Decimation, frame {k} : {Ndecim[0]} support points ({Pdecim:.0%} of original sets)")
        else:
            for k in range(self.K):
                for s in range(self.S):
                    self.supp_ids[k,s,0], self.supp_ids[k,s,1] = range(self.N[k,s]), range(0)

        self.Rcoverwarning = Rcoverwarning

        # Initial values for support points vs non-support points in each frame k (concatenated across all structures s)
        self.q0 = [None] * self.K
        self.remx0 = [None] * self.K
        for k in range(self.K):
            self.q0[k] = torch.cat(tuple( self.x0[k,s][self.supp_ids[k,s,0]] for s in range(self.S) ), dim=0
                                   ).to(**self.compspec).contiguous()
            self.remx0[k] = torch.cat(tuple( self.x0[k,s][self.supp_ids[k,s,1]] for s in range(self.S) ), dim=0
                                      ).to(**self.compspec).contiguous()

        # Initial LDDMM momenta a0[k] (nota: all structures s are concatenated in a0[k])
        # Start with zero speeds (which is NOT a0=0 in the logdet model!)
        self.a0 = [None] * self.K
        for k in range(self.K):
            v0 = torch.zeros(self.q0[k].shape, **self.compspec)
            self.a0[k] = self.LMi.v2p(self.q0[k], v0)
            # self.a0[k] = self.LMi.v2p( self.q0[k], v0, alpha=1e-3, version='ridge_keops')


    ################################################################
    ### Partial optimization, LDDMM part (for each frame k in turn)

    ## Functor returning the data loss function to use in LDDMM optimization for frame k. That is, (q,x) --> dataloss(q,x)
    # with q all support points in frame k, and x all non-support points in frame k, concatenated across all structures s

    def QuadLossFunctor(self, k):

        # quadratic targets (make a contiguous copy to optimize keops reductions)
        y_supp = torch.cat(tuple( self.y[k,s][self.supp_ids[k,s,0]] for s in range(self.S) ), dim=0
                           ).to(**self.compspec).contiguous()
        y_nonsupp = torch.cat(tuple( self.y[k,s][self.supp_ids[k,s,1]] for s in range(self.S) ), dim=0
                              ).to(**self.compspec).contiguous()
        # associated sigma values (ugly because depends on s)
        sig2_supp = torch.cat(tuple( self.GMMi[s].sigma**2 * torch.ones(len(self.supp_ids[k,s,0])) for s in range(self.S) )
                              ).to(**self.compspec).contiguous()
        sig2_nonsupp = torch.cat(tuple( self.GMMi[s].sigma**2 * torch.ones(len(self.supp_ids[k,s,1])) for s in range(self.S) )
                                 ).to(**self.compspec).contiguous()

        def dataloss_func(q,x):
            return ( (q - y_supp)**2 / (2*sig2_supp[:,None]) ).sum() + ( (x - y_nonsupp)**2 / (2*sig2_nonsupp[:,None]) ).sum()

        return dataloss_func

    ## LDDMM registration optimization function

    def Reg_opt(self, nmax=10, tol=1e-3):

        for k in range(self.K):
            ### Optimize a0[k] (the long line!)
            self.a0[k], self.shoot[k], self.regloss[k], datal, isteps, change = \
                self.LMi.Optimize(self.QuadLossFunctor(k), self.q0[k], self.a0[k], self.remx0[k], tol=tol, nmax=nmax)

            # re-assign to corresponding structures
            qk, remxk = self.shoot[k][-1][0], self.shoot[k][-1][3]
            allxk = [qk, remxk]
            #testos = [self.q0[k],self.remx0[k]]
            for u in range(2):
                last = 0
                for s in range(self.S):
                    first, last = last, last + len(self.supp_ids[k,s,u])
                    self.x1[k,s][self.supp_ids[k,s,u]] = allxk[u][first:last].to(**self.dataspec)
                    # print(f"LDDMM check 1 : {(self.x0[k,s][self.supp_ids[k,s,u]]-testos[u][first:last]).norm()}")   # small_tests correct indexing

            # update quadratic losses
            for s in range(self.S):
                self.update_quadloss(k,s)

            # print("LDDMM check 2 : ", datal, " = ", self.quadloss[k,:].sum().item())  # small_tests that quadlosses are well computed and compatible

            # Report for this frame, print full free energy (to check that it only decreases!)
            self.update_FE(message = f"Frame {k} : {isteps} optim steps, loss={self.regloss[k] + datal:.4}, change ={change:.4}.")

            # Check whether all non-support points stayed covered by support points during the shooting
            if self.Rcoverwarning is not None:
                for t in range(len(self.shoot[k])):
                    qk, remxk = self.shoot[k][t][0], self.shoot[k][t][3]
                    uncoveredxk = Vi(remxk).sqdist(Vj(qk)).min(axis=1) > (self.Rcoverwarning * self.LMi.Kernel.sigma)**2
                    if uncoveredxk.any():
                        print(f"WARNING : shooting, time step {t} : {uncoveredxk.sum()} uncovered points ({uncoveredxk.sum()/remxk.shape[0]:.2%})")
                        warnings.warn("Uncovered points during LDDMM shooting. Check Rdecim and Rcoverwarning values.", RuntimeWarning)


#######################################################################
###
### Derived class affinePSR : multiPSR with affine (viz. euclidian, rigid) registrations
###
#######################################################################
# T(X) = X * M' + t'      with
# X(N,d): input data points
# t(d,1): translation vector
# M(d,d): linear deformation matrix

class affinePSR(multiPSR):

    ###################################################################
    # Initialization
    #
    # x = list of input point sets, as in multiPSR
    # GMMi = GMM model (and parameters) used in the algorithm, as in multiPSR
    # AffMi = Affine model (and parameters) used in the algorithm, as provided in class AffineModel
    # version = 'euclidian' (rotation+translation), 'rigid' (euclidian+scaling), 'linear' (unconstrained affine)

    def __init__(self, x, GMMi, AffMi, dataspec=defspec, compspec=defspec):

        # Initialize common features of the algorithm (class multiPSR)
        super().__init__(x, GMMi, dataspec=dataspec, compspec=compspec)
        self.AffMi = AffMi
        # Affine transform applied to each frame k  (T(x) = M*x+t)
        self.M = [None] * self.K
        self.t = [None] * self.K

    ################################################################
    ### Partial optimization, registration part (for each frame k in turn)

    def Reg_opt(self, tol=1e-5):

        for k in range(self.K):
            ### Find best-fitting linear transform for frame k

            X = torch.cat(tuple(self.x0[k,:]), dim=0).to(**self.compspec)   # (N,D)
            Y = torch.cat(tuple(self.y[k,:]), dim=0).to(**self.compspec)    # (N,D). Must fit Y = X*M' + t'
            z = torch.cat(tuple( 1/(2*self.GMMi[s].sigma**2) * torch.ones(self.N[k,s]) for s in range(self.S) )
                          ).to(**self.compspec)  # ugly because depends on s

            self.M[k], self.t[k], TX, datal, self.regloss[k] = self.AffMi.Optimize(X,Y,z)

            ### Update self.x1
            last = 0
            for s in range(self.S):
                first, last = last, last + self.N[k,s]
                self.x1[k,s] = TX[first:last].to(**self.dataspec)

            ### update quadratic losses (between x1 and targets y)
            for s in range(self.S):
                self.update_quadloss(k,s)

            # print("Affine reg check : ", datal, " = ", self.quadloss[k,:].sum().item())  # small_tests that quadlosses are well computed and compatible

            # Compute a representative "shooting", as in the LDDMM case, mainly for plotting purposes.
            self.shoot[k] = self.AffMi.Shoot(self.M[k], self.t[k], X)

            ### Report for this frame, print full free energy (to check that it only decreases!)
            self.update_FE(message = f"Frame {k} : loss={self.regloss[k] + datal:.4}.")

