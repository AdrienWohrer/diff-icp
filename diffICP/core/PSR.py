'''
Classes implementing the "general" version of diffICP algorithm (and also the classic, affine version)
'''

import copy
import warnings

import numpy as np
from matplotlib import pyplot as plt
plt.ion()

import torch

#######################################################################
# Import from other files in this directory :

from diffICP.core.GMM import GaussianMixtureUnif
from diffICP.core.LDDMM import LDDMMModel
from diffICP.core.affine import AffineModel
from diffICP.core.registrations import LDDMMRegistration, AffineRegistration
from diffICP.tools.decimate import decimate
from diffICP.tools.spec import defspec
from diffICP.tools.inout import read_point_sets

from diffICP.visualization.visu import get_bounds       # (recycled for grid bounds computation)

#######################################################################
###
### Base class MultiPSR : multiple Point Set Registration algorithms
###
#######################################################################

# This is a virtual class, that will be specified into "linear" and "diffeomorphic" versions of the algorithm

# Typical optimization loop  (to be called externally) :
#
#    for it in range(50):
#        print("ITERATION NUMBER ",it)
#        PSR.GMM_opt(repeat=10)
#        PSR.Reg_opt(tol=1e-5)

class MultiPSR:

    ###################################################################

    def __init__(self, x, GMMi: GaussianMixtureUnif, dataspec=defspec, compspec=defspec):
        '''
        :param x: list of input point sets. Three possible formats:
            x = torch tensor of size (N, D) : single point set;
            x[k] = torch tensor of size (N[k], D) : point set from frame k;
            x[k][s] = torch tensor of size (N[k,s], D) : point set in structure s from frame k;

        :param GMMi: GMM model (and parameters) used in the algorithm. Two possible formats:
            GMMi = single GMM model (for a single structure);
            GMMi[s] = GMM model for structure s;
            In any case, the GMMs given as input will be *copied* in the multiPSR object;

        :param dataspec: spec (dtype+device) under which all point sets are stored (see diffICP/spec.py)

        :param compspec: spec (dtype+device) under which the actual computations (GMM and registrations) will be performed.
            These two concepts are kept separate in case one wishes to use Gpu for the computations (use compspec["device"]='cuda:0')
            but there are too many frames/point sets to store them all on the Gpu (use dataspec["device"]='cpu').
            Of course, ideally, everything should be kept on the same device to avoid copies between devices.
        '''

        self.dataspec, self.compspec = dataspec, compspec

        ### Read input point sets and various dimensions.
        #   x: point sets, now cast in the format x[k][s] ;
        #   self.K = number of frames
        #   self.S = number of structures
        #   self.D = dimension of space

        x, self.K, self.S, self.D = read_point_sets(x)

        ### Use np.arrays with dtype=object to store the point sets (allows simpler indexing than "list of list of list")
        # These point sets are stored on the device given by self.dataspec.

        # This must be made carefully to avoid unwanted conversions from pytorch to numpy, see
        # https://github.com/pytorch/pytorch/issues/85606
        # https://stackoverflow.com/questions/33983053/how-to-create-a-numpy-array-of-lists

        # self.x0[k,s] : unregistered point sets
        # self.x1[k,s] : registered (warped) point sets
        # self.y[k,s] : "quadratic targets" for each point (as produced by the GMM E step)

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
            raise ValueError("Spec (dtype+device) error : GMM 'spec' and multiPSR 'compspec' attributes should be the same")

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
        self.FE = None                                          # keep a trace of current value of free energy

        # Store last shoot for each frame (for plotting, etc.)
        self.shoot = [None] * self.K

    # Hack to ensure a correct value of spec when Unpickling. See spec.CPU_Unpickler and
    # https://docs.python.org/3/library/pickle.html#handling-stateful-objects
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.dataspec = defspec
        self.compspec = defspec


    ###################################################################
    ### More user-friendly accessors to the point sets

    def get_data_points(self, k=0, s=0):
        '''
        Data point set for frame (=patient) k and structure s
        '''
        return self.x0[k,s]

    def get_warped_data_points(self, k=0, s=0):
        '''
        Warped data point set for frame (=patient) k and structure s
        '''
        return self.x1[k,s]

    def get_template(self, s=0):
        '''
        Template (GMM model centroids) for structure s
        '''
        return self.GMMi[s].mu


    ################################################################
    ################################################################

    def update_quadloss(self, k, s):
        '''
        Update quadratic error between point sets x1[k,s] (warped points) and y[k,s] (GMM target points).
        '''

        self.quadloss[k,s] = ((self.x1[k,s]-self.y[k,s])**2).sum() / (2 * self.GMMi[s].sigma ** 2)


    ################################################################
    ################################################################

    def update_FE(self, message=""):
        '''
        Recompute and store current value of free energy.
        '''

        FE = sum(self.Cfe) + sum(self.regloss) + self.quadloss.sum().item()
        print(message.ljust(70)+f"Total free energy = {FE:.8}")
        if self.FE is not None and FE > self.FE:
            print("WARNING: measured increase in free energy ! Should not happen.")
        self.FE = FE


    ################################################################
    ################################################################

    def GMM_opt(self, max_iterations=100, tol=1e-5):
        '''
        Partial optimization function, GMM part (for each structure s in turn).
        '''

        for s in range(self.S):
            allx1s = torch.cat(tuple(self.x1[:,s]), dim=0).to(**self.compspec)

            allys, self.Cfe[s], _, i = self.GMMi[s].EM_optimization(allx1s, max_iterations=max_iterations, tol=tol)
            # for i in range(repeat):
            #     # Possibly repeat several GMM loops on every iteration (since it's faster than LDDMM)
            #     allys, self.Cfe[s] = self.GMMi[s].EM_step(allx1s)

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
            message = f"GMM optim (structure {s}) : {i} EM steps"
            if self.GMMi[s].outliers:
                p0 = 1 / (1 + np.exp(-self.GMMi[s].outliers["eta0"]))
                message += f", p_outlier={p0:.4}"
            else:
                message += "."
            self.update_FE(message = message)


    ################################################################
    ################################################################

    def Reg_opt(self, tol=1e-5):
        '''
        Partial optimization, registration part (for each frame k in turn).
        This is a virtual function, to be redefined in each derived class.
        '''

        raise NotImplementedError("function Reg_opt must be written in derived classes.")


    ################################################################
    ################################################################

    def Registration(self, k=0):
        '''
        Return a `registration` object for frame number k (k=0 by default, when there is a single frame).
        `registration` objects are convenient interfaces to apply a registration to an external set of points.
        See registrations.py.
        '''

        if isinstance(self, DiffPSR):
            return LDDMMRegistration(self.LMi, self.q0[k], self.a0[k])
        elif isinstance(self, AffinePSR):
            return AffineRegistration(self.AffMi, self.M[k], self.t[k])


    ################################################################
    ################################################################

    def plot_trajectories(self, k=0, support=False, shoot=None, **kwargs):
        '''
        Visualization of trajectories for the point sets in the registration process.
        :param k: frame to represent (k=0 by default, when there is a single frame).
        :param support: only used in class diffPSR. If true, visualize trajectories of support points.
        :param shoot: custom "shoot" variable containing the trajectories (can be used to visualize trajectories for an "external" set of points).
        :param kwargs: plotting arguments passed to pyplot.plot
        '''

        # Some default plotting values
        if "alpha" not in kwargs.keys():
            kwargs["alpha"] = .5    # default transparency
        if "color" not in kwargs.keys():
            kwargs["color"] = "C" + str(k)

        if shoot is None:
            shoot = self.shoot[k]

        if shoot is None:  # must recompute
            if isinstance(self, DiffPSR):
                if self.support_scheme:
                    shoot = self.LMi.Shoot(self.q0[k], self.a0[k], self.allx0[k])
                else:
                    shoot = self.LMi.Shoot(self.q0[k], self.a0[k])
            else:
                X = torch.cat(tuple(self.x0[k, :]), dim=0).to(**self.compspec)
                shoot = self.AffMi.Shoot(self.M[k], self.t[k], X)

        if isinstance(self, DiffPSR) and self.support_scheme and not support:
            x = [tup[-1] for tup in shoot]   # plot "non-support" points of the shooting
        else:
            x = [tup[0] for tup in shoot]   # in every other case

        for n in range(x[0].shape[0]):
            xnt = np.array([xt[n, :].cpu().tolist() for xt in x])
            plt.plot(xnt[:, 0], xnt[:, 1], **kwargs)


#######################################################################
###
### Derived class DiffPSR : MultiPSR with diffeomorphic (LDDMM) registrations
###
#######################################################################

class DiffPSR(MultiPSR):
    '''
    MultiPSR algorithm with diffeomorphic (LDDMM) registrations.
    '''

    def __init__(self, x, GMMi: GaussianMixtureUnif, LMi: LDDMMModel, dataspec=defspec, compspec=defspec):
        '''
        :param x: list of input point sets. Three possible formats:
            x = torch tensor of size (N, D) : single point set;
            x[k] = torch tensor of size (N[k], D) : point set from frame k;
            x[k][s] = torch tensor of size (N[k,s], D) : point set in structure s from frame k;

        :param LMi: LLDDMM model (and parameters) used in the algorithm, as provided in class LDDMMModel. See LDDMM.py

        :param GMMi: GMM model (and parameters) used in the algorithm. Two possible formats:
            GMMi = single GMM model (for a single structure);
            GMMi[s] = GMM model for structure s;
            In any case, the GMMs given as input will be *copied* in the multiPSR object;

        :param dataspec: spec (dtype+device) under which all point sets are stored (see diffICP/spec.py)

        :param compspec: spec (dtype+device) under which the actual computations (GMM and registrations) will be performed.
            These two concepts are kept separate in case one wishes to use Gpu for the computations (use compspec["device"]='cuda:0')
            but there are too many frames/point sets to store them all on the Gpu (use dataspec["device"]='cpu').
            Of course, ideally, everything should be kept on the same device to avoid copies between devices.
        '''

        # Initialize common features of the algorithm (class multiPSR)
        super().__init__(x, GMMi, dataspec=dataspec, compspec=compspec)

        # LDDMM Hamitonian system for inference. Also has a "spec" attribute, although almost useless (see kernel.py)
        if LMi.Kernel.spec != compspec:
            raise ValueError("Spec (dtype+device) error : LDDMMmodel kernel 'spec' and diffPSR 'compspec' attributes should be the same")
        self.LMi = LMi

        # All x0 points in frame k
        self.allx0 = [None] * self.K
        for k in range(self.K):
            self.allx0[k] = torch.cat(tuple(self.x0[k,:]), dim=0).to(**self.compspec).contiguous()

        # Initial LDDMM support points q0[k] : by defaut, all points x0 in frame k
        # NOTA : This behavior can be overridden by using function self.set_support_scheme() below
        self.support_scheme, self.rho = None, None
        self.q0 = self.allx0

        # Initial LDDMM momenta a0[k] at zero speeds (nota: all structures s are concatenated in a0[k])
        self.a0 = [None] * self.K
        self.initialize_a0()
        # self.initialize_a0(alpha=1e-3, version='ridge_keops')

    ################################################################

    def initialize_a0(self, **v2p_args):
        '''
        Set initial momenta a0 corresponding (approximately) to zero speeds, given current support points q0.
        (Note that this is NOT a0=0 if there is a logdet component.)
        '''
        for k in range(self.K):
            v0k_at_q0 = torch.zeros(self.q0[k].shape, **self.compspec)
            self.a0[k] = self.LMi.v2p(self.q0[k], v0k_at_q0, **v2p_args)

    def update_a0(self, q0_prev, a0_prev=None, **v2p_args):
        '''
        Update a0 so that the new vector field v(q0,a0) be as close as possible to v(q0_prev,a0_prev).
        (Corresponding to the projection of v(q0_prev,a0_prev) on the RKHS span of {K_z |z in q0})
        If a0_prev is None (default), use self.a0, so the update simply reflects the change of support points.
        '''
        if a0_prev is None:
            a0_prev = self.a0
        for k in range(self.K):
            v0k_at_q0 = self.LMi.v(self.q0[k], q0_prev[k], a0_prev[k])
            self.a0[k] = self.LMi.v2p(self.q0[k], v0k_at_q0, **v2p_args)

    ################################################################
    ### Fixing a smaller number of LDDMMM support points : use decimation, or a rectangular grid

    def set_support_scheme(self, scheme="decim", rho=1.0, xticks=None, yticks=None, q0=None):
        '''
        Define a specific set of support points for LDDMM shooting. A smaller support point set (larger rho)
        allows to accelerate the LDDMM optimization. (Note that the registration + GMM optimization still apply
        to the full sets of points.)

        :param rho: Relative coverage radius. The actual coverage radius is defined as
                Rcover = rho * LMi.Kernel.sigma

        :param scheme: "decim" or "grid" or "custom".
            If "decim", we pick LDDMM support points out of the full point sets, through a greedy decimation algorithm,
            ensuring that the support points cover the full point sets with a covering radius of Rcover =rho*sigma.
            If "grid", we pick LDDMM support points on a grid, with bounds given by data, and grid step size given by Rcover = rho*sigma.
            Alternatively, grid location can be set by hand, by providing lists xticks and yticks. (If provided, these
            options override the value of "rho" parameter, which becomes useless.)
            If "custom", the location of support points are provided by hand in torch tensor q0.

        :param xticks: list or 1d array. X coordinates of the support grid when scheme="grid" (overrides "rho" parameter).
        :param yticks: idem for Y coordinates of the support grid.
        :param q0: custom location of support points, when scheme = "custom"
        '''

        self.rho = rho
        Rcover = rho * self.LMi.Kernel.sigma
        self.support_scheme = scheme
        q0_prev = self.q0

        if scheme == "decim":
            # supp_ids[k,s] = ids of support points in original point set x[k,s]
            supp_ids = np.array([[None] * self.S] * self.K, dtype=object)
            self.q0 = [None] * self.K
            for k in range(self.K):
                for s in range(self.S):
                    supp_ids[k,s],_ = decimate(self.x0[k,s], Rcover)
                # Report amount of decimation for each frame k (across all structures s)
                Ndecim = sum([len(supp_ids[k,s]) for s in range(self.S)])
                Pdecim = Ndecim / sum([self.N[k,s] for s in range(self.S)])
                print(f"Decimation, frame {k} : {Ndecim} support points ({Pdecim:.0%} of original sets)")
                # And thus
                self.q0[k] = torch.cat(tuple(self.x0[k,s][supp_ids[k,s]] for s in range(self.S)), dim=0).to(**self.compspec).contiguous()

        elif scheme == "grid":
            if xticks is None or yticks is None:
                xmin,xmax,ymin,ymax = get_bounds(*self.allx0, relmargin=0.1)
            if xticks is None:
                xticks = np.arange(xmin-Rcover/2, xmax+Rcover/2, Rcover)
            if yticks is None:
                yticks = np.arange(ymin-Rcover/2, ymax+Rcover/2, Rcover)
            gridpoints = np.stack(np.meshgrid(xticks, yticks), axis=2)                          # grid points (shape (Nx,Ny,2))
            gridpoints = torch.tensor(gridpoints.reshape((-1,2),order='F'), **self.compspec).contiguous() # convert to torch tensor (shape (Nx*Ny,2))
            # Use same support points for all frames (can be useful to compare them)
            self.q0 = [gridpoints] * self.K

        elif scheme == "custom":
            assert q0 is not None, "For a custom support scheme, please specify argument q0"
            self.q0 = q0.clone().detach().to(**self.compspec).contiguous()

        else:
            raise ValueError(f"Unknown value of support point scheme : {scheme}. Only values available are 'decim', 'grid' and 'custom'.")

        # Don't forget to update a0 in consequence
        self.update_a0(q0_prev, rcond=1e-1)

    ################################################################
    ################################################################

    def QuadLossFunctor(self, k):
        '''
        Partial optimization, LDDMM part (for frame k).
        Nota: only works for an LDDMM Gaussian kernel.

        :return : the data loss function to use in LDDMM optimization for frame k. That is, (q,x) --> dataloss(q,x)
            with q all support points in frame k, and x all data points in frame k, concatenated across all structures s.
        '''

        # quadratic targets (make a contiguous copy to optimize keops reductions)
        y = torch.cat(tuple(self.y[k,:]), dim=0).to(**self.compspec).contiguous()

        # associated sigma values (ugly because depends on s)
        sig2 = torch.cat(tuple(self.GMMi[s].sigma**2 * torch.ones(self.N[k,s]) for s in range(self.S))).to(**self.compspec).contiguous()

        if self.support_scheme is None:
            # (default) dense scheme : support_points q = data_points x
            def dataloss_func(q,x):
                return ( (q - y)**2 / (2*sig2[:,None]) ).sum()

        else:
            # other support scheme : support_points q != data_points x
            def dataloss_func(q,x):
                return ( (x - y)**2 / (2*sig2[:,None]) ).sum()

        return dataloss_func

    ################################################################
    ################################################################

    def Reg_opt(self, nmax=10, tol=1e-3):
        '''
        LDDMM registration optimization function.
        :param nmax : max number of iterations.
        :param tol : relative tolerance for stopping (before nmax).
        '''

        for k in range(self.K):
            ### Optimize a0[k] (the long line!)

            if self.support_scheme is None:
                # (default) dense scheme : support_points q = data_points x
                self.a0[k], self.shoot[k], self.regloss[k], datal, isteps, change = \
                    self.LMi.Optimize(self.QuadLossFunctor(k), self.q0[k], self.a0[k], tol=tol, nmax=nmax)
                # Recover warped data points
                allx1k = self.shoot[k][-1][0]

            else:
                # other support scheme : support_points q != data_points x
                self.a0[k], self.shoot[k], self.regloss[k], datal, isteps, change = \
                    self.LMi.Optimize(self.QuadLossFunctor(k), self.q0[k], self.a0[k], self.allx0[k], tol=tol, nmax=nmax)
                # Recover warped data points
                allx1k = self.shoot[k][-1][3]

            # Re-assign to corresponding structures
            last = 0
            for s in range(self.S):
                first, last = last, last + self.N[k,s]
                self.x1[k,s] = allx1k[first:last].to(**self.dataspec)

            # update quadratic losses
            for s in range(self.S):
                self.update_quadloss(k,s)

            # Check whether all data points stayed covered by support points during the shooting.
            # A warning is issued if some warped data points end up at a distance > Rcoverwarning*LMi.Kernel.sigma
            # from all support points, at any time during the shooting procedure (unlike "rho" which only concerns time t=0).

            if self.support_scheme is not None:
                Rcoverwarning = 2.0                       # (hard-modify here if necessary)
                for t in range(len(self.shoot[k])):
                    qk, xk = self.shoot[k][t][0], self.shoot[k][t][3]
                    uncoveredxk = self.LMi.Kernel.check_coverage(xk, qk, Rcoverwarning)
                    if uncoveredxk.any():
                        print(f"WARNING : shooting, time step {t} : {uncoveredxk.sum()} uncovered points ({uncoveredxk.sum()/xk.shape[0]:.2%})")
                        warnings.warn("Uncovered points during LDDMM shooting. Choose a smaller rho when defining the support scheme.", RuntimeWarning)

            # Report for this frame, print full free energy (to check that it only decreases!)
            self.update_FE(message = f"Frame {k} : {isteps} optim steps, loss={self.regloss[k] + datal:.4}, change ={change:.4}.")


#######################################################################
###
### Derived class AffinePSR : MultiPSR with affine (viz. euclidian, rigid) registrations
###
#######################################################################

class AffinePSR(MultiPSR):
    '''
    multiPSR algorithm with affine (viz. euclidian, rigid) registrations. That is,
        T(X) = X * M' + t'      with

        X(N,d): input data points ;
        t(d,1): translation vector ;
        M(d,d): linear deformation matrix ;
    '''

    def __init__(self, x, GMMi: GaussianMixtureUnif, AffMi: AffineModel, dataspec=defspec, compspec=defspec):
        '''
        :param x: list of input point sets. Three possible formats:
            x = torch tensor of size (N, D) : single point set;
            x[k] = torch tensor of size (N[k], D) : point set from frame k;
            x[k][s] = torch tensor of size (N[k,s], D) : point set in structure s from frame k;

        :param GMMi: GMM model (and parameters) used in the algorithm. Two possible formats:
            GMMi = single GMM model (for a single structure);
            GMMi[s] = GMM model for structure s;
            In any case, the GMMs given as input will be *copied* in the multiPSR object;

        :param AffMi: Affine model (and parameters) used in the algorithm, as provided in class AffineModel.
            In particular, version = 'euclidian' (rotation+translation), 'rigid' (euclidian+scaling), 'linear' (unconstrained affine).

        :param dataspec: spec (dtype+device) under which all point sets are stored (see diffICP/spec.py)

        :param compspec: spec (dtype+device) under which the actual computations (GMM and registrations) will be performed.
            These two concepts are kept separate in case one wishes to use Gpu for the computations (use compspec["device"]='cuda:0')
            but there are too many frames/point sets to store them all on the Gpu (use dataspec["device"]='cpu').
            Of course, ideally, everything should be kept on the same device to avoid copies between devices.
        '''

        # Initialize common features of the algorithm (class multiPSR)
        super().__init__(x, GMMi, dataspec=dataspec, compspec=compspec)
        self.AffMi = AffMi
        # Affine transform applied to each frame k  (T(x) = M*x+t)
        self.M = [torch.eye(self.D, **dataspec)] * self.K
        self.t = [torch.zeros(self.D, **dataspec)] * self.K

    ################################################################
    ################################################################

    def Reg_opt(self):
        '''
        Affine registration optimization function.
        '''

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

