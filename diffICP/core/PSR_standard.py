'''
Diffeomorphic registration of point sets viewed as point distributions.
(classic articles of Glaunès et al 2004, Durrleman 09, etc.).
Basically this is a reimplementation of the point set registration algorithm present in the Deformetrica software.
'''

import warnings

import numpy as np
from matplotlib import pyplot as plt
plt.ion()

import torch

from diffICP.tools.optim import LBFGS_optimization

#######################################################################
# Import from other files in this directory :

from diffICP.core.GMM import GaussianMixtureUnif
from diffICP.core.LDDMM import LDDMMModel
from diffICP.core.affine import AffineModel
from diffICP.core.registrations import LDDMMRegistration, AffineRegistration

from diffICP.tools.decimate import decimate
from diffICP.tools.spec import defspec
from diffICP.tools.kernel import GenKernel
from diffICP.visualization.visu import get_bounds       # (recycled for grid bounds computation)
from diffICP.tools.inout import read_point_sets


#######################################################################
### A useful function : distance in the RKHS data space of point clouds viewed as signed measures

def data_distance(Kernel:GenKernel, x, y, w=None):
    '''
    Compute RKHS distance between two (weighted) point sets:
        - x, data point set.
        - y, template point set, optionnally with associated scalar weights w.

    Setting X <-- (y,x) [totality of the points, template + target data], we have
        L = \sum_i \sum_j c_i c_j K(X_i,X_j) ;
    where
        - c_i=1/Nx if X_i belongs to the target data set (x)
        - c_i=-1/Ny if X_i belongs to the template set (y) [or c_i = -w_i, if template weights w are used]
    '''
    KB = Kernel.KBase
    Nx = x.shape[0]
    Ny = y.shape[0]

    if w is None:
        L = KB(x, x).sum() / Nx ** 2 + KB(y, y).sum() / Ny ** 2 - 2 * KB(y, x).sum() / (Nx * Ny)
    else:
        KRS = Kernel.KRedScal
        L = KB(x, x).sum() / Nx ** 2 + (KRS(y, y, w).flatten() * w).sum() - 2 * (KB(y, x).flatten() * w).sum() / Nx
    return L


#######################################################################
###
### Base class MultiPSR_std : standard (multiple) Point Set Registration algorithms
###
#######################################################################

class MultiPSR_std:

    ###################################################################

    def __init__(self, x, y_template, noise_std, DataKernel:GenKernel, template_weights=False, dataspec=defspec, compspec=defspec):
        '''
        :param x: list of data point sets (fixed). Three possible formats:
            x = torch tensor of size (N, D) : single point set;
            x[k] = torch tensor of size (N[k], D) : point set from frame k;
            x[k][s] = torch tensor of size (N[k,s], D) : point set in structure s from frame k;

        :param y_template: template point set (the one that will be deformed). Two possible formats:
            y_template = single point set (for a single structure);
            y_template[s] = point set for structure s (for multiple structures);
            In any case, the point sets given as input will be *copied* in the multiPSR_std object;

        :param noise_std: reference value for dataloss, for each structure s:
            noise_std = single reference value (for a single structure);
            noise_std[s] = reference value for structure s (for multiple structures);
            The smaller noise_std, the more exact is required the match between point sets (at the cost of more deformation).
            Globally, noise_std plays the same role as sqrt(lambda) [LDDMM regularization constant] in my own framework;

        :param DataKernel: RKHS Kernel for the Data points (dedicated class);

        :param template_weights: set at True to associate each template point to a different scalar weight;

        :param dataspec: spec (dtype+device) under which all point sets are stored (see diffICP/spec.py);

        :param compspec: spec (dtype+device) under which the actual computations (GMM and registrations) will be performed.
            These two concepts are kept separate in case one wishes to use Gpu for the computations (use compspec["device"]='cuda:0')
            but there are too many frames/point sets to store them all on the Gpu (use dataspec["device"]='cpu').
            Of course, ideally, everything should be kept on the same device to avoid copies between devices.
        '''

        self.dataspec, self.compspec = dataspec, compspec
        self.DataKernel = DataKernel

        ### Read input point sets and various dimensions.
        #   x: point sets, now cast in the format x[k][s] ;
        #   self.K = number of frames
        #   self.S = number of structures
        #   self.D = dimension of space

        x, self.K, self.S, self.D = read_point_sets(x)

        ### Template point set
        # self.y0[s] : unregistered template point set (one per structure s)

        if isinstance(y_template, torch.Tensor):
            self.y0 = [y_template.clone().contiguous().detach().to(**self.dataspec) for s in range(self.S)]
        else:
            if not isinstance(y_template, list) or len(y_template) != self.S:
                raise ValueError("y_template should be a single point set (torch tensor), or a list with S point sets")
            self.y0 = [y_tmp.clone().contiguous().detach().to(**self.dataspec) for y_tmp in y_template]

        ### Noise_std parameter of each structure s
        # self.noise_std[s]

        if not isinstance(noise_std, list):
            self.noise_std = [noise_std] * self.S
        else:
            self.noise_std = noise_std
        assert len(self.noise_std) == self.S

        ### Use np.arrays with dtype=object to store the point sets (allows simpler indexing than "list of list of list")
        # These point sets are stored on the device given by self.dataspec.
        # Careful : https://github.com/pytorch/pytorch/issues/85606

        # self.x[k,s] : targets (input data point sets)
        # self.y1[k,s] : registered (warped) template point sets

        self.y1 = np.empty((self.K,self.S), dtype=object)
        self.x = np.empty((self.K,self.S), dtype=object)
        for s in range(self.S):
            for k in range(self.K):
                self.x[k,s] = x[k][s].contiguous().detach().to(**self.dataspec)
                self.y1[k,s] = self.y0[s].clone().contiguous().detach().to(**self.dataspec)
                
        # self.Nx[k,s] = number of points in data point set x[k,s]
        self.Nx = np.array([[self.x[k,s].shape[0] for s in range(self.S)] for k in range(self.K)])
        # self.Ny[s] = number of points in template point set y[s]
        self.Ny = np.array([self.y0[s].shape[0] for s in range(self.S)])

        ### EXPERIMENTAL ! Associate each point in y0 to a different WEIGHT. (Viewing y0 as a distribution)
        self.template_weights = template_weights
        if self.template_weights:
            self.w0 = [ torch.ones(self.Ny[s])/self.Ny[s] for s in range(self.S) ]
        else:
            self.w0 = [None] * self.S

        # Store last shoot for each frame (for plotting, etc.)
        self.shoot = [None] * self.K

        # The FULL optimization energy writes
        #   E = \sum_{k,s} dataloss[k,s] + \sum_k regloss[k]

        self.regloss = [0] * self.K                         # updated after Reg_opt
        self.dataloss = np.zeros((self.K,self.S))           # updated both after GMM_opt and Reg_opt
        # Compute energy values at initialization
        for k in range(self.K):
            for s in range(self.S):
                self.dataloss[k,s] = data_distance(self.DataKernel, self.x[k,s], self.y0[s], self.w0[s]) / self.noise_std[s]**2
        self.E = sum(self.regloss) + self.dataloss.sum().item()

    ###################################################################
    # Hack to ensure a correct value of spec when Unpickling. See diffICP.spec.CPU_Unpickler and
    # https://docs.python.org/3/library/pickle.html#handling-stateful-objects

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.dataspec = defspec
        self.compspec = defspec

    ###################################################################
    ### More user-friendly accessors to the point sets

    def get_template(self, s=0):
        '''
        Template point set for structure s.
        '''
        return self.y0[s]

    def get_warped_template(self, k=0, s=0):
        '''
        Warped template point set for frame (=patient) k and structure s
        '''
        return self.y1[k,s]

    def get_data_points(self, k=0, s=0):
        '''
        Data point set for frame (=patient) k and structure s
        '''
        return self.x[k,s]

    ################################################################

    def Registration(self, k=0):
        '''
        Return a `registration` object for frame number k (k=0 by default, when there is a single frame).
        `registration` objects are convenient interfaces to apply a registration to an external set of points.
        See registrations.py.
        '''

        if isinstance(self, DiffPSR_std):
            return LDDMMRegistration(self.LMi, self.q0, self.a0[k])
        elif isinstance(self, AffinePSR_std):
            return AffineRegistration(self.AffMi, self.M[k], self.t[k])


    ###############################################################################

    def Template_opt(self, nmax=10, tol=1e-3, errthresh=1e8):
        '''
        Partial optimization function, template part (for each structure s in turn).
        '''

        # Optimize each template y0[s] independently
        for s in range(self.S):

            def lossfunc(y0s, ws):
                L = torch.Tensor([0.0])
                for k in range(self.K):
                    L += data_distance(self.DataKernel, self.x[k,s], self.Registration(k).apply(y0s), ws)
                return L

            if self.template_weights:
                # Optimize lossfunc wrt. y0s and w0s (experimental)
                p, L, nsteps, change = LBFGS_optimization([self.y0[s],self.w0[s]], lossfunc, nmax=nmax, tol=tol, errthresh=errthresh)
                self.y0[s] = p[0]
                self.w0[s] = p[1]
            else:
                # Optimize lossfunc wrt. y0s (default)
                p, L, nsteps, change = LBFGS_optimization([self.y0[s]], lambda y0s: lossfunc(y0s,None), nmax=nmax, tol=tol, errthresh=errthresh)
                self.y0[s] = p[0]

            # Update variables
            self.update_state(s=s, caller=self.Template_opt)
            # Print energy (to check that it only decreases)
            print(f"Template {s} : {nsteps} optim steps, loss={L:.4}, change={change:.4}.".ljust(70)
                  + f"Total energy = {self.E:.8}")

    ################################################################

    def Reg_opt(self, tol=1e-5):
        '''
        Partial optimization, registration part (for each frame k in turn).
        This is a virtual function, to be redefined in each derived class.
        '''

        raise NotImplementedError("function Reg_opt must be written in derived classes.")

    ###############################################################

    def update_state(self, k=None, s=None, caller=None):
        '''
        Update various dependent variables, e.g., after an optimization step.

        :param k: some k index, or None (-> loop over all k=1...K)
        :param s: some s index, or None (-> loop over all s=1...S)
        :param caller: function calling update_state() : self.Template_opt, self.Reg_opt, or None (some other caller)
        '''
        if k is None:
            klist = range(self.K)
        else:
            klist = [k]
        if s is None:
            slist = range(self.S)
        else:
            slist = [s]

        ### Recompute warped templates and dataloss
        for k in klist:
            for s in slist:
                if isinstance(self, DiffPSR_std) and caller != self.Reg_opt :   # (self.Reg_opt already does it)
                    self.y1[k,s] = self.Registration(k).apply(self.y0[s]).detach()
                self.dataloss[k,s] = data_distance(self.DataKernel, self.x[k,s], self.y1[k,s], self.w0[s]) / self.noise_std[s]**2

        ### In case of an external, punctual call, (re)compute regloss (as self.Reg_opt may not have been called yet)
        if caller is None and isinstance(self, DiffPSR_std):
            for k in klist:
                self.regloss[k] = self.LMi.trajloss(self.Registration(k).shoot(None))

        ### If caller=Template_opt : update ally0, LDDMM support points q0, and momenta a0
        if isinstance(self, DiffPSR_std) and caller == self.Template_opt :
            self.ally0 = torch.cat(tuple(self.y0), dim=0).clone().to(**self.compspec).detach().contiguous()  # safer to clone ?
            q0_prev = self.q0                           # previous support points
            if self.support_scheme is None:
                self.q0 = self.ally0                    # new  support points
                self.update_a0(q0_prev, rcond=1e-1)     # Update a0 to reflect the change in support points
            elif self.support_scheme=='decim':
                # Recompute support points with a new decimation ! TODO Warning : not tested at all
                self.set_support_scheme("decim", self.rho)

        ### Recompute and store current value of optimization energy
        E = sum(self.regloss) + self.dataloss.sum().item()
        if self.E is not None and E > self.E:
            warnings.warn("WARNING: measured increase in optimization energy ! Should not happen.")
            print("WARNING: measured increase in optimization energy ! Should not happen.")
        self.E = E

    ################################################################
    ################################################################

    def plot_trajectories(self, k=0, support=False, shoot=None, **kwargs):
        '''
        Visualization of trajectories for the point sets in the registration process.
        :param k: frame to represent (k=0 by default, when there is a single frame).
        :param support: only used in class diffPSR. If true, visualize trajectories of support points only.
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
            if isinstance(self, DiffPSR_std):
                if self.support_scheme:
                    shoot = self.LMi.Shoot(self.q0, self.a0[k], self.ally0)
                else:
                    shoot = self.LMi.Shoot(self.q0, self.a0[k])
            else:
                shoot = self.AffMi.Shoot(self.M[k], self.t[k], self.ally0)   # NOT TESTED

        if isinstance(self, DiffPSR_std) and self.support_scheme and not support:
            y = [tup[-1] for tup in shoot]   # plot "non-support" points of the shooting
        else:
            y = [tup[0] for tup in shoot]   # in every other case

        for n in range(y[0].shape[0]):
            ynt = np.array([yt[n, :].cpu().tolist() for yt in y])
            plt.plot(ynt[:, 0], ynt[:, 1], **kwargs)


#######################################################################
###
### Derived class DiffPSR_std : multiPSR_std with diffeomorphic (LDDMM) registrations
###
#######################################################################

class DiffPSR_std(MultiPSR_std):
    '''
    standard multiPSR algorithm with diffeomorphic (LDDMM) registrations.
    '''

    def __init__(self, x, y_template, noise_std, LMi:LDDMMModel, DataKernel:GenKernel, template_weights=False, dataspec=defspec, compspec=defspec):
        '''
        :param x: list of data point sets (fixed). Three possible formats:
            x = torch tensor of size (N, D) : single point set;
            x[k] = torch tensor of size (N[k], D) : point set from frame k;
            x[k][s] = torch tensor of size (N[k,s], D) : point set in structure s from frame k;

        :param y_template: template point set (the one that will be deformed). Two possible formats:
            y_template = single point set (for a single structure);
            y_template[s] = point set for structure s (for multiple structures);
            In any case, the point sets given as input will be *copied* in the multiPSR_std object;

        :param noise_std: reference value for dataloss, for each structure s:
            noise_std = single reference value (for a single structure);
            noise_std[s] = reference value for structure s (for multiple structures);
            The smaller noise_std, the more exact is required the match between point sets (at the cost of more deformation).
            Globally, noise_std plays the same role as sqrt(lambda) [LDDMM regularization constant] in my own framework;

        :param LMi: LLDDMM model (and parameters) used in the algorithm, as provided in class LDDMMModel. See LDDMM_logdet.py

        :param DataKernel: RKHS Kernel for the Data points (dedicated class);

        :param template_weights: set at True to associate each template point to a different scalar weight;

        :param dataspec: spec (dtype+device) under which all point sets are stored (see diffICP/spec.py);

        :param compspec: spec (dtype+device) under which the actual computations (GMM and registrations) will be performed.
            These two concepts are kept separate in case one wishes to use Gpu for the computations (use compspec["device"]='cuda:0')
            but there are too many frames/point sets to store them all on the Gpu (use dataspec["device"]='cpu').
            Of course, ideally, everything should be kept on the same device to avoid copies between devices.
        '''

        # Initialize common features of the algorithm (class MultiPSR_std)
        super().__init__(x, y_template, noise_std, DataKernel=DataKernel, template_weights=template_weights, dataspec=dataspec, compspec=compspec)

        # LDDMM Hamitonian system for inference. Also has a "spec" attribute, although almost useless (see kernel.py)
        if LMi.Kernel.spec != compspec:
            raise ValueError("Spec (dtype+device) error : LDDMMmodel 'spec' and diffPSR 'compspec' attributes should be the same")
        self.LMi = LMi

        # All y0 points (across all structures s)
        self.ally0 = torch.cat(tuple(self.y0), dim=0).clone().to(**self.compspec).detach().contiguous()      # safer to clone ?

        # Initial LDDMM support points q0 : by defaut, all points y0
        # NOTA : This behavior can be overridden by using function self.set_support_scheme() (TODO write!)
        self.support_scheme = None
        self.q0 = self.ally0

        # Initial LDDMM momenta a0[k] (nota: all structures s are concatenated in a0[k])
        # Start with zero speeds (which is NOT a0=0 in the logdet model!)
        self.a0 = [None] * self.K
        self.initialize_a0()
        # self.initialize_a0(alpha=1e-3, version='ridge_keops')

    ################################################################

    def initialize_a0(self, **v2p_args):
        '''
        Set initial momenta a0 corresponding (approximately) to zero speeds.
        '''
        for k in range(self.K):
            v0k_at_q0 = torch.zeros(self.q0.shape, **self.compspec)
            self.a0[k] = self.LMi.v2p(self.q0, v0k_at_q0, **v2p_args)

    def update_a0(self, q0_prev, a0_prev=None, **v2p_args):
        '''
        Update a0 so that the new vector field v(q0,a0) be as close as possible to v(q0_prev,a0_prev).
        (Corresponding to the projection of v(q0_prev,a0_prev) on the RKHS span of {K_z |z in q0})
        If a0_prev is None (default), use self.a0, so the update simply reflects the change of support points.
        '''
        if a0_prev is None:
            a0_prev = self.a0
        for k in range(self.K):
            v0k_at_q0 = self.LMi.v(self.q0, q0_prev, a0_prev[k])
            self.a0[k] = self.LMi.v2p(self.q0, v0k_at_q0, **v2p_args)

    ################################################################
    ### Fixing a smaller number of LDDMMM support points : use decimation, or a rectangular grid

    def set_support_scheme(self, scheme="decim", rho=1.0, xticks=None, yticks=None, q0=None):
        '''
        Define a specific set of support points for LDDMM shooting. A smaller support point set (larger rho)
        allows to accelerate the LDDMM optimization. (Note that the registration + template optimization still apply
        to the full sets of points.)

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
            # supp_ids[s] = ids of support points in original point set y0[s]
            # NOT TESTED !
            supp_ids = [None] * self.S
            for s in range(self.S):
                supp_ids[s],_ = decimate(self.y0[s], Rcover)
            # Report amount of decimation (across all structures s)
            Ndecim = sum([len(supp_ids[s]) for s in range(self.S)])
            Pdecim = Ndecim / sum(self.Ny)
            print(f"Decimation : {Ndecim} support points ({Pdecim:.0%} of original sets)")
            # And thus
            self.q0 = torch.cat(tuple(self.y0[s][supp_ids[s]] for s in range(self.S)), dim=0).to(**self.compspec).contiguous()

        elif scheme == "grid":
            if xticks is None or yticks is None:
                xmin,xmax,ymin,ymax = get_bounds(*self.y0, relmargin=0.1)
            if xticks is None:
                xticks = np.arange(xmin-Rcover/2, xmax+Rcover/2, Rcover)
            if yticks is None:
                yticks = np.arange(ymin-Rcover/2, ymax+Rcover/2, Rcover)
            gridpoints = np.stack(np.meshgrid(xticks, yticks), axis=2)                          # grid points (shape (Nx,Ny,2))
            gridpoints = torch.tensor(gridpoints.reshape((-1,2),order='F'), **self.compspec).contiguous() # convert to torch tensor (shape (Nx*Ny,2))
            self.q0 = gridpoints

        elif scheme == "custom":
            assert q0 is not None, "For a custom support scheme, please specify argument q0"
            self.q0 = q0.clone().detach().to(**self.compspec).contiguous()

        else:
            raise ValueError(f"Unknown value of support point scheme : {scheme}. Only values available are 'decim', 'grid' and 'custom'.")

        # Don't forget to update a0 in consequence
        self.update_a0(q0_prev, rcond=1e-2)

    ################################################################
    ################################################################

    def Reg_opt(self, nmax=10, tol=1e-3):
        '''
        LDDMM registration optimization function.
        :param nmax : max number of iterations.
        :param tol : relative tolerance for stopping (before nmax).
        '''

        for k in range(self.K):

            # Dataloss function to use in the optimization.
            # y = concatenation of (warped) template points, across all structures s
            def dataloss_func(y):
                L = torch.tensor([0.0])
                last = 0
                for s in range(self.S):
                    first, last = last, last + self.Ny[s]
                    L += data_distance(self.DataKernel, self.x[k,s], y[first:last], self.w0[s]) / self.noise_std[s] ** 2
                return L

            ### Optimize a0[k] (the long line!)
            if self.support_scheme is None:
                # (default) dense scheme : support_points q = template points y
                self.a0[k], self.shoot[k], self.regloss[k], datal, isteps, change = \
                    self.LMi.Optimize(lambda q,y: dataloss_func(q), self.q0, self.a0[k], tol=tol, nmax=nmax)
                # Recover warped template points
                ally1k = self.shoot[k][-1][0]

            else:
                # other support scheme : support_points q != template points y
                self.a0[k], self.shoot[k], self.regloss[k], datal, isteps, change = \
                    self.LMi.Optimize(lambda q,y: dataloss_func(y), self.q0, self.a0[k], self.ally0, tol=tol, nmax=nmax)
                # Recover warped template points
                ally1k = self.shoot[k][-1][-1]

            # Re-assign to corresponding structures
            last = 0
            for s in range(self.S):
                first, last = last, last + self.Ny[s]
                self.y1[k,s] = ally1k[first:last].to(**self.dataspec)   # (not checked)

            # Check whether all data points stayed covered by support points during the shooting.
            # A warning is issued if some warped data points end up at a distance > Rcoverwarning*LMi.Kernel.sigma
            # from all support points, at any time during the shooting procedure (unlike "rho" which only concerns time t=0).

            if self.support_scheme is not None:
                Rcoverwarning = 2.0                           # (hard-modify here if necessary)
                for t in range(len(self.shoot[k])):
                    qk, yk = self.shoot[k][t][0], self.shoot[k][t][-1]
                    uncoveredyk = self.LMi.Kernel.check_coverage(yk, qk, Rcoverwarning)
                    if uncoveredyk.any():
                        print(f"WARNING : shooting, time step {t} : {uncoveredyk.sum()} uncovered points ({uncoveredyk.sum()/yk.shape[0]:.2%})")
                        warnings.warn("Uncovered points during LDDMM shooting. Choose a smaller rho when defining the support scheme.", RuntimeWarning)

            # Update variables and print energy (to check that it only decreases!)
            self.update_state(k=k, caller=self.Reg_opt)
            # self.update_energy(message = f"Frame {k} : {isteps} optim steps, loss={self.regloss[k] + datal:.4}, change ={change:.4}.")
            print(f"Frame {k} : {isteps} optim steps, loss={self.regloss[k] + datal:.4}, change={change:.4}.".ljust(70)
                  + f"Total energy = {self.E:.8}")


#######################################################################
###
### Derived class AffinePSR : multiPSR with affine (viz. euclidian, rigid) registrations
###
#######################################################################

# TODO !!!!


class AffinePSR_std(MultiPSR_std):
    '''
    multiPSR_std algorithm with affine (viz. euclidian, rigid) registrations. That is,
        T(X) = X * M' + t'      with

        X(N,d): input data points ;
        t(d,1): translation vector ;
        M(d,d): linear deformation matrix ;
    '''

    def __init__(self, x, GMMi: GaussianMixtureUnif, AffMi: AffineModel, dataspec=defspec, compspec=defspec):

        raise NotImplementedError("Affine version of 'standard' PSR algorithm not implemented yet")

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

    def Reg_opt(self, tol=1e-5):
        '''
        Affine registration optimization function.
        :param tol : relative tolerance for stopping (before nmax).
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

