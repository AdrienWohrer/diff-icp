'''
Gaussian Mixture Model functionalities. Adaptation of the GaussianMixture presented in KeOps tutorials, with the following main modifications:

- only uniform isotropic covariances sigma^2*Id

- added functions for EM fitting of the model to a set of data
'''

# Standard imports

import copy, math, time
import importlib.util
import warnings

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.cm as cm

import torch
from torch.nn import Module
from torch.nn.functional import softmax, log_softmax

# Look for keops and use it if possible
use_keops = importlib.util.find_spec("pykeops") is not None
if use_keops:
    from pykeops.torch import Vi, Vj
else:
    print("Warning: pykeops not installed. Consider installing it, if you are on linux")

from diffICP.tools.spec import defspec
from diffICP.visualization.visu import get_bounds, my_scatter


#####################################################################################
#####################################################################################
#####################################################################################

class GaussianMixtureUnif(Module):

    def __init__(self, mu, use_outliers=False, spec=defspec, computversion="keops"):
        '''
        Gaussian Mixture Model with uniform isotropic covariances sigma^2*Id

        Implementation is based on log scores, log odd-ratios, etc., which are more stable numerically.

        Component priors are encoded by the SCORES w_c such that pi_c = exp(w_c) / sum_{c'} exp(w_{c'}). Thus (in Pytorch),

        - pi = softmax(w)

        - logpi = w - w.logsumexp()

        Similarly, EM responsibitilies are encoded by their logarithm, lgamma_nc = log(gamma_nc).

        The Outlier component, if present, is encoded with the following scheme :

        - The overall balance of outlier/non outlier if given by a LOG-ODDS-RATIO eta0

        - Component scores w_c above refer to "GMM only" scores, thus for example the overall weight of component c when counting outliers is pi_c / (1+exp(eta0))

        - Similarly, the balance of outlier/non outlier responsibility on a given sample n is given by a LOG-ODDS-RATIO eta0n.

        - Variables lgamma_nc above refer to "GMM only" responsibilities ; the overall responsibility of component c on sample n when counting outliers is gamma_nc / (1+exp(eta0n))

        This is the basic initialization function. Many other parameters (self.to_optimize, self.outliers, etc.) can be set afterwards, see comments in the code.

        :param mu:  initial emplacement of centroids (mu.shape[0] thus provides the -fixed- number of Gaussian components)
        :param use_outliers: set at True to add handling of outliers. (Can also be done afterwards, by modifying self.outliers.)
        :param spec:  dictionary (dtype and device) where the GMM model operates (see diffICP.spec)
        :param computversion: "keops" or "torch"
        '''

        super(GaussianMixtureUnif, self).__init__()			# https://rhettinger.wordpress.com/2011/05/26/super-considered-super/

        self.params = {}
        self.spec = spec
        self.mu = mu.clone().detach().to(**spec)            # Just to be sure, copy and detach from any computational graph
        self.C, self.D = self.mu.shape
        r = self.mu.var(0).sum().sqrt().item()              # "typical radius" of the cloud of centroids. Hence, each centroid takes a "typical volume" r^D/C
        self.sigma = 0.1 * (r / self.C**(1/self.D))         # 0.1 times the "typical radius" of each centroid
        self.w = torch.zeros(self.C, **spec)                # w_c = "score" de chaque composante. Son poids est pi_c = exp(w_c) / sum_{c'} exp(w_{c'})
        self.to_optimize = {                    # (Can be modified afterwards, externally, if required)
            "sigma" : True,
            "mu" : True,
            "w" : True,
            "eta0" : True                       # Optimize outlier log-odds-ratio, if an outlier scheme is used
        }
        if use_outliers:
            self.outliers = {                   # (Can also be modified afterwards, externally, if required)
                "vol0": None,   # Reference volume of the outlier distribution (if 'None', fixed automatically at first call to E_step)
                "eta0": 0.0     # (Initial) value of log-odds-ratio for "outlier vs. GMM"
            }
        else:
            self.outliers = None                # Model with no outlier component

        self.set_computversion(computversion)

    # ----------------------------------------
    def __deepcopy__(self, memo):
        '''
        Custom deepcopy = copy GMM parameters.
        '''
        G2 = GaussianMixtureUnif(self.mu, spec=self.spec, computversion=self.computversion)
        G2.sigma = self.sigma
        G2.w = self.w.clone().detach()
        G2.to_optimize = copy.deepcopy(self.to_optimize)
        G2.outliers = copy.deepcopy(self.outliers)
        return G2

    # ---------------------------------------

    def set_computversion(self, version):
        '''
        Set computation version ('keops' or 'pytorch').
        '''
        if version == "keops" and not use_keops:
            warnings.warn(
                "Asked for keops kernel, but keops is not available on this machine. Switching to torch kernel.")
            version = "torch"
        # Aliases for the reductions (keops or torch version) :
        if version == 'keops':
            # KeOps versions : work even for large datasets
            self.EM_step = self.EM_step_keops
        elif version == "torch":
            # PyTorch versions : faster on CPU + small datasets
            self.EM_step = self.EM_step_torch
        else:
            raise ValueError(f"unkown computversion : {version}. Choices are 'keops' or 'torch'")
        self.computversion = version

    # ---------------------------------------

    def set_vol0(self, X: torch.Tensor):
        '''
        Set a "reference volume" for outlier distribution, from the bounding box of given data points X.
        :param X: torch.tensor of size (N,D) : data points
        :return: None
        '''
        if self.outliers is not None:
            self.outliers["vol0"] = (X.max(dim=0)[0]-X.min(dim=0)[0]).prod().item()

    # ---------------------------------------

    def __str__(self):
        '''
        Printable summary string of the GMM's parameters.
        '''
        s = super(GaussianMixtureUnif, self).__str__()
        s+= ": Gaussian Mixture with Uniform covariances. Parameters:\n"
        s+= "    C [# components] : "+str(self.C)+"\n"
        s+= "    sigma [unif. std] : " +str(self.sigma)+"\n"
        s+= "    mu_c [centroids] :" +str(self.mu)+"\n"
        s+= "    w_c [component scores]:" +str(self.w)+"\n"
        if self.outliers is not None:
            s+= "    vol0 [ref. volume for outliers]:" +str(self.outliers["vol0"])+"\n"
            s+= "    eta0 [outlier vs GMM log-ratio]:" +str(self.outliers["eta0"])+"\n"
        return s

    # -------------------------------------

    # Hack to ensure a correct value of spec when Unpickling. See diffICP.spec.CPU_Unpickler and
    # https://docs.python.org/3/library/pickle.html#handling-stateful-objects
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.spec = defspec


    #####################################################################
    ###
    ### EM algorithm functions
    ###
    #####################################################################

    def log_ratio_to_proba(self, eta):
        '''
        From log-odds-ratio to log probabilities in a Bernoulli distribution.
        This can be parallelized, i.e., eta, p and q can be arrays (of similar size).

        :param eta: log-odds-ratio of a Bernoulli distribution, i.e., eta=log(p/q)
        :return: log p, log q
        '''

        if not isinstance(eta, torch.Tensor):
            eta = torch.tensor(eta, **self.spec)                                    # try to convert
        Z = torch.stack((torch.zeros(eta.shape), eta), dim=0).logsumexp(dim=0)      # exp(Z) = 1 + exp(eta)
        return eta-Z, -Z

    # -------------------------------

    def log_responsibilities(self, X):
        '''
        Compute log-responsibilities (without outliers).

        :param X: torch.tensor of size (N,D) : data points
        :return: torch.tensor of size (N,C) : log-responsibility lgamma_nc of each data point X_n to each GMM component c
        '''

        X = X.detach()                                                              # to be sure
        D2_nc = ((X[:,None,:]-self.mu[None,:,:])**2).sum(-1)                        # shape (N,C)
        lgamma_nc = log_softmax(self.w[None,:] - D2_nc/(2*self.sigma**2), dim=1)    # shape (N,C)
        return lgamma_nc

    # -------------------------------

    def EM_step_torch(self, X):
        '''
        EM_step function (torch version). Performs ONE alternation of (E step, M step) for fitting the GMM model to the data points X.

        E step : compute (log)-responsibilities gamma_nc of each data points X_n to each GMM component c (plus outlier responsibilities if present).

        M step : update GMM parameters based on the new log-responsibilities.

        This function updates the GMM's internal state + returns three quantities of potential interest, Y, Cfe and FE (see below).

        :param X: torch.tensor(N,D) - data points.
        :return:
          - Y(N,D) : updated "quadratic target" associated to each data point : y_n = sum_c ( gamma_{nc}*mu_c ) ;
          - Cfe (number) : updated "free energy offset", i.e., the terms in EM free energy which do not involve (directly) the input X values ;
          - FE (number) : updated free energy, FE = Cfe + sum_n (1-gamma0_n) * (x_n-y_n)**2 / (2*sigma**2)
        '''

        ### This is the plain _pytorch implementation. Reductions over either dimension (N or C) can be performed, e.g., as:
        #   res_c = sum_n ( gamma_{nc}*stuff(n,c) ) / sum_n gamma_{nc}   --> res = (softmax(lgamma_nc,dim=0) * stuff).sum(dim=0)
        #   res_n = sum_c ( gamma_{nc}*stuff(n,c) )                      --> res = (lgamma_nc.exp() * stuff).sum(dim=1)
        # NOTA: in this Pytorch version, it would be suboptimal to separate the E_step and M_step in two separate functions,
        # because both rely on the quadratic distance matrix D2_nc, that should better be computed only once.

        X = X.detach()                                              # to be sure
        N = X.shape[0]

        D2_nc = ((X[:,None,:]-self.mu[None,:,:])**2).sum(-1)        # _nc means that shape = (N,C)
        loggaussnorm = self.D * ( np.log(self.sigma) + 0.5*np.log(2*math.pi) )
        Zw = self.w.logsumexp(dim=0)

        ### "E step" : compute responsibilities (using previous GMM parameters)

        t_nc = self.w[None, :] - Zw - D2_nc/(2*self.sigma**2) - loggaussnorm     # exp(t_nc) = pi_c * P(x_n|cluster_c)
        T_n = t_nc.logsumexp(dim=1)                                         # exp(T_n) = \sum_{c>=1} exp(t_nc)  ; "total component score"
        lgamma_nc = t_nc - T_n[:,None]
        # equivalently,  lgamma_nc = log_softmax(t_nc, dim=1).  But T_n is explicitly required when there are outliers, see below.
        gamma_nc = lgamma_nc.exp()

        if self.outliers is not None:
            # In that case, gamma_nc above represent CONDITIONAL responsibilities, given that x_n is not an outlier.
            eta0 = self.outliers["eta0"]                 # eta0 = log(pi_0/1-pi_0) ; current outliers log-odds-ratio for GMM model
            if self.outliers["vol0"] is None:
                self.set_vol0(X)                         # (first call to E_step fixes vol0 if it was not precised)
            logJ0 = -np.log(self.outliers["vol0"])       # J0 = P(x|outlier) = 1 / vol0
            eta0_n = eta0 + logJ0 - T_n                                # outliers log-odds-ratio for EM responsibilities of component n
            lgamma0_n, lgammaT_n = self.log_ratio_to_proba(eta0_n)     # gamma0_n = outlier responsibility for component n ; gammaT_n = 1 - gamma0_n

        ### "M step" : update GMM parameters

        if self.to_optimize["mu"]:
            self.mu = softmax(lgamma_nc,dim=0).t() @ X      # shape (C,2)

        if self.outliers is not None and self.to_optimize["eta0"]:
            self.outliers["eta0"] = (lgamma0_n.logsumexp(dim=0) - lgammaT_n.logsumexp(dim=0)).item()     # eta0 = log(pi0/1-pi0) = sum_n gamma0_n / sum_n gammaT_n

        if self.to_optimize["w"]:
            self.w = lgamma_nc.logsumexp(dim=0)                             # exp(w_c) = sum_n gamma_{nc} [GMM score without outliers]

        if self.to_optimize["sigma"]:
            NDsigma2 = (gamma_nc * D2_nc).sum()
            self.sigma = ( NDsigma2/(self.D*N) ).sqrt().item()              # .item() to return a simple float

        ### "Quadratic targets" Y(N,D)
        Y = (gamma_nc[:,:,None] * self.mu[None,:,:]).sum(1).reshape(N,self.D)

        ### "Free energy offset" term (part of the free energy that does not depend directly on the points X)
        # Without outliers:
        #   Cfe = sum_n Cfe_n
        #   Cfe_n = sum_c gamma_{nc} * [ (|mu_c|^2 - |y_n|^2)/(2*sig^2) + D*log(sig) -D/2*log(2pi) - log(pi_c) + log(gamma_{nc}) ]
        # With outliers:
        #   Cfe = sum_n [ (1-gamma0_n) [ Cfe_n + log(1-gamma0_n) - log(1-pi0)] + gamma0_n [ -logJ0 + log gamma0_n - log pi0 ] ]

        lpi_c = self.w - self.w.logsumexp(dim=0)
        Cfe_n_comp = (gamma_nc * ( ( (self.mu ** 2).sum(-1)[None,:] - (Y**2).sum(-1)[:,None] ) / (2 * self.sigma ** 2)
                            + lgamma_nc - lpi_c[None,:] )).sum(dim=1) + loggaussnorm
        if self.outliers is None:
            Cfe = Cfe_n_comp.sum()
        else:
            gamma0_n = lgamma0_n.exp()
            gammaT_n = lgammaT_n.exp()
            lpi0, lpiT = self.log_ratio_to_proba(self.outliers["eta0"])
            Cfe = (gammaT_n * (Cfe_n_comp + lgammaT_n - lpiT) + gamma0_n * (-logJ0 + lgamma0_n - lpi0)).sum().item()

        FE = Cfe + (gammaT_n * ((X-Y)**2).sum(-1)).sum().item() / (2*self.sigma**2)

        return Y, Cfe, FE


    # -------------------------------

    def EM_optimization(self, X, max_iterations=100, tol=1e-5):
        '''
        Iterated EM optimization of the model on the data points X.

        :param X: torch.tensor(N,D) - data points.
        :param max_iterations: maximum number of EM steps before break.
        :param tol: (relative) numerical tolerance on EM free energy for break.

        :return:
          - Y(N,D) : updated "quadratic target" associated to each data point : y_n = sum_c ( gamma_{nc}*mu_c ) ;
          - Cfe (number) : updated "free energy offset", i.e., the terms in EM free energy which do not involve (directly) the input X values ;
          - FE (number) : updated free energy, FE = Cfe + sum_n (1-gamma0_n) * (x_n-y_n)**2 / (2*sigma**2) ;
          - i (number) : required number of EM iterations
        '''

        Y, Cfe, FE, last_FE = None, None, None, None
        for i in range(max_iterations):
            Y, Cfe, FE = self.EM_step(X)
            if last_FE is not None and tol is not None and abs(FE-last_FE) < tol * abs(last_FE):
                print(f"GMM optimization - reached required tolerance of {tol} in {i+1} EM steps")
                return Y, Cfe, FE, i+1
            last_FE = FE
        print(f"GMM optimization - reached maximum number of iterations : {max_iterations}")
        return Y, Cfe, FE, i+1

    #####################################################################################
    ###
    ### KeOps implementation of the EM functions, if available (hopefully faster)
    ###
    #####################################################################################

    # Same principle as above, but now responsibilities "lgam_nc" are a SYMBOLIC 2D LazyTensor, so we cannot access its
    # value directly. Responsibilities are only available through the application of a reduction in either dimension :
    #
    #   res_c = sum_n ( gamma_{nc}*stuff(n,c) ) / sum_n gamma_{nc}   --> res = lgam_nc.sumsoftmaxweight(stuff,axis=0)
    #   res_n = sum_c ( gamma_{nc}*stuff(n,c) )                      --> res = lgam_nc.sumsoftmaxweight(stuff,axis=1)
    # or alternatively
    #   res_c = sum_n ( gamma_{nc}*stuff(n,c) )  with stuff > 0       --> res = lgam_nc.logsumexp(weight=stuff,axis=0).exp()

    if use_keops:

        def E_step_keops(self, X):
            '''
            E step (keops implementation). Compute component responsibilities, for current GMM model and given data points.
            :param X: X: torch.tensor(N,D) - data points.
            :return: lgam_nc [log responsibilities as a KeOps symbolic tensor], lgam0_n [outlier log-responsibilities], lgamT_n [=log(1-gam0_n))]
            '''

            X = X.detach()                                                      # to be sure
            D2_nc = Vi(X).sqdist(Vj(self.mu))                                   # Quadratic distances (symbolic Keops matrix)
            loggaussnorm = torch.tensor([self.D * (np.log(self.sigma) + 0.5 * np.log(2 * math.pi))], **self.spec)
            Zw = self.w.logsumexp(dim=0)
            t_nc = Vj(self.w.view(self.C, 1)) - D2_nc / (2 * self.sigma ** 2) - Zw - loggaussnorm   # exp(t_nc) = pi_c * P(x_n|cluster_c)
            T_n = t_nc.logsumexp(axis=1)                                                            # exp(T_n) = sum_c exp(t_nc) ; "total component score"
            lgam_nc = t_nc - Vi(T_n)                                                                # gamma_nc = exp(lgam_nc) ; sum_c gamma_nc = 1

            if self.outliers is not None:
                # In that case, gamma_nc above represent CONDITIONAL responsibilities, given that x_n is not an outlier.
                eta0 = self.outliers["eta0"]                # eta0 = log(pi_0/1-pi_0) ; current outliers log-odds-ratio for GMM model
                if self.outliers["vol0"] is None:
                    self.set_vol0(X)                        # (first call to E_step fixes vol0 if it was not precised)
                logJ0 = -np.log(self.outliers["vol0"])      # J0 = P(x|outlier) = 1 / vol0
                eta0_n = eta0 + logJ0 - T_n                 # outliers log-odds-ratio for EM responsibilities of component n
                lgam0_n, lgamT_n = self.log_ratio_to_proba(eta0_n)  # gamma0_n = outlier responsibility for component n ; gammaT_n = 1 - gamma0_n
            else:
                lgam0_n, lgamT_n = None, None

            return lgam_nc, lgam0_n, lgamT_n

        # ----------------------------------------------------------------

        def M_step_keops(self, X, lgam_nc, lgam0_n=None, lgamT_n=None):
            '''
            Update GMM parameters, given data X and log-responsibilities lgam_nc (plus outlier responsibilities if present).
            :param lgam_nc, lgam0_n, lgam_T : as produced by self.E_step_keops()
            :return: None
            '''

            X = X.detach()  # to be sure
            N = X.shape[0]

            if self.to_optimize["mu"]:
                self.mu = lgam_nc.sumsoftmaxweight(Vi(X), axis=0).reshape(self.C, self.D)

            if self.to_optimize["w"]:
                h_c = lgam_nc.logsumexp(axis=0)         # exp(h_c) = N_c = \sum_n gamma_{nc}  [GMM score without outliers]
                self.w = h_c.view(self.C)               # (w = h_c + arbitrary constant, chosen here as 0)

            if self.outliers is not None and self.to_optimize["eta0"]:
                self.outliers["eta0"] = (lgam0_n.logsumexp(dim=0) - lgamT_n.logsumexp(dim=0)).item()  # eta0 = log(pi0/1-pi0) = sum_n gamma0_n / sum_n gammaT_n

            if self.to_optimize["sigma"]:
                D2_nc = Vi(X).sqdist(Vj(self.mu))  # Quadratic distances (symbolic Keops matrix)
                # NDsigma2 = lgam_nc.sumsoftmaxweight(D2_nc, axis=1).sum()
                NDsigma2 = lgam_nc.logsumexp(weight=D2_nc, axis=0).exp().sum()  # marginally faster (~= -10% when N=10000)
                self.sigma = (NDsigma2 / (self.D * N)).sqrt().item()  # (.item() to return a simple float)

        # ------------------------------------------------------------

        def EM_values_keops(self, lgam_nc, lgam0_n=None, lgamT_n=None):
            '''
            Compute EM-related values : Y (quadratic targets) and Cfe (free energy offset)
            :param lgam_nc, lgam0_n, lgam_T: as produced by self.E_step_keops()
            :return:
              - Y(N,D) : updated "quadratic target" associated to each data point : y_n = sum_c ( gamma_{nc}*mu_c ) ;
              - Cfe (number) : updated "free energy offset", i.e., the terms in EM free energy which do not involve (directly) the input X values
            '''

            # "Quadratic targets" Y(N,D)
            N = lgam_nc.shape[0]
            Y = lgam_nc.sumsoftmaxweight(Vj(self.mu), axis=1).reshape(N, self.D)

            ### "Free energy offset" term (part of the free energy that does not depend directly on the points X)
            # Without outliers:
            #   Cfe = sum_n Cfe_n
            #   Cfe_n = sum_c gamma_{nc} * [ (|mu_c|^2 - |y_n|^2)/(2*sig^2) + D*log(sig) -D/2*log(2pi) - log(pi_c) + log(gamma_{nc}) ]
            # With outliers:
            #   Cfe = sum_n [ (1-gamma0_n) [ Cfe_n + log(1-gamma0_n) - log(1-pi0)] + gamma0_n [ -logJ0 + log gamma0_n - log pi0 ] ]

            lpi_c = (self.w - torch.logsumexp(self.w, 0)).view(self.C, 1)  # keops requires 2D
            loggaussnorm = self.D * (np.log(self.sigma) + 0.5 * np.log(2 * math.pi))

            Cfe_n_comp = lgam_nc.sumsoftmaxweight( Vj(self.mu).sqnorm2() / (2 * self.sigma ** 2) - Vi(Y).sqnorm2() / (2 * self.sigma ** 2)
                                                       - Vj(lpi_c) + lgam_nc , axis=1) + loggaussnorm
            if self.outliers is None:
                Cfe = Cfe_n_comp.sum()
            else:
                gamma0_n = lgam0_n.exp()
                gammaT_n = lgamT_n.exp()
                lpi0, lpiT = self.log_ratio_to_proba(self.outliers["eta0"])
                logJ0 = -np.log(self.outliers["vol0"])  # J0 = P(x|outlier) = 1 / vol0
                Cfe = (gammaT_n * (Cfe_n_comp + lgamT_n - lpiT) + gamma0_n * (-logJ0 + lgam0_n - lpi0)).sum().item()

            return Y, Cfe

        # ---------------------------------------------------------------
        # All three operations (E+M+compute Y and Cfe). Does the same thing as EM_step_torch().

        def EM_step_keops(self, X):
            '''
            EM_step function (keops version). Performs ONE alternation of (E step, M step) for fitting the GMM model to the data points X.

            E step : compute (log)-responsibilities gamma_nc of each data points X_n to each GMM component c (plus outlier responsibilities if present).

            M step : update GMM parameters based on the new log-responsibilities.

            This function updates the GMM's internal state + returns three quantities of potential interest, Y, Cfe and FE (see below).

            :param X: torch.tensor(N,D) - data points.
            :return:
              - Y(N,D) : updated "quadratic target" associated to each data point : y_n = sum_c ( gamma_{nc}*mu_c ) ;
              - Cfe (number) : updated "free energy offset", i.e., the terms in EM free energy which do not involve (directly) the input X values ;
              - FE (number) : updated free energy, FE = Cfe + sum_n (1-gamma0_n) * (x_n-y_n)**2 / (2*sigma**2)
            '''
            log_gammas = self.E_step_keops(X)
            self.M_step_keops(X, *log_gammas)
            Y, Cfe = self.EM_values_keops(*log_gammas)
            if self.outliers is not None:
                gammaT_n = log_gammas[2].exp()
            else:
                gammaT_n = 1
            FE = Cfe + (gammaT_n * ((X - Y) ** 2).sum(-1)).sum().item() / (2 * self.sigma ** 2)

            return Y, Cfe, FE

    #####################################################################
    ###
    ### Other GMM functions
    ###
    #####################################################################

    def pi(self):
        '''Return the vector of GMM component weights pi_c (without outliers).'''
        return softmax(self.w)

    # ------------------------------------------------

    def get_sample(self, N):
        '''Generate a random sample of N points from the GMM distribution.'''
        samp = self.sigma * torch.randn(N, self.D, **self.spec)     # random normal samples
        # center around (random) components
        c = torch.distributions.categorical.Categorical(logits=self.w).sample((N,))
        for n in range(N):
            samp[n,:] += self.mu[c[n],:]
        return samp

    # --------------------------------------------------

    def plot(self, *samples, bounds=None, heatmap=True, color=None, cmap=cm.RdBu, heatmap_amplification=-1, registration=None):
        """
        Displays the model in 2D (adapted from a KeOps tutorial).
        Boundaries for plotting can be specified in either of two ways :

        - bounds = [xmin,xmax,ymin,max] provides hard-coded limits (primary used information if present)

        - samples = (possibly multiple) points sets of size (?,2). In which case, the plotting boundaries are automatically computed to match to extent of these point sets. (Note that the *actual* plotting of these point sets, if required, must be performed externally.)
        """

        ### bounds for plotting

        if bounds != None:
            # Use provided bounds directly
            xmin, xmax, ymin, ymax = tuple(bounds)
        else:
            # Compute from centroids and/or associated samples
            if len(samples) == 0:
                # get bounds directly from centroids
                samples = (self.mu,)
            xmin, xmax, ymin, ymax = get_bounds(*samples)    # in diffICP.visu

        ### Create a uniform grid on the unit square:

        res = 200
        xticks = np.linspace(xmin, xmax, res + 1)[:-1]  # + 0.5 / res
        yticks = np.linspace(ymin, ymax, res + 1)[:-1]  # + 0.5 / res
        X, Y = np.meshgrid(xticks, yticks)
        grid = torch.from_numpy(np.vstack((X.ravel(), Y.ravel())).T).to(**defspec).contiguous()
        # Adrien : https://stackoverflow.com/questions/48915810/what-does-contiguous-do-in-pytorch

        ### Show the pushforward of GMM distribution by a given registration (experimental!)

        if registration is not None:
            reggrid = registration.apply(grid)
            # estimate det of registration at all grid points
            vol_g = [None]*2
            for i,g in enumerate([reggrid,grid]):
                g2d = g.view(res,res,2)
                g_x = g2d[2:,:,:] - g2d[:-2,:,:]
                g_y = g2d[:,2:,:] - g2d[:,:-2,:]
                # duplicate missing lines / columns on the border
                g_x = torch.concatenate((g_x[0,:,:][None,:,:], g_x, g_x[-1,:,:][None,:,:]), dim = 0)
                g_y = torch.concatenate((g_y[:,0,:][:,None,:], g_y, g_y[:,-1,:][:,None,:]), dim = 1)
                # estimate volume of each cell in the grid
                vol_g[i] = (g_x[:,:,0]*g_y[:,:,1] - g_x[:,:,1]*g_y[:,:,0]).reshape((-1,))
            det_grid = vol_g[0] / vol_g[1]      # (=Jacobian of registration in each grid point)

        ### Heatmap:

        if heatmap:
            if registration is None:
                heatmap = self.likelihoods(grid)
            else:
                heatmap = self.likelihoods(reggrid) * det_grid      # * or / ?
            heatmap = (
                heatmap.view(res, res).data.cpu().numpy()
            )  # reshape as a "background" image

            scale = np.amax(np.abs(heatmap[:]))
            plt.imshow(
                heatmap_amplification * heatmap,
                interpolation="bilinear",
                origin="lower",
                vmin=-scale,
                vmax=scale,
                cmap=cmap,
                extent=(xmin, xmax, ymin, ymax),
            )

        ### Log-contours:
        if registration is None:
            log_heatmap = self.log_likelihoods(grid)
        else:
            log_heatmap = self.log_likelihoods(reggrid) + det_grid.log()    # + or - ?
        log_heatmap = log_heatmap.view(res, res).data.cpu().numpy()
        scale = np.amax(np.abs(log_heatmap[:]))
        levels = np.linspace(-scale, scale, 41)

        if color is None:
            color = "#C8A1A1"

        plt.contour(
            log_heatmap,
            origin="lower",
            linewidths=1.0,
            colors=color,
            levels=levels,
            extent=(xmin, xmax, ymin, ymax),
        )

    # ----------------------------------------------------------------

    def plot_bis(self, *samples, lgam_nc=None, bounds=None, cluster_colors=None):
        """
        Alternative plotting function in 2d, for demonstration.
        Each cluster is associated to a different color.
        Boundaries for plotting can be specified in either of two ways :
            (a) bounds = [xmin,xmax,ymin,max] provides hard-coded limits (primary used information if present)
         or (b) *samples = list of points sets of size (?,2). In which case, the plotting boundaries are automatically
                computed to match to extent of these point sets. (Note that the *actual* plotting of these point sets,
                if required, must be performed externally.)
        """

        ### bounds for plotting

        if bounds != None:
            # Use provided bounds directly
            xmin, xmax, ymin, ymax = tuple(bounds)
        else:
            # Compute from centroids and/or associated samples
            if len(samples) == 0:
                # get bounds directly from centroids
                samples = (self.mu,)
            xmin, xmax, ymin, ymax = get_bounds(*samples)    # in diffICP.visu

        ### Cycling colors
        if cluster_colors is None:
            cluster_colors = [matplotlib.colors.to_rgb(str) for str in plt.rcParams['axes.prop_cycle'].by_key()['color']]

        ### plot provided point sets with color representing the main cluster

        for X in samples:
            if lgam_nc is None:
                lgam_nc = self.log_responsibilities(X)
                affect = lgam_nc.argmax(dim=1)
            affect = lgam_nc.argmax(dim=1)
            # plt.plot(X[:,0],X[:,1],color=[cluster_colors[i] for i in affect])
            for c in range(self.C):
                plt.plot(X[affect==c,0],X[affect==c,1],'.',color=cluster_colors[c], alpha=.6)
        for c in range(self.C):
            plt.plot(self.mu[c,0], self.mu[c,1], "X", color="black", markersize=14)
        plt.pause(.1)


    # ----------------------------------------------------------------------
    # Functions below are directly adapted from the original KeOps tutorial.
    # They are included because they are used in the plot() function above
    # ----------------------------------------------------------------------

    def update_covariances(self):   # actually, gamma = *inverse* covariance
        """Computes the full covariance matrices from the model's parameters."""
        self.params["gamma"] = (torch.eye(self.D, **self.spec) * self.sigma**(-2))[None,:,:].repeat([self.C,1,1]).view(self.C,self.D**2)

    def weights(self):
        """Scalar factor in front of the exponential, in the density formula."""
        return softmax(self.w, 0) / self.sigma**self.D

    def weights_log(self):
        """Logarithm of the scalar factor, in front of the exponential."""
        return log_softmax(self.w, 0) - self.D * math.log(self.sigma)

    def likelihoods(self, sample):
        """Samples the density on a given point cloud."""
        sample = sample.to(**self.spec)
        self.update_covariances()
        # return (-Vi(sample).weightedsqdist(Vj(self.mu), Vj(self.params["gamma"]))/2).exp() @ self.weights()  # (keops version from the tutorial)
        return (-((sample[:,None,:]-self.mu[None,:,:])**2).sum(-1)/(2*self.sigma**2)).exp() @ self.weights()   # plain pytorch version

    def log_likelihoods(self, sample):
        """Log-density, sampled on a given point cloud."""
        sample = sample.to(**self.spec)
        self.update_covariances()
        # K_ij = -Vi(sample).weightedsqdist(Vj(self.mu), Vj(self.params["gamma"]))/2
        # return K_ij.logsumexp(dim=1, weight=Vj(self.weights().reshape(self.C,1)))  # (keops version from the tutorial)
        return (-((sample[:,None,:]-self.mu[None,:,:])**2).sum(-1)/(2*self.sigma**2) + self.weights_log()[None,:]).logsumexp(dim=1) # plain pytorch version



############################################################################################
############################################################################################
###
### Test
###
############################################################################################
############################################################################################

if __name__ == '__main__':
    # Running as a script

    ### Test EM functions

    if True:

        plt.ion()

        ## Create datapoints (spiral)

        N = 1000
        t = torch.linspace(0, 2 * np.pi, N + 1)[:-1]
        x = torch.stack((0.5 + 0.4 * (t / 7) * t.cos(), 0.5 + 0.3 * t.sin()), 1)
        x = x + 0.02 * torch.randn(x.shape)

        ## Create datapoints (iris)

        #from sklearn.datasets import load_iris
        #iris = load_iris()
        #x = tensor(iris['data'])[:,:2].contiguous().type(torchdtype)  # 2 premières colonnes uniquement
        #N = x.shape[0]

        from diffICP.visualization.visu import get_bounds
        xmin, xmax, ymin, ymax = get_bounds(x)

        # Add outliers
        N_outliers = 100
        outliers = torch.tensor([xmin, ymin])[None, :] + torch.tensor([xmax - xmin, ymax - ymin])[None, :] * torch.rand(N_outliers, 2)
        x = torch.cat((x, outliers), dim=0)
        x = x[torch.randperm(x.shape[0])]  # mix point order (to be sure)

        # Launch EM

        C = 10
        mu0 = x[torch.randint(0,N,(C,)),:]
        GMM = GaussianMixtureUnif(mu0, use_outliers=False)
        GMM.to_optimize = {
            "mu" : True,
            "sigma" : True,
            "w" : True,
            "eta0" : True
        }
        n = 0
        start = time.time()
        while n<1000:
            n+=1
            print(n)
            plt.clf()
            GMM.plot(x)
            my_scatter(x)

            # Compare Pytorch and Keops E steps : OK
            # lgam_torch = GMM.log_responsibilities(x)            # (N,C)
            # print(softmax(lgam_torch,dim=0).t() @ x)            # (C,2)
            # lgam_keops,_ = GMM.E_step_keops(x)
            # print(lgam_keops.sumsoftmaxweight(Vi(x), axis=0).reshape(C,2))

            # print(GMM.likelihoods(x))
            # print(GMM.log_likelihoods(x))

            GMM.EM_step(x)
            # GMM.EM_step_pytorch(x)
            if GMM.outliers:
                print(f"Outlier weight: {GMM.log_ratio_to_proba(GMM.outliers['eta0'])[0].exp().item()}" )
    #        print(GMM)
    #        input()

            plt.pause(.1)

        # input()
        print(f"Done. Total time : {time.time()-start}")

    ### Test plotting function with registration (experimental!)

    if False:

        plt.ion()
        bounds = (-0.5,1.5,-0.5,1.5)

        ### Load existing diffeomorphic registration (simpler)
        from diffICP.tools.spec import CPU_Unpickler
        loadfile = "saving/test_basic.pkl"
        with open(loadfile, 'rb') as f:
            yo = CPU_Unpickler(f).load()        # modified dill Unpickler (see diffPSR.spec.CPU_Unpickler)
        reg = yo["PSR"].Registration()
        amplif = 1                              # modify strength of a0 (for testing)
        reg.a0 *= amplif

        ### Also apply registration to a grid, for visualization
        from diffICP.visualization.grid import Gridlines
        bounds = (-0.5,1.5,-0.5,1.5)
        gridlines = Gridlines(np.linspace(*bounds[:2],30), np.linspace(*bounds[2:],30))
        reglines = gridlines.register(reg, backward=True)

        ### Apply registration to GMM model
        GMM = GaussianMixtureUnif(mu0=torch.tensor([[0.8,0.4],[0.2,0.5],[0.5,0.6]]))
        GMM.sigma = 0.1
        GMM.w = torch.tensor([0,0.5,1],**defspec)
        plt.figure()
        GMM.plot(bounds=bounds)
        gridlines.plot(color='gray',linewidth=0.5)
        plt.figure()
        GMM.plot(registration=reg, bounds=bounds)
        reglines.plot(color='gray',linewidth=0.5)
        plt.pause(.1)
        input()

    ### Basic K-means demo

    if False:

        plt.ion()

        ## Create datapoints
        N = 1000
        t = torch.linspace(0, 2 * np.pi, N + 1)[:-1]
        x = torch.stack((0.5 + 0.4 * (t / 7) * t.cos(), 0.5 + 0.3 * t.sin()), 1)
        x = x + 0.1 * torch.randn(x.shape)   # for demo
        C = 10
        mu0 = x[torch.randint(0, N, (C,)), :]
        GMM = GaussianMixtureUnif(mu0)
        GMM.to_optimize = {
            "mu": True,
            "sigma": False,
            "w": False
        }
        GMM.sigma = 0.00001       # close to 0, so ~= K-means

        n = 0
        while n < 100:
            n += 1
            print(n)
            plt.clf()

            GMM.plot_bis(x)
            input()

            GMM.EM_step(x)





