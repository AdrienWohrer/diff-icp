'''
Gaussian Mixture Model functionalities. Adaptation of the GaussianMixture presented in KeOps tutorials, with the following main modifications:
- only uniform isotropic covariances sigma^2*Id
- added functions for EM fitting of the model to a set of data
'''

# Standard imports

import copy, math
import importlib.util

import numpy as np
from matplotlib import pyplot as plt
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
###
### _basic version : does not use KeOps, so in theory it can also be used in Windows
###
#####################################################################################
#####################################################################################

class GaussianMixtureUnif_basic(Module):

    ### Initialization
    # mu0 = initial emplacement of centroids
    # spec = dictionary (dtype and device) where the GMM model operates (see diffICP.spec)

    def __init__(self, mu0, spec=defspec):
        super(GaussianMixtureUnif_basic, self).__init__()			# https://rhettinger.wordpress.com/2011/05/26/super-considered-super/

        self.params = {}
        self.spec = spec
        self.mu = mu0.clone().detach().to(**spec)           # Just to be sure, copy and detach from any computational graph
        self.C, self.D = self.mu.shape
        r = self.mu.var(0).sum().sqrt().item()              # "typical radius" of the cloud of centroids. Hence, each centroid takes a "typical volume" r^D/C
        self.sigma = 0.1 * (r / self.C**(1/self.D))         # 0.1 times the "typical radius" of each centroid
        self.w = torch.zeros(self.C, **spec)                # w_c = "score" de chaque composante. Son poids est pi_c = exp(w_c) / sum_{c'} exp(w_{c'})
        self.to_optimize = {
            "sigma" : True,
            "mu" : True,
            "w" : True
        }

    def __str__(self):
        s = super(GaussianMixtureUnif_basic, self).__str__()
        s+= ": Gaussian Mixture with Uniform covariances. Parameters:\n"
        s+= "    C [# components] : "+str(self.C)+"\n"
        s+= "    sigma [unif. std] : " +str(self.sigma)+"\n"
        s+= "    mu_c [centroids] :" +str(self.mu)+"\n"
        s+= "    w_c [component scores]:" +str(self.w)+"\n"
        return s

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

    ### Basic implementation principle :
    # the responsibilities gamma_nc of each centroid mu_c for each data point X_n are coded by their log lgamma_nc. Thus (in Pytorch),
    #       gamma_nc = lgamma_nc.exp()
    # Similarly, component priors are encoded by the SCORES w_c such that pi_c = exp(w_c) / sum_{c'} exp(w_{c'}). Thus (in Pytorch),
    #       pi = w.softmax()
    #       logpi = w - w.logsumexp()

    ### Useful function : Compute log-responsibilities
    #     lgamma = log_responsibilities(X)
    # with
    #   - X(N,D) : data points [pytorch tensor]
    #   - lgamma(N,C) : log-responsibility of each data point X_n to each GMM component c [pytorch tensor]

    def log_responsibilities(self,X):

        X = X.detach()                                                              # to be sure
        D2_nc = ((X[:,None,:]-self.mu[None,:,:])**2).sum(-1)                        # shape (N,C)
        lgamma_nc = log_softmax(self.w[None,:] - D2_nc/(2*self.sigma**2), dim=1)    # shape (N,C)
        return lgamma_nc

    ### EM_step function (pytorch version)
    #       Y,Cfe = EM_step(X)
    # The function performs ONE alternation of (E step,M step) for fitting the GMM model to the data points X.
    # E step : compute (log)-responsibilities gamma_nc of each data points X_n to each GMM component c
    # M step : update GMM parameters based on the new log-responsibilities
    # Input :
    #   - X(N,D) : data points
    # Outputs :
    #   - Y(N,D) : updated "quadratic target" associated to each data point : y_n = sum_c ( gamma_{nc}*mu_c )
    #   - Cfe (number) : updated "free energy offset", i.e., the terms in EM free energy which do not involve (directly) the input X values

    ### This is the plain _pytorch implementation :
    #
    # Reductions over either dimension (N or C) can be performed, e.g., as:
    #   res_c = sum_n ( gamma_{nc}*stuff(n,c) ) / sum_n gamma_{nc}   --> res = (softmax(lgamma_nc,dim=0) * stuff).sum(dim=0)
    #   res_n = sum_c ( gamma_{nc}*stuff(n,c) )                      --> res = (lgamma_nc.exp() * stuff).sum(dim=1)

    def EM_step_pytorch(self,X):

        X = X.detach()                                              # to be sure
        N = X.shape[0]

        D2_nc = ((X[:,None,:]-self.mu[None,:,:])**2).sum(-1)        # shape (N,C)

        # "E step" : compute responsibilities (using previous GMM parameters)

        lgamma_nc = log_softmax(self.w[None,:] - D2_nc/(2*self.sigma**2), dim=1)  # shape (N,C)
        gamma_nc = lgamma_nc.exp()

        # "M step" : update GMM parameters

        if self.to_optimize["mu"]:
            self.mu = softmax(lgamma_nc,dim=0).t() @ X      # shape (C,2)
            
        if self.to_optimize["w"]:
            h_c = lgamma_nc.logsumexp(dim=0)             # exp(h_c) = N_c = \sum_n gamma_{nc}
            self.w = h_c - h_c.mean()                    # (w = h_c + arbitrary constant, chosen here so that w.mean()=0 )

        if self.to_optimize["sigma"]:
            NDsigma2 = (gamma_nc * D2_nc).sum()
            self.sigma = ( NDsigma2/(self.D*N) ).sqrt().item()      # .item() to return a simple float

        # "Quadratic targets" Y(N,D)
        Y = (gamma_nc[:,:,None] * self.mu[None,:,:]).sum(1).reshape(N,self.D)

        # "Free energy offset" term (part of the free energy that does not depend directly on the points X)
        # Cfe = sum_n sum_c gamma_{nc} * [ (|mu_c|^2 - |y_n|^2)/(2*sig^2) + D*log(sig) - log(pi_c) + log(gamma_{nc}) ]
        logpi_c = self.w - self.w.logsumexp(dim=0)
        Cfe = (gamma_nc * ( ( (self.mu ** 2).sum(-1)[None,:] - (Y**2).sum(-1)[:,None] ) / (2 * self.sigma ** 2)
                            + self.D * np.log(self.sigma) - logpi_c[None,:] + lgamma_nc ) ).sum()

        return Y, Cfe.item()

    # NOTA: in this Pytorch version, it would be suboptimal to separate the E_step and M_step in two separate functions,
    # because both rely on the quadratic distance matrix D2_nc, that should better be computed only once.


    #####################################################################
    ###
    ### Other GMM functions
    ###
    #####################################################################

    ### Produce a sample from the GMM distribution

    def get_sample(self, N, rng=np.random.default_rng()):
        """Generates a sample of N points."""
        samp = self.sigma * torch.randn(N, self.D, **self.spec)     # random normal samples
        # center around (random) components
        c = torch.distributions.categorical.Categorical(logits=self.w).sample((N,))
        for n in range(N):
            samp[n,:] += self.mu[c[n],:]
        return samp

    ### PLOTTING FUNCTION (adapted from a KeOps tutorial)

    #

    def plot(self, *samples, bounds=None, heatmap=True, color=None, cmap=cm.RdBu, heatmap_amplification=-1, registration=None):
        """Displays the model in 2D (adapted from a KeOps tutorial).
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

    ### Functions below are directly adapted from the original KeOps tutorial.
    ### They are included because they are used in the plotting function above

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



#####################################################################################
#####################################################################################
###
### _keops version : added KeOps functionalities if available (hopefully faster)
###
#####################################################################################
#####################################################################################

class GaussianMixtureUnif_keops(GaussianMixtureUnif_basic):

    ### KeOps implementation of the EM functions
    ############################################

    # Same principle as above, but now responsibilities "lgam_nc" are a SYMBOLIC 2D LazyTensor, so we cannot access its
    # value directly. Responsibilities are only available through the application of a reduction in either dimension :
    #
    #   res_c = sum_n ( gamma_{nc}*stuff(n,c) ) / sum_n gamma_{nc}   --> res = lgam_nc.sumsoftmaxweight(stuff,axis=0)
    #   res_n = sum_c ( gamma_{nc}*stuff(n,c) )                      --> res = lgam_nc.sumsoftmaxweight(stuff,axis=1)
    # or alternatively
    #   res_c = sum_n ( gamma_{nc}*stuff(n,c) )  with stuff > 0       --> res = lgam_nc.logsumexp(weight=stuff,axis=0).exp()

    ### E step : Compute log-responsibilities (returned as a symbolic keops LazyTensor).
    # For convenience, also compute and return h_c such that exp(h_c) = N_c = \sum_n gamma_{nc}

    def E_step_keops(self, X):

        X = X.detach()  # to be sure
        D2_nc = Vi(X).sqdist(Vj(self.mu))  # Quadratic distances (symbolic Keops matrix)
        k_nc = Vj(self.w.view(self.C, 1)) - D2_nc / (2 * self.sigma ** 2)
        # log(gamma_nc) before normalization
        Z_n = k_nc.logsumexp(axis=1)  # exp(Z_n) = sum_c exp(k_nc)
        lgam_nc = k_nc - Vi(Z_n)  # hence gamma_nc = exp(lgam_nc) ; and sum_c gamma_nc = 1
        h_c = lgam_nc.logsumexp(axis=0)  # exp(h_c) = N_c = \sum_n gamma_{nc}
        return lgam_nc, h_c

    ### "M step" : Update GMM parameters, given data X and log-responsibilities lgam_nc.
    # Computation can be made a little faster by providing numbers h_c such that exp(h_c) = N_c = \sum_n gamma_{nc}

    def M_step_keops(self, X, lgam_nc, h_c=None):

        X = X.detach()  # to be sure
        N = X.shape[0]

        if self.to_optimize["mu"]:
            self.mu = lgam_nc.sumsoftmaxweight(Vi(X), axis=0).reshape(self.C, self.D)

        if self.to_optimize["w"]:
            if h_c is None:
                h_c = lgam_nc.logsumexp(axis=0)  # exp(h_c) = N_c = \sum_n gamma_{nc}
            self.w = (h_c - h_c.mean()).view(self.C)  # (w = h_c + arbitrary constant, chosen here so that w.mean()=0 )

        if self.to_optimize["sigma"]:
            D2_nc = Vi(X).sqdist(Vj(self.mu))  # Quadratic distances (symbolic Keops matrix)
            # NDsigma2 = lgam_nc.sumsoftmaxweight(D2_nc, axis=1).sum()
            NDsigma2 = lgam_nc.logsumexp(weight=D2_nc,
                                         axis=0).exp().sum()  # marginally faster (like, -10% when N=10000)
            self.sigma = (NDsigma2 / (self.D * N)).sqrt().item()  # (.item() to return a simple float)

    ### Compute EM-related values : Y (quadratic targets) and Cfe (free energy offset)
    # Computation can be made a little faster by providing numbers h_c such that exp(h_c) = N_c = \sum_n gamma_{nc}

    def EM_values_keops(self, lgam_nc, h_c=None):

        # "Cibles quadratiques" Y(N,D)
        N = lgam_nc.shape[0]
        Y = lgam_nc.sumsoftmaxweight(Vj(self.mu), axis=1).reshape(N, self.D)

        # "Free energy offset" term (part of the free energy that does not depend directly on the points X)
        # Cfe = sum_n sum_c gamma_{nc} * [ (|mu_c|^2 - |y_n|^2)/(2*sig^2) + D*log(sig) - log(pi_c) + log(gamma_{nc}) ]

        logpi_c = (self.w - torch.logsumexp(self.w, 0)).view(self.C, 1)  # keops requires 2D
        if h_c is None:  # computation not requiring explicitly N_c (a little slower)
            Cfe = N * self.D * np.log(self.sigma) + lgam_nc.sumsoftmaxweight(
                Vj(self.mu).sqnorm2() / (2 * self.sigma ** 2) - Vi(Y).sqnorm2() / (2 * self.sigma ** 2) - Vj(
                    logpi_c) + lgam_nc, axis=1).sum()
        else:  # a little faster
            N_c = h_c.exp()
            Cfe = ((N_c * (self.mu ** 2).sum(-1, True)).sum() - (Y ** 2).sum()) / (2 * self.sigma ** 2) \
                  + N * self.D * np.log(self.sigma) - (N_c * logpi_c).sum() \
                  + (lgam_nc.sumsoftmaxweight(lgam_nc, axis=0) * N_c).sum()

        return Y, Cfe.item()  # .item() to return a simple float

    ### All three operation (E+M+compute Y and Cfe). Kept for simplicity

    def EM_step_keops(self, X):
        lgam_nc, h_c = self.E_step_keops(X)
        self.M_step_keops(X, lgam_nc, h_c)
        Y, Cfe = self.EM_values_keops(lgam_nc, h_c)
        return Y, Cfe


#####################################################################################
#####################################################################################
###
### Actual version used : GaussianMixtureUnif. _pytorch or _keops version depending on availability
###
#####################################################################################
#####################################################################################

if use_keops:
    class GaussianMixtureUnif(GaussianMixtureUnif_keops):
        EM_step = GaussianMixtureUnif_keops.EM_step_keops

        # Custom deepcopy = copy GMM
        def __deepcopy__(self, memo):
            G2 = GaussianMixtureUnif(self.mu, self.spec)
            G2.sigma = self.sigma
            G2.w = self.w.clone().detach()
            G2.to_optimize = copy.deepcopy(self.to_optimize)
            return G2

else:
    class GaussianMixtureUnif(GaussianMixtureUnif_basic):
        EM_step = GaussianMixtureUnif_basic.EM_step_pytorch

        # Custom deepcopy = copy GMM (ugly to copy code twice, but heck...)
        def __deepcopy__(self, memo):
            G2 = GaussianMixtureUnif(self.mu, self.spec)
            G2.sigma = self.sigma
            G2.w = self.w.clone().detach()
            G2.to_optimize = copy.deepcopy(self.to_optimize)
            return G2


############################################################################################
###
### Test
###
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

        # Launch EM

        C = 10
        mu0 = x[torch.randint(0,N,(C,)),:]
        GMM = GaussianMixtureUnif(mu0)
        GMM.to_optimize = {
            "mu" : True,
            "sigma" : False,
            "w" : False
        }

        n = 0
        while n<100:
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

            print(GMM.likelihoods(x))
            print(GMM.log_likelihoods(x))

            GMM.EM_step(x)
    #        GMM.EM_step_pytorch(x)
    #        print(GMM)
    #        input()
            plt.pause(.1)

        # input()
    

    ### Test plotting function with registration (experimental!)
    if True:

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
        GMM = GaussianMixtureUnif(mu0=torch.tensor([[0.8,0.4],[0.2,0.5],[0.5,0.6]],**defspec))
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
