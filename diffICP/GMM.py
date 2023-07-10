'''
Gaussian Mixture Model functionalities
'''

# Standard imports

import os, time

import copy

import numpy as np

import torch
from torch.nn import Module
from torch.nn.functional import softmax, log_softmax

from matplotlib import pyplot as plt
import matplotlib.cm as cm

from pykeops.torch import Vi, Vj, LazyTensor, Pm

from diffICP.visu import my_scatter
from diffICP.spec import defspec, getspec

#################################################
# Adaptation de la "GaussianMixture" présentée dans les exemples KeOps (voir z_commente_gaussian_mixture),
# mais avec des covariances uniformes isotropes sigma^2*Id
#
# mu0 = emplacement initial des centroides
# spec = dictionary (dtype and device) where the GMM model operates (see diffICP.spec)

class GaussianMixtureUnif(Module):
    def __init__(self, mu0, spec=defspec):
        super(GaussianMixtureUnif, self).__init__()			# https://rhettinger.wordpress.com/2011/05/26/super-considered-super/

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
        s = super(GaussianMixtureUnif, self).__str__()
        s+= ": Gaussian Mixture with Uniform covariances. Parameters:\n"
        s+= "    C [# components] : "+str(self.C)+"\n"
        s+= "    sigma [unif. std] : " +str(self.sigma)+"\n"
        s+= "    mu_c [centroids] :" +str(self.mu)+"\n"
        s+= "    w_c [component scores]:" +str(self.w)+"\n"
        return s
        
    # Custom deepcopy = copy GMM 
    def __deepcopy__(self, memo):
        G2 = GaussianMixtureUnif(self.mu, self.spec)
        G2.sigma = self.sigma
        G2.w = self.w.clone().detach()
        G2.to_optimize = copy.deepcopy(self.to_optimize)
        return G2
    #
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

    # Basic usage :
    #       Y,Cfe = EM_step(X)
    # Input :
    #   - X(N,D) : data points
    # The function performs ONE alternation of (E step,M step) for fitting the GMM model to the data points X.
    # E step : compute (log)-responsibilities gamma_nc of each data points X_n to each GMM component c
    # M step : update GMM parameters based on the new log-responsibilities
    # Outputs :
    #   - Y(N,D) : updated "quadratic target" associated to each data point : y_n = sum_c ( gamma_{nc}*mu_c )
    #   - Cfe (number) : updated "free energy offset", i.e., the terms in EM free energy which do not involve (directly) the input X values
    #
    # The following helper function can also be useful :
    #       lgamma = E_step(X)
    # lgamma(N,C) are the log-responsibilities log(gamma_{nc}) for X in the current GMM model.
    #
    # The functions exist both in "_pytorch" versions (plain pytorch, manipulating large 2D arrays)
    # and "keops" versions (faster, use symbolic 2D LazyTensors---but as a drawback the LazyTensors cannot be used directly)

    ### Basic PyTorch versions
    ##########################

    # The responsibilities gamma_nc of each centroid mu_c for each data point X_n are coded by their log lgamma_nc. Thus (in Pytorch),
    #       gamma_nc = lgamma_nc.exp()
    # Similarly, component priors are encoded by the SCORES w_c such that pi_c = exp(w_c) / sum_{c'} exp(w_{c'}). Thus (in Pytorch),
    #       pi = w.softmax()
    #       logpi = w - w.logsumexp()
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
    # Instead, we provide an *additional* function E_step_pytorch, when one only needs to compute responsibilities.

    ### E step : Compute log-responsibilities (returned as a pytorch 2D array)

    def E_step_pytorch(self,X):

        X = X.detach()                                                              # to be sure
        D2_nc = ((X[:,None,:]-self.mu[None,:,:])**2).sum(-1)                        # shape (N,C)
        lgamma_nc = log_softmax(self.w[None,:] - D2_nc/(2*self.sigma**2), dim=1)    # shape (N,C)
        return lgamma_nc

    ### Equivalent KeOps versions
    #############################

    # Same principle as above, but now responsibilities "lgam_nc" are a SYMBOLIC 2D LazyTensor, so we cannot access its
    # value directly. Responsibilities are only available through the application of a reduction in either dimension :
    #
    #   res_c = sum_n ( gamma_{nc}*stuff(n,c) ) / sum_n gamma_{nc}   --> res = lgam_nc.sumsoftmaxweight(stuff,axis=0)
    #   res_n = sum_c ( gamma_{nc}*stuff(n,c) )                      --> res = lgam_nc.sumsoftmaxweight(stuff,axis=1)
    # or alternatively
    #   res_c = sum_n ( gamma_{nc}*stuff(n,c) )  with stuff > 0       --> res = lgam_nc.logsumexp(weight=stuff,axis=0).exp()

    ### E step : Compute log-responsibilities (returned as a symbolic keops LazyTensor).
    # For convenience, also compute and return h_c such that exp(h_c) = N_c = \sum_n gamma_{nc}

    def E_step(self, X):

        X = X.detach()                              # to be sure
        D2_nc = Vi(X).sqdist(Vj(self.mu))           # Quadratic distances (symbolic Keops matrix)
        k_nc = Vj(self.w.view(self.C,1)) - D2_nc/(2*self.sigma**2)
                                                    # log(gamma_nc) before normalization
        Z_n = k_nc.logsumexp(axis=1)                # exp(Z_n) = sum_c exp(k_nc)
        lgam_nc = k_nc - Vi(Z_n)                    # hence gamma_nc = exp(lgam_nc) ; and sum_c gamma_nc = 1
        h_c = lgam_nc.logsumexp(axis=0)             # exp(h_c) = N_c = \sum_n gamma_{nc}
        return lgam_nc, h_c

    ### "M step" : Update GMM parameters, given data X and log-responsibilities lgam_nc.
    # Computation can be made a little faster by providing numbers h_c such that exp(h_c) = N_c = \sum_n gamma_{nc}

    def M_step(self, X, lgam_nc, h_c=None):

        X = X.detach()                              # to be sure
        N = X.shape[0]

        if self.to_optimize["mu"]:
            self.mu = lgam_nc.sumsoftmaxweight(Vi(X),axis=0).reshape(self.C,self.D)

        if self.to_optimize["w"]:
            if h_c is None:
                h_c = lgam_nc.logsumexp(axis=0)          # exp(h_c) = N_c = \sum_n gamma_{nc}
            self.w = (h_c - h_c.mean()).view(self.C)     # (w = h_c + arbitrary constant, chosen here so that w.mean()=0 )

        if self.to_optimize["sigma"]:
            D2_nc = Vi(X).sqdist(Vj(self.mu))           # Quadratic distances (symbolic Keops matrix)
            #NDsigma2 = lgam_nc.sumsoftmaxweight(D2_nc, axis=1).sum()
            NDsigma2 = lgam_nc.logsumexp(weight=D2_nc, axis=0).exp().sum()  # marginally faster (like, -10% when N=10000)
            self.sigma = (NDsigma2 / (self.D * N)).sqrt().item()            # (.item() to return a simple float)

    ### Compute EM-related values : Y (quadratic targets) and Cfe (free energy offset)
    # Computation can be made a little faster by providing numbers h_c such that exp(h_c) = N_c = \sum_n gamma_{nc}

    def EM_values(self, lgam_nc, h_c=None):

        # "Cibles quadratiques" Y(N,D)
        N = lgam_nc.shape[0]
        Y = lgam_nc.sumsoftmaxweight(Vj(self.mu),axis=1).reshape(N,self.D)

        # "Free energy offset" term (part of the free energy that does not depend directly on the points X)
        # Cfe = sum_n sum_c gamma_{nc} * [ (|mu_c|^2 - |y_n|^2)/(2*sig^2) + D*log(sig) - log(pi_c) + log(gamma_{nc}) ]

        logpi_c = (self.w - torch.logsumexp(self.w,0)).view(self.C,1)       # keops requires 2D
        if h_c is None:                 # computation not requiring explicitly N_c (a little slower)
            Cfe = N * self.D * np.log(self.sigma) + lgam_nc.sumsoftmaxweight(
                Vj(self.mu).sqnorm2()/(2* self.sigma**2) - Vi(Y).sqnorm2()/(2* self.sigma**2) - Vj(logpi_c) + lgam_nc, axis=1).sum()
        else:                           # a little faster
            N_c = h_c.exp()
            Cfe = ( (N_c * (self.mu**2).sum(-1,True)).sum() - (Y**2).sum() ) / (2*self.sigma**2) \
                + N * self.D * np.log(self.sigma) - (N_c*logpi_c).sum() \
                + (lgam_nc.sumsoftmaxweight(lgam_nc,axis=0) * N_c).sum()

        return Y, Cfe.item()            # .item() to return a simple float

    ### All three operation (E+M+compute Y and Cfe). Kept for simplicity

    def EM_step(self, X):
        lgam_nc, h_c = self.E_step(X)
        self.M_step(X, lgam_nc, h_c)
        Y, Cfe = self.EM_values(lgam_nc, h_c)
        return Y, Cfe


    ##################################################################################################################"

    ### Addition Adrien (only valid in the case of uniform covariances)

    def get_sample(self, N, rng=np.random.default_rng()):
        """Generates a sample of N points."""
        samp = self.sigma * torch.randn(N, self.D, **self.spec)     # random normal samples
        # center around (random) components
        # (OLD) numpy version
        # pi = softmax(self.w,0).numpy().flatten()                    # component weights
        # c = rng.multinomial(1,pi,N)                                 # random component indices
        # c = np.where(c==1)[1]
        # (NEW) Torch version
        c = torch.distributions.categorical.Categorical(logits=self.w).sample((N,))
        for n in range(N):
            samp[n,:] += self.mu[c[n],:]
        return samp

    ##################################################################################################################"

    ##################################################
    ###
    ### Functions below are taken from a KeOps tutorial. Kept only to the use the plotting function (self.plot() below)
    ###
    ##################################################

    def update_covariances(self):   # actually, gamma = *inverse* covariance
        """Computes the full covariance matrices from the model's parameters."""
        self.params["gamma"] = (torch.eye(self.D, **self.spec) * self.sigma**(-2))[None,:,:].repeat([self.C,1,1]).view(self.C,self.D**2)

#    def covariances_determinants(self): # en fait, det(gamma) = det(Sigma)^(-1)
#        return self.sigma**(-2*self.D) * torch.ones(self.C)      # TODO CHECK (not tested)

    def weights(self):
        """Scalar factor in front of the exponential, in the density formula."""
        return softmax(self.w, 0) / self.sigma**self.D

    def weights_log(self):
        """Logarithm of the scalar factor, in front of the exponential."""
        return log_softmax(self.w, 0) + self.D * self.sigma.log()

    def likelihoods(self, sample):
        """Samples the density on a given point cloud."""
        sample = sample.to(**self.spec)
        self.update_covariances()
        return (
            -Vi(sample).weightedsqdist(Vj(self.mu), Vj(self.params["gamma"]))
        ).exp() @ self.weights()

    def log_likelihoods(self, sample):
        """Log-density, sampled on a given point cloud."""
        sample = sample.to(**self.spec)
        self.update_covariances()
        K_ij = -Vi(sample).weightedsqdist(Vj(self.mu), Vj(self.params["gamma"]))
        return K_ij.logsumexp(dim=1, weight=Vj(self.weights().reshape(self.C,1)))

    def neglog_likelihood(self, sample):
        """Returns -log(likelihood(sample)) up to an additive factor."""
        ll = self.log_likelihoods(sample)
        log_likelihood = torch.mean(ll)
        return -log_likelihood #+ self.sparsity * softmax(self.w, 0).sqrt().mean()

    # - *samples = list of points sets of size (?,D), only input here to compute boundaries.
    #              The actual plotting of these point sets must be done elsewhere
    # OR
    # - bounds = [xmin,xmax,ymin,max] to provide hard-coded limits (in which case samples are useless)

    def plot(self, *samples, bounds=None, heatmap=True, color=None):
        """Displays the model."""
        
        ### bounds for plotting

        if bounds != None:
            # Use provided bounds directly
            gmin, gmax = bounds[0::2], bounds[1::2]
            
        else:
            # Compute from centroids and/or associated samples
            if len(samples)==0:
                # get bounds directly from centroids
                samples = [self.mu]
            
            mins = torch.cat( tuple( xy.min(0).values.reshape(1,2) for xy in samples if len(xy)>0 ) , 0).min(0).values
            maxs = torch.cat( tuple( xy.max(0).values.reshape(1,2) for xy in samples if len(xy)>0 ) , 0).max(0).values
            
            relmargin = 0.2
            gmin = ((1+relmargin)*mins - relmargin*maxs).tolist()
            gmax = ((1+relmargin)*maxs - relmargin*mins).tolist()
        
        # Create a uniform grid on the unit square:
        res = 200
        xticks = np.linspace(gmin[0], gmax[0], res + 1)[:-1] #+ 0.5 / res
        yticks = np.linspace(gmin[1], gmax[1], res + 1)[:-1] #+ 0.5 / res
        X, Y = np.meshgrid(xticks, yticks)
        grid = torch.from_numpy(np.vstack((X.ravel(), Y.ravel())).T).contiguous()
                # Adrien : https://stackoverflow.com/questions/48915810/what-does-contiguous-do-in-pytorch

        # Heatmap:
        if heatmap:
            heatmap = self.likelihoods(grid)
            heatmap = (
                heatmap.view(res, res).data.cpu().numpy()
            )  # reshape as a "background" image

            scale = np.amax(np.abs(heatmap[:]))
            plt.imshow(
                -heatmap,
                interpolation="bilinear",
                origin="lower",
                vmin=-scale,
                vmax=scale,
                cmap=cm.RdBu,
                extent=(gmin[0], gmax[0], gmin[1], gmax[1]),
            )

        # Log-contours:
        log_heatmap = self.log_likelihoods(grid)
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
            extent=(gmin[0], gmax[0], gmin[1], gmax[1]),
        )



############################################################################################
###
### Test
###
############################################################################################

    
if False:
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
        # lgam_torch = GMM.E_step_pytorch(x)                  # (N,C)
        # print(softmax(lgam_torch,dim=0).t() @ x)            # (C,2)
        # lgam_keops,_ = GMM.E_step(x)
        # print(lgam_keops.sumsoftmaxweight(Vi(x), axis=0).reshape(C,2))

        # GMM.EM_step(x)
#        GMM.EM_step_pytorch(x)
#        print(GMM)
#        input()
        plt.pause(.1)
    
    input()
    

