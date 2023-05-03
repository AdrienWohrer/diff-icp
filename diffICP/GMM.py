'''
Gaussian Mixture Model functionalities
'''

# Standard imports

import os, time

import copy

import numpy as np
rng = np.random.default_rng(seed=1234)

import torch
from torch.nn import Module
from torch.nn.functional import softmax, log_softmax
torch.manual_seed(1234)

from matplotlib import pyplot as plt
import matplotlib.cm as cm

from pykeops.torch import Vi, Vj, LazyTensor, Pm
import pykeops

# torch type and device
use_cuda = torch.cuda.is_available()
torchdeviceId = torch.device("cuda:0") if use_cuda else "cpu"
torchdtype = torch.float32
tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
torch.manual_seed(1234)

# PyKeOps counterpart
KeOpsdeviceId = torchdeviceId.index  # id of Gpu device (in case Gpu is used)
KeOpsdtype = torchdtype.__str__().split(".")[1]  # 'float32'


#################################################
# Import from other files in this directory :

from diffICP.visu import my_scatter


#################################################
# Adaptation de la "GaussianMixture" présentée dans les exemples KeOps (voir z_commente_gaussian_mixture),
# mais avec des covariances uniformes isotropes sigma^2*Id
#
# mu0 = emplacement initial des centroides


class GaussianMixtureUnif(Module):
    def __init__(self, mu0):
        super(GaussianMixtureUnif, self).__init__()				# Adrien : https://rhettinger.wordpress.com/2011/05/26/super-considered-super/

        self.params = {}
        self.mu = mu0.clone().detach()                      # Just to be sure, copy and detach from any computational graph
        self.C, self.D = self.mu.shape
        r = self.mu.var(0).sum().sqrt().item()              # "typical radius" of the cloud of centroids. Hence, each centroid takes a "typical volume" r^D/C
        self.sigma = 0.1 * (r / self.C**(1/self.D))         # 0.1 times the "typical radius" of each centroid
        self.w = torch.zeros(self.C, 1).type(torchdtype)    # w_c = "score" de chaque composante. Son poids est pi_c = exp(w_c) / sum_{c'} exp(w_{c'})
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
        G2 = GaussianMixtureUnif(self.mu)
        G2.sigma = self.sigma #.clone().detach() # (when sigma was wrapped inside a pytorch tensor. Now a simple float)
        G2.w = self.w.clone().detach()
        G2.to_optimize = copy.deepcopy(self.to_optimize)
        return G2
        
    #####################################################################
    ###
    ### Les fonctions dont j'ai vraiment besoin ici : algorithme EM
    ###
    #####################################################################

    # X(N,D) : data points
    #
    # La fonction renvoie :
    # - la "cible quadratique" associée à chaque datapoint : y_n = sum_c ( gamma_{nc}*mu_c )
    # - le "free energy offset" Cfe, la partie de l'énergie libre qui ne dépend pas (directement) des valeurs X en entrée

    ### Version "PyTorch basique" (pour debug ; peut causer des nan dans le calcul de mu)
    # TODO: mettre à jour en utilisant les fonctions softmax de pytorch
    # TODO : séparer en E_step, M_step, comme la version keops

    def EM_step_pytorch(self,X):

        X = X.detach()                                              # to be sure
        N = X.shape[0]
        
        D2_nc = ((X[:,None,:]-self.mu[None,:,:])**2).sum(-1)        # shape (N,C)
        # "E step" : mise à jour des 'responsibilities'
        pi_c = self.w.exp().reshape(self.C)          # plus simple avec une seule dimension (mais la shape (C,1) de w reste nécessaire pour la partie LazyTensors)
        pi_c = pi_c / pi_c.sum()
        gamma_nc = pi_c[None,:] * (-D2_nc/(2*self.sigma**2)).exp()  # shape (N,C)
        gamma_nc = gamma_nc / gamma_nc.sum(1)[:,None]
        
        # "M step" : mise à jour des paramètres GMM
        
        if self.to_optimize["mu"]:
            self.mu =( (gamma_nc[:,:,None]*X[:,None,:]).sum(0) / gamma_nc[:,:,None].sum(0) ).reshape(self.C,self.D)
            # warning: can cause nan)
            
        if self.to_optimize["w"]:
            pi_c = gamma_nc.mean(0)
            self.w = pi_c.log().reshape(self.C,1)
            self.w[self.w.isnan()] = -100                # just in case
            self.w -= self.w.mean()                      # center the scores
            
        if self.to_optimize["sigma"]:
            NDsigma2 = (gamma_nc * D2_nc).sum()
            self.sigma = ( NDsigma2/(self.D*N) ).sqrt().item()      # .item() to return a simple float

        # "Cibles quadratiques" Y(N,D)
        Y = (gamma_nc[:,:,None] * self.mu[None,:,:]).sum(1).reshape(N,self.D)

        # "Free energy offset" term (part of the free energy that does not depend directly on the points X)
        # Cfe = sum_n sum_c gamma_{nc} * [ (|mu_c|^2 - |y_n|^2)/(2*sig^2) + D*log(sig) - log(pi_c) + log(gamma_{nc}) ]
        Cfe = (gamma_nc * ( ( (self.mu ** 2).sum(-1)[None,:] - (Y**2).sum(-1)[:,None] ) / (2 * self.sigma ** 2)
                            + self.D * np.log(self.sigma) - pi_c.log()[None,:] + gamma_nc.log() ) ).sum()

        return Y, Cfe.item()


    ### Versions KeOps équivalentes
    ###############################

    # Les responsabilités gamma_nc de chaque centroïde mu_c sur chaque point X_n sont codées par leur log,
    # lgam_nc tel que gamma_nc = exp(lgam_nc)
    # d'où les équivalences:
    #   res_c = sum_n ( gamma_{nc}*truc(n,c) ) / sum_n gamma_{nc}   --> res = lgam_nc.sumsoftmaxweight(truc,axis=0)
    #   res_n = sum_c ( gamma_{nc}*truc(n,c) )                      --> res = lgam_nc.sumsoftmaxweight(truc,axis=1)
    # ou alternativement
    #   res_c = sum_n ( gamma_{nc}*truc(n,c) )  avec truc > 0       --> res = lgam_nc.logsumexp(weight=truc,axis=0).exp()
    #
    # De même, les scores w_c sont tels que pi_c = exp(w_c) / sum_{c'} exp(w_{c'}), et donc (en Pytorch)
    #   pi_c = w.softmax()
    #   logpi_c = w - w.logsumexp()

    ### E step : update and return log-responsibilities.
    # For convenience, also compute and return h_c such that exp(h_c) = N_c = \sum_n gamma_{nc}

    def E_step(self, X):

        X = X.detach()                              # to be sure
        D2_nc = Vi(X).sqdist(Vj(self.mu))           # Quadratic distances (symbolic Keops matrix)
        k_nc = Vj(self.w) - D2_nc/(2*self.sigma**2) # log(gamma_nc) avant normalisation
        Z_n = k_nc.logsumexp(axis=1)                # exp(Z_n) = sum_c exp(k_nc)
        lgam_nc = k_nc - Vi(Z_n)                    # d'où gamma_nc = exp(lgam_nc) ; et sum_c gamma_nc = 1
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
                h_c = lgam_nc.logsumexp(axis=0)     # exp(h_c) = N_c = \sum_n gamma_{nc}
            self.w = h_c - h_c.mean()               # (w = h_c + random constant, chosen here so that w.mean()=0 )

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

        logpi_c = self.w - torch.logsumexp(self.w,0)
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

    ### Rajout Adrien (uniquement dans le cas des covariances uniformes!)

    def get_sample(self, N):
        """Generates a sample of N points."""
        samp = torch.zeros(N, self.D).normal_(0,self.sigma)     # random normal samples
        # center around (random) components
        pi = softmax(self.w,0).numpy().flatten()                # component weights
        c = rng.multinomial(1,pi,N)                             # random component indices    
        c = np.where(c==1)[1]
        for n in range(N):
            samp[n,:] += self.mu[c[n],:]
        return samp

    ##################################################################################################################"

    ##################################################
    ###
    ### Les fonctions ci-dessous sont conservées principalement pour utiliser la fonctionnalité de plot
    ###
    ##################################################

    def update_covariances(self):   # en fait, gamma = *inverse* covariance
        """Computes the full covariance matrices from the model's parameters."""
        self.params["gamma"] = (torch.eye(self.D)*self.sigma**(-2))[None,:,:].repeat([self.C,1,1]).view(self.C,self.D**2)      # TODO CHECK (écrit vite fait)

#    def covariances_determinants(self): # en fait, det(gamma) = det(Sigma)^(-1)
#        return self.sigma**(-2*self.D) * torch.ones(self.C).type(dtype)      # TODO CHECK (écrit vite fait)

    def weights(self):
        """Scalar factor in front of the exponential, in the density formula."""
        return softmax(self.w, 0) / self.sigma**self.D

    def weights_log(self):
        """Logarithm of the scalar factor, in front of the exponential."""
        return log_softmax(self.w, 0) + self.D * self.sigma.log()

    def likelihoods(self, sample):
        """Samples the density on a given point cloud."""
        self.update_covariances()
        return (
            -Vi(sample).weightedsqdist(Vj(self.mu), Vj(self.params["gamma"]))
        ).exp() @ self.weights()

    def log_likelihoods(self, sample):
        """Log-density, sampled on a given point cloud."""
        self.update_covariances()
        K_ij = -Vi(sample).weightedsqdist(Vj(self.mu), Vj(self.params["gamma"]))
        return K_ij.logsumexp(dim=1, weight=Vj(self.weights()))

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
        grid = torch.from_numpy(np.vstack((X.ravel(), Y.ravel())).T).contiguous().type(torchdtype)	
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
    x = x.type(torchdtype)
    
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
        GMM.plot(x)
        my_scatter(x)
        
        GMM.EM_step(x)
#        GMM.EM_step_pytorch(x)
        print(GMM)
#        input()
        plt.pause(.1)
    
    input()
    

