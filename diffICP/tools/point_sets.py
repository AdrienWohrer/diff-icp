'''
Miscellaneous helper functions regarding the structure and manipulation of point sets.
'''

import math, warnings
import numpy as np
import torch
from pykeops.torch import LazyTensor
from diffICP.tools.kernel import GaussKernel

################################################

def intrinsic_scale(x):
    '''
    Compute an "intrinsic scale" sigma_x for point set x (provided as a torch 2d tensor).
    sigma_x is defined as the mean distance from each point to its nearest neighbor in the set.
    Intuitively, sigma_x represents the minimal amount of blurring necessary to "stop seeing" the point set structure.
    '''

    # TODO any cleaner way of doing this ?

    x = x.contiguous()
    D_ij = ((LazyTensor(x[:,None,:]) - LazyTensor(x[None,:,:])) ** 2).sum(-1)
    min_D_i = D_ij.Kmin(2, dim=1)[:,1] # (=second smallest ; the smallest is always 0, i.e., each point with itself)
    sigma_X = min_D_i.mean().sqrt()
    return 1 * sigma_X.item()     # multiplicative factor, close to 1, is ad hoc. TODO: probably depends on dimension D of the point sets (think!)


#########################################################################
# POINT SET DISTANCE
#########################################################################

# 1) Define some scale sigma_X for point set X, and similarly sigma_Y for point set Y.
#    (A classic RKHS point set distance is recovered when sigma_X = sigma_Y)

# 2) Let N(mu,sigma)(z) := (2*pi*sigma)^{-D/2} exp(-(z-mu)^2/2*sigma^2)
# View point set X (and similarly point set Y) as a (L1-normed) mixture of Gaussians:
#     fX(z) := 1/NX sum_{i=1}^NX N(x_i,sigmaX)(z)
#
# 3) Finally, define the "intrinsic" distance between point sets X and Y as
#     dist(X,Y) := || fX - fY ||_2^2
# To compute it, use the following result regarding L2 product of Gaussians:
#     <N(mu1,sigma1), N(mu2,sigma2)>_2 = N(mu1-mu2, sqrt(sigma1^2+sigma2^2))(0)
# (which is essentially the well-known convolution result on Gaussians)

def point_set_distance(X, Y, sigma_X=None, sigma_Y=None, w_X=None, w_Y=None):
    '''
    Compute "point set distance" between two point sets X and Y (provided as torch 2d tensors).

    - sigma_X [sigma_Y] = scale for point set X [Y]. If None, use "intrinsic scale" for this point set.

    - We can possibly use non-linear weights for point sets X (w_X) and/or point set Y (w_Y)
      If w_X = None, set w_X = 1/N_X (homogenous weights summing to 1) and similarly for w_Y

    Setting Z <-- (X,Y) [totality of the points, from point sets X and Y], we have
        L = \sum_i \sum_j c_{ij} N(z_i-z_j, sigma_{ij}) ;
    with
        - c_{ij} = 1/(N_i*N_j) where N_i and N_j are NX or NY, depending on the origin point sets of z_i and z_j
        - sigma_{ij} = sqrt(sigma_i**2+sigma_j**2) where sigma_i and sigma_j are sigma_X or sigma_Y (or sigma/sqrt(2))
    '''

    D = X.shape[1]

    sigma_X_int = intrinsic_scale(X)
    if sigma_X is None:
        sigma_X = sigma_X_int
    if sigma_X < sigma_X_int:
        warnings.warn("Required data distance scale `sigma_X` is smaller than 'intrinsic' scale for point set X. You should probably augment sigma_X.")

    sigma_Y_int = intrinsic_scale(Y)
    if sigma_Y is None:
        sigma_Y = sigma_Y_int
    if sigma_Y < sigma_Y_int:
        warnings.warn("Required data distance scale `sigma_Y` is smaller than 'intrinsic' scale for point set Y. You should probably augment sigma_Y.")

    if w_X is None:
        w_X = torch.ones(X.shape[0]) / X.shape[0]
    if w_Y is None:
        w_Y = torch.ones(Y.shape[0]) / Y.shape[0]

    sigma_XX = math.sqrt(2) * sigma_X
    sigma_YY = math.sqrt(2) * sigma_Y
    sigma_XY = math.sqrt(sigma_X ** 2 + sigma_Y ** 2)

    KXX = GaussKernel(sigma_XX, D).KRedScal
    KYY = GaussKernel(sigma_YY, D).KRedScal
    KXY = GaussKernel(sigma_XY, D).KRedScal

    cXX = 1 / ((2*math.pi)**(D/2) * sigma_XX**D)
    cYY = 1 / ((2*math.pi)**(D/2) * sigma_YY**D)
    cXY = 1 / ((2*math.pi)**(D/2) * sigma_XY**D)

    return cXX * (KXX(X, X, w_X).flatten() * w_X).sum() \
        + cYY * (KYY(Y, Y, w_Y).flatten() * w_Y).sum() \
        - 2 * cXY * (KXY(X, Y, w_Y).flatten() * w_X).sum()


#############################################################
# POINT SET DECIMATION
#############################################################

def decimate(x,R):
    '''
     Greedy decimation of point set x with radius R.
     A subset y of point set x is computed, such that every point in x is at distance <= R to some point in y.

    :param x: pytorch tensor of size (N,D)
    :param R: radius of the decimation
    :return: (kept, rejected).
    kept = indices of points in x to be kept in the decimated set ;
    rejected = indices of points in x NOT kept in the decimated set (just for convenience)
    '''

    x_i = x[:, None, :]  # (N, 1, 2)
    x_j = x[None, :, :]  # (1, N, 2)
    D = ((x_i - x_j) ** 2).sum(-1)
    M = (D <= R**2)

    N = x.shape[0]
    notcovered = list(range(N))
    kept = []

    while len(notcovered)>0 :
        # index of point with most neighbours...
        id = M[np.ix_(notcovered,notcovered)].sum(axis=0).argmax()  # use numpy for the ix_ routine :)
        id = notcovered[id]     # global index
        kept.append(id)
        neighbors = torch.where(M[id,:])[0].tolist()                # list of neighbours of id
        notcovered = [x for x in notcovered if x not in neighbors]  # remove from 'notcovered'

    # Done. Just for convenience, explicitly compute the complementary set of kept
    rejected = [i for i in range(N) if i not in kept]
    return kept, rejected


#########################################################################
# Testing
#########################################################################

if __name__ == '__main__':
    # Running as a script

    import matplotlib.pyplot as plt
    from diffICP.visualization.visu import my_scatter

    if False:
        x = torch.randn(100,2)
        R = 0.5
        kept, rejected = decimate(x,R)
        print(kept)
        my_scatter(x,color='b')
        my_scatter(x[kept,:],color='r')
        plt.show()