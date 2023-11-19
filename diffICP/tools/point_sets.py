'''
Miscellaneous helper functions regarding the structure and manipulation of point sets.
'''

import numpy as np
import torch
from pykeops.torch import LazyTensor
import matplotlib.pyplot as plt
from diffICP.visualization.visu import my_scatter

################################################

def intrinsic_scale(X):
    '''
    Compute an "intrinsic scale" sigma_X for point set X (provided as a torch 2d tensor).
    sigma_X is defined as the mean distance from each point to its nearest neighbor in the set.
    Intuitively, sigma_X represents the minimal amount of blurring necessary to "stop seeing" the point set structure.
    '''

    # TODO any cleaner way of doing this ?

    X_i = LazyTensor(X[:, None, :])
    X_j = LazyTensor(X[None, :, :])
    D_ij = ((X_i - X_j) ** 2).sum(-1)
    min_D_i = D_ij.Kmin(2, dim=1)[:,1] # (=second smallest ; the smallest is always 0, i.e., each point with itself)
    sigma_X = min_D_i.mean().sqrt()
    return 1 * sigma_X.item()     # multiplicative factor, close to 1, is ad hoc. TODO: probably depends on dimension D of the point sets (think!)


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

# Testing
if __name__ == '__main__':
    # Running as a script

    x = torch.randn(100,2)
    R = 0.5
    kept, rejected = decimate(x,R)
    print(kept)
    my_scatter(x,color='b')
    my_scatter(x[kept,:],color='r')
    plt.show()