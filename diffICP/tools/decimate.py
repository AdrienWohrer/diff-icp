
import numpy as np
import torch
import matplotlib.pyplot as plt
from diffICP.visualization.visu import my_scatter

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

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