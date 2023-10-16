'''
Various small functions to handle data formats, input to functions, etc.
'''

import torch

def read_point_sets(x):
    '''
    :param x: list of data point sets to be read. Three possible formats:
        x = torch tensor of size (N, D) : single point set;
        x[k] = torch tensor of size (N[k], D) : point set from frame k;
        x[k][s] = torch tensor of size (N[k,s], D) : point set in structure s from frame k;
    :return x,K,S,D.
        x: point sets, always cast in the format x[k][s] (even when there is a single frame k and/or structure s) ;
        K = number of frames (= x.shape[0]) ;
        S = number of structures (= x[0].shape[0]) ;
        D = dimension of space (= x[0][0].shape[1]) ;
    '''

    if isinstance(x, torch.FloatTensor) or isinstance(x, torch.cuda.FloatTensor):
        # single point set (single frame and structure)
        x = [[x]]
    elif isinstance(x, list):
        if isinstance(x[0], torch.FloatTensor) or isinstance(x[0], torch.cuda.FloatTensor):
            # multiple frames / single structure
            x = [[xk] for xk in x]
        else:
            x = [xk.copy() for xk in x]  # copy x as a list of lists (does not copy the data point sets)
    else:
        raise ValueError("Wrong format for input x")

    # Number of frames
    K = len(x)

    # Number of structures
    allSs = list(set([len(xk) for xk in x]))
    if len(allSs) > 1:
        raise ValueError("All frames should have same number of structures")
    S = allSs[0]

    # Point set dimension
    allDs = list(set([xks.shape[1] for xk in x for xks in xk]))
    if len(allDs) > 1:
        raise ValueError("All point sets should have same axis-1 dimension")
    D = allDs[0]

    return x,K,S,D
