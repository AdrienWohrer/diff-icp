'''
Gradient-based optimization procedure, used in different places of the module.
This is basically a wrapper for Pytorch's L-BFGS optimization procedure.
'''

import os, time, math, copy
import warnings
import torch

def LBFGS_optimization(p0, lossfunc, nmax=10, tol=1e-3, errthresh=1e8):
    '''
    Wrapper for pytorch's L-BFGS optimization (advanced gradient descent-like algorithm).

    :param p0: list [a0, b0, c0, ...] with a0, b0, ... = pytorch tensors = initial parameter values
    :param lossfunc: loss function, in the form L = lossfunc(a,b,c,...)
    :param nmax: maximum number of optimizer steps (pytorch L-BFGS optimizer)
    :param tol: stopping tolerance (on Delta L / L)
    :param errthresh: threshold on L for error reporting
    :return: p = [a,b,c,...] (optimal parameters found), nsteps (number of required optimizer steps), change (on loss function)
    '''

    # Variables to optimize (detach, and create new gradient graph)
    p = [ a.clone().contiguous().detach().requires_grad_(True) for a in p0 ]

    # Optimizer over this variable
    optimizer = torch.optim.LBFGS(p, max_iter=20, max_eval=100, history_size=100, line_search_fn="strong_wolfe")
    # optimizer = torch.optim.Adam(p, lr=0.001)     # try stuff... meh
    # optimizer = torch.optim.SGD(p, lr=0.00001)     # try stuff... crap
    # optimizer = torch.optim.RMSprop(p)     # try stuff... meh

    # Keep track of the optimizer's repeated function evaluations during all the procedure (for manual debug and handling of exceptions)
    iter_L, best_L, best_p = [], math.inf, None

    def closure():
        # Reset grads to 0 between every call to function .backward()
        optimizer.zero_grad()
        # Compute new loss
        L = lossfunc(*p)
        Ld = L.detach().item()      # manual tracking of function evaluations and parameters
        iter_L.append(Ld)           # list of all values of L encountered during optimization
        nonlocal best_L, best_p     # https://stackoverflow.com/questions/64323757/why-does-python-3-8-0-allow-to-change-mutable-types-from-enclosing-function-scop
        if Ld < best_L:
            best_L = Ld
            best_p = [ a.clone().detach() for a in p ]
        # Back-propagate
        L.backward()                # Update gradients of all the chain of computations from p0 to L
        return L

    i, keepOn, L = 0, True, math.inf
    while i < nmax and keepOn:
        i += 1
        p_prev = [ a.clone().detach() for a in p ]
        # For closure-based optimizer (like LBFGS)
        optimizer.step(closure)                         # !!! The long line !!!
        Lprev, L = L, iter_L[-1]                # value of L before / after optimizer step
        # For single-evaluation optimizers (like Adam, SGD...)
        # L = closure()
        # optimizer.step()

        if L > Lprev or L > errthresh or math.isnan(L):
            # Detect some form of divergent behavior!

            # Print some debug information
            if math.isnan(L):
                print("WARNING: NaN value for loss L during L-BFGS optimization.")
            elif L > errthresh:
                print("WARNING: Aberrantly large value for loss L during L-BFGS optimization.")
            elif L > Lprev:
                print("WARNING: Increase of loss L during L-BGFS optimization.")
            # print(f"iter {i} , all L values during optimization : {iter_L}")
            # print(f"iter {i} , best L value during optimization : {best_L}")
            # print(f"iter {i} , last L value during optimization : {L}")

            # Use some fallback value for p0
            if best_L < Lprev:
                # Some better p0 value than p0prev has been encoutered during the optimizer step --> use it.
                p = [ a.clone() for a in best_p ]
                L = best_L
                print(
                    "L-BFGS optimization. Found an intermediate 'best_p' value for this iteration.")
            else:
                # Found no better value than p_prev. Simply add some noise to p_prev.
                # TODO: strategy in this case should be fixed externally (by the calling function)
                rmod = 0.01
                p = [ a + rmod * a.std() * torch.randn(a.shape, dtype=a.dtype, device=a.device) for a in best_p ]
                L = lossfunc(*p)
                print(
                    f"L-BFGS optimization. Trying a random perturbation of parameter from its current value, with relative strength {rmod}.")
                # print(f"best_p: {best_p}")
                # print(f"p: {p}")
                # print(f"L: {L}")

            # NEW VERSION : we do not exit the optimizer loops. But instead relaunch an optimizer with value p, and
            # without line search (maybe more robust?)
            change = "None (divergent iteration step)"
            p = [ a.requires_grad_(True) for a in p ]
            optimizer = torch.optim.LBFGS(p, max_iter=20, max_eval=100, history_size=100, line_search_fn=None)

        else:  # normal behavior
            # changes in parameter value :
            changes = [((a - a_prev) ** 2).mean().sqrt().detach().cpu().numpy() for a,a_prev in zip(p,p_prev)]
            # reference values :
            refs = [(a_prev ** 2).mean().sqrt().detach().cpu().numpy() for a_prev in p_prev]
            keepOn = any( change > tol*ref for change,ref in zip(changes,refs) )
            change = max(changes)

    # Done !
    best_p = [ a.detach() for a in best_p ]
    nsteps = i
    return best_p, best_L, nsteps, change


##############

# Old version, kept for a while
# (iter_L and best_L were reinitialized at each optimizer step)

def LBFGS_optimization_old(p0, lossfunc, nmax=10, tol=1e-3, errthresh=1e8):
    '''
    Wrapper for pytorch's L-BFGS optimization (advanced gradient descent-like algorithm).

    :param p0: list [a0, b0, c0, ...] with a0, b0, ... = pytorch tensors = initial parameter values
    :param lossfunc: loss function, in the form L = lossfunc(a,b,c,...)
    :param nmax: maximum number of optimizer steps (pytorch L-BFGS optimizer)
    :param tol: stopping tolerance (on Delta L / L)
    :param errthresh: threshold on L for error reporting
    :return: p = [a,b,c,...] (optimal parameters found), nsteps (number of required optimizer steps), change (on loss function)
    '''

    # Variables to optimize (detach, and create new gradient graph)
    p = [ a.clone().contiguous().detach().requires_grad_(True) for a in p0 ]

    # Optimizer over this variable
    optimizer = torch.optim.LBFGS(p, max_iter=20, max_eval=100, history_size=100, line_search_fn="strong_wolfe")
    # optimizer = torch.optim.Adam(p, lr=0.001)     # try stuff... meh
    # optimizer = torch.optim.SGD(p, lr=0.00001)     # try stuff... crap
    # optimizer = torch.optim.RMSprop(p)     # try stuff... meh

    # Keep some track of the optimizer's repeated function evaluations during this step (for manual debug and handling of exceptions)
    iter_L, best_L, best_p = [], None, None

    def closure():
        # Reset grads to 0 between every call to function .backward()
        optimizer.zero_grad()
        # Compute new loss
        L = lossfunc(*p)
        # Back-propagate
        L.backward()                # Update gradients of all the chain of computations from p0 to L
        Ld = L.detach().item()      # manual tracking of function evaluations and parameters
        iter_L.append(Ld)           # list of all values of L encountered during current optimizer step
        nonlocal best_L, best_p     # https://stackoverflow.com/questions/64323757/why-does-python-3-8-0-allow-to-change-mutable-types-from-enclosing-function-scop
        if Ld < best_L:
            best_L = Ld
            best_p = [ a.clone().detach() for a in p ]
        return L

    i, keepOn = 0, True
    while i < nmax and keepOn:
        i += 1
        p_prev = [ a.clone().detach() for a in p ]
        iter_L, best_L, best_p = [], math.inf, None     # keep track of evaluations during optimizer.step
        # For closure-based optimizer (like LBFGS)
        optimizer.step(closure)                         # !!! The long line !!!
        Lprev, L = iter_L[0], iter_L[-1]                # value of L before / after optimizer step
        # For single-evaluation optimizers (like Adam, SGD...)
        # L = closure()
        # optimizer.step()

        if L > Lprev or L > errthresh or math.isnan(L):
            # Detect some form of divergent behavior!

            # Print some debug information
            if math.isnan(L):
                print("WARNING: NaN value for loss L during L-BFGS optimization.")
            elif L > errthresh:
                print("WARNING: Aberrantly large value for loss L during L-BFGS optimization.")
            elif L > Lprev:
                print("WARNING: Increase of loss L during L-BGFS optimization.")
            print(f"iter {i} , all L values during iteration : {iter_L}")
            print(f"iter {i} , best L value during iteration : {best_L}")
            print(f"iter {i} , last L value during iteration : {L}")

            # Use some fallback value for p0
            if best_L < Lprev:
                # Some better p0 value than p0prev has been encoutered during the optimizer step --> use it.
                p = best_p
                L = best_L
                print(
                    "Exiting current L-BFGS optimization. Found an intermediate 'best_p' value to use instead.")
            else:
                # Found no better value than p_prev. Simply some noise to p_prev.
                # TODO: strategy in this case should be fixed externally (by the calling function)
                rmod = 0.01
                p = [ a + rmod * a.std() * torch.randn(a.shape, dtype=a.dtype, device=a.device) for a in p ]
                L = lossfunc(*p)
                print(
                    f"Exiting current L-BFGS optimization. Trying a random perturbation of parameter from its current value, with relative strength {rmod}.")

            change = "STOP"
            keepOn = False  # exit the iterations (optimization has failed)

        else:  # normal behavior
            # changes in parameter value :
            changes = [((a - a_prev) ** 2).mean().sqrt().detach().cpu().numpy() for a,a_prev in zip(p,p_prev)]
            # reference values :
            refs = [(a_prev ** 2).mean().sqrt().detach().cpu().numpy() for a_prev in p_prev]
            keepOn = any( change > tol*ref for change,ref in zip(changes,refs) )
            change = max(changes)

    p = [ a.detach() for a in p ]
    nsteps = i
    return p, L, nsteps, change
