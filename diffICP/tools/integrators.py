'''
Helper functions : various ODE integrators.

They take the form
    final_x = my_integrator(ODESystem, x0, nt, deltat)
where
    - The variables -- x0, final_x, etc. -- are TUPLES of torch tensors. Even when the ODE applies to a single
      tensor x, it should be wrapped inside a tuple, i.e.: x0 = (x,)
    - ODESystem is the function such that xdot = ODESystem(*x)   -- where xdot and x are tuples of torch tensors.
    - nt is the number of discretizations used by the numerical scheme
    - deltat is the final time (i.e., integrate from t=0 to deltat)
'''

# Adapted from the LDDMM demo script of library KeOps :
# https://www.kernel-operations.io/keops/_auto_tutorials/surface_registration/plot_LDDMM_Surface.html
# A Wohrer, 2023

import torch

def EulerIntegrator(ODESystem, x0, nt=11, deltat=1.0):
    x = tuple(map(lambda x: x.clone(), x0))     # Nota : clone() transmits the computational graph
    dt = deltat / nt
    l = [x]
    for i in range(nt):
        # print("integration, step ",i)
        xdot = ODESystem(*x)
        x = tuple(map(lambda x,xdot:
                       x + dt * xdot,
            x,xdot))
        l.append(x)
    return l

#####


def RalstonIntegrator(ODESystem, x0, nt=11, deltat=1.0):
    x = tuple(map(lambda x: x.clone(), x0))     # Nota : clone() transmits the computational graph
    dt = deltat / nt
    l = [x]
    for i in range(nt):
        # print("integration, step ",i)
        xdot = ODESystem(*x)
        xi = tuple(map(lambda x,xdot:
                       x + (2 * dt / 3) * xdot,
            x,xdot))
        xdoti = ODESystem(*xi)
        x = tuple(map(lambda x,xdot,xdoti:
                      x + (0.25 * dt) * (xdot + 3 * xdoti),
                x,xdot,xdoti))
        l.append(x)
    return l


