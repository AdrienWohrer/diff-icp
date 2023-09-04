'''
Build and visualize a 2d grid, and possibly deform it through a Registration model
'''

import os, time, math

import numpy as np
from matplotlib import pyplot as plt
import torch
from diffICP.spec import defspec
from diffICP.registrations import Registration

class Gridlines:

    # --------------
    def __init__(self, xticks, yticks, points_per_line=100):
        """Rectangular initialization of a Gridlines object, with a given precision (number of points per line)"""

        self.xticks = xticks
        self.yticks = yticks
        self.points_per_line = points_per_line
        self.lines = []
        # Create vertical lines
        for x in xticks:
            newline_x = np.array([x] * points_per_line)
            newline_y = np.linspace(yticks.min(), yticks.max(), points_per_line)
            self.lines.append(np.stack((newline_x, newline_y),axis=1))
        # Create horizontal lines
        for y in yticks:
            newline_y = np.array([y] * points_per_line)
            newline_x = np.linspace(xticks.min(), xticks.max(), points_per_line)
            self.lines.append(np.stack((newline_x, newline_y),axis=1))

    # --------------
    def plot(self, **kwargs):
        """plot a Gridlines object, with plotting options given in **kwargs"""

        for line in self.lines:
            plt.plot(line[:,0], line[:,1], **kwargs)

    # --------------
    def get_points(self):
        """return all points in a Gridlines object, as a single 2d np array"""

        return np.concatenate(self.lines, axis=0)

    # --------------
    @classmethod
    def from_points(cls, points:np.ndarray, points_per_line:int):
        """build a Gridlines object from a single list of points (2d np array), and a known number of points per line"""

        gl = Gridlines([], [], points_per_line)
        gl.lines = np.split(points, points.shape[0]/points_per_line, axis=0)
        return gl

    # --------------
    def register(self, registration:Registration, backward=False):
        """register a Gridlines object using a Registration object. Outputs a new Gridlines object."""

        # load all grid points, as a suitable torch tensor
        gridpoints = torch.tensor(self.get_points(), **defspec)
        # apply the registration
        if backward:
            regpoints = registration.backward(gridpoints)
        else:
            regpoints = registration.apply(gridpoints)

        # regpoints = PSR.register(gridpoints)
        # back to a Gridlines object
        return Gridlines.from_points(regpoints.numpy(), self.points_per_line)

    # --------------
    def shoot(self, registration:Registration, require_v=False, backward=False):
        """shoot a Gridlines object using a Registration object. Outputs a list of Gridlines (one per time step).
        If require_v=True, also compute and return a list of vector fields (one per time step) corresponding to the time derivatives of the flow.
        The vector field is only evaluated at the grid intersections given by xticks and yticks"""

        # load all grid points, as a suitable torch tensor
        gridpoints = torch.tensor(self.get_points(), **defspec)
        # apply the shooting
        shoot = registration.shoot(gridpoints, backward=backward)
        # back to a Gridlines objects
        shootgrids = [Gridlines.from_points(tup[3].numpy(), self.points_per_line) for tup in shoot]
        # also compute vector fields = speeds at the grid intersections
        if require_v:
            intersec = np.stack(np.meshgrid(self.xticks, self.yticks), axis=2)                          # grid intersection (shape (Nx,Ny,2))
            intersec = torch.tensor(intersec.reshape((-1,2),order='F'), **defspec).contiguous()         # convert to torch tensor (shape (Nx*Ny,2))
            print(intersec.shape)
            shoot = registration.shoot(intersec, backward=backward)               # shoot
            intersecs_t = [ tup[3].numpy() for tup in shoot ]
            speeds_t = [ registration.LMi.ODE(*tup)[3].numpy() for tup in shoot ]
            return shootgrids, intersecs_t, speeds_t
        else:
            return shootgrids

### Testing

if False:
    plt.ion()
    savefigs = False
    savename = 'grid_diffeo'
    format = 'png'
    backward = False
    with_arrows = True

    from matplotlib.ticker import FormatStrFormatter

    gridlines = Gridlines(np.linspace(-0.5,1.5,30), np.linspace(-0.5,1.5,30))

    ### For comparison : first plot a "non-diffeomorphism"

    # fig = plt.figure()
    # from diffICP.kernel import GaussKernel
    # GK = GaussKernel(0.4,2)
    # N = 10                  # number of support points
    # q = torch.rand((N,2))   # random support points
    # p = torch.randn((N,2))   # random momenta
    #
    # gridpoints = torch.tensor(gridlines.get_points(), **defspec)                    # grid points as torch array
    # gridpoints += 0.2 * GK.KRed(gridpoints, q,p)                                      # add a (large enough) multiple of the vector field
    # newgrid = Gridlines.from_points(gridpoints, gridlines.points_per_line)          # back to Gridlines object
    #
    # newgrid.plot(color='blue', linewidth=1)
    # plt.xticks(np.arange(-0.5, 1.5 + 0.1, 0.5))
    # plt.yticks(np.arange(-0.5, 1.5 + 0.1, 0.5))
    # plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))  # Ensure correctly formated ticks
    # plt.gca().set_aspect('equal')
    # plt.gca().autoscale(tight=True)
    # if True:
    #     plt.savefig(f"figs/grid_not_diffeo.{format}", format=format, bbox_inches='tight')
    # plt.pause(1)
    # exit()

    ### Transform grid by a diffeomorphism (create one, or simply load one, as here)

    from diffICP.spec import CPU_Unpickler
    loadfile = "saving/test_basic.pkl"
    with open(loadfile, 'rb') as f:
        yo = CPU_Unpickler(f).load()        # modified dill Unpickler (see diffPSR.spec.CPU_Unpickler)
    reg = yo["PSR"].Registration()
    amplif = 2                              # modify strength of a0 (for testing)
    reg.a0 *= amplif
    reglines = gridlines.register(reg, backward=backward)

    # plt.figure()
    # reglines.plot(color='blue', linewidth=1)
    # plt.axis('equal')
    # if savefigs:
    #     plt.savefig(f"figs/{savename}_final.{format}", format=format)

    plt.pause(.1)

    fig = plt.figure()
    shootgrids, intersecs, speeds = gridlines.shoot(reg, require_v=True, backward=backward)
    for t,grid in enumerate(shootgrids):
        grid.plot(color='blue', linewidth=1)
        if with_arrows:
            plt.quiver(intersecs[t][:,0], intersecs[t][:,1], speeds[t][:,0], speeds[t][:,1], scale=amplif/1.3, color='red', width=0.005)
        plt.xticks(np.arange(-0.5, 1.5+0.1, 0.5))
        plt.yticks(np.arange(-0.5, 1.5+0.1, 0.5))
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))    # Ensure correctly formated ticks
        plt.gca().set_aspect('equal')
        plt.gca().autoscale(tight=True)
        if savefigs:
            plt.savefig(f"figs/{savename}_{t}.{format}", format=format, bbox_inches='tight')
        plt.pause(.1)
        plt.clf()

    input()
