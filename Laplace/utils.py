"""
Author: Minglang Yin, minglang_yin@brown.edu
"""
import numpy as np
import scipy
import torch
import argparse
import scipy.io as io
import matplotlib.pyplot as plt
# from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from scipy.linalg import svd

def ParseArgument():
    parser = argparse.ArgumentParser(description='DeepONet')
    parser.add_argument('--epochs', type=int, default=50000, metavar='N',
                        help = 'number of epochs to train (default: 1000)')
    parser.add_argument('--device', type=str, default='cuda', metavar='N',
                        help = 'computing device (default: GPU)')
    parser.add_argument('--save-step', type=int, default=10000, metavar='N',
                        help = 'save_step (default: 10000)')
    parser.add_argument('--restart', type=int, default=0, metavar='N',
                        help = 'if restart (default: 0)')
    parser.add_argument('--test-model', type=int, default=0, metavar='N',
                        help = 'default training, testing as 1')

    args = parser.parse_args()
    return args

def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or '\
            'np.ndarray, but got {}'.format(type(input)))


def SVD_reduce(Ax, Ay, mode):
    # Singular-value decomposition
    Ux, sx, VxT = svd(Ax)
    Uy, sy, VyT = svd(Ay)

    # Truncate sigma
    Sigmax = np.zeros((Ax.shape[0], Ax.shape[1]))
    Sigmax[:Ax.shape[1], :Ax.shape[1]] = np.diag(sx)

    Sigmay = np.zeros((Ay.shape[0], Ay.shape[1]))
    Sigmay[:Ay.shape[1], :Ay.shape[1]] = np.diag(sy)
    
    #
    Ux = Ux[:, :mode]
    Sigmax = Sigmax[:mode, :]

    Uy = Uy[:, :mode]
    Sigmay = Sigmay[:mode, :]

    # reconstruct
    Bx = Ux.dot(Sigmax.dot(VxT))
    By = Uy.dot(Sigmay.dot(VyT))

    ## Figure, plot energy mode
    fig = plt.figure(constrained_layout=False, figsize=(10, 5))
    gs = fig.add_gridspec(1, 2)

    ## Energy in x
    ax = fig.add_subplot(gs[0])
    EnergySqr = sx**2
    ax.scatter(np.linspace(1, Ax.shape[1], Ax.shape[1]), EnergySqr/sum(EnergySqr), color='red', label='Mode Energy %')
    ax.plot(np.linspace(1, Ax.shape[1], Ax.shape[1]), EnergySqr/sum(EnergySqr), color='red')
    ax.axvline(x=mode, color='blue', linestyle='--', label='cut off mode = 10')
    ax.set_title(r"Energy fraction ($\sigma^{2}_{i}/\Sigma_{j}\sigma^{2}_{j}$) of $\Delta x$")
    ax.set_xlabel("POD Mode")
    ax.set_ylabel("Energy Fraction")
    # ax.set_xlim([0, 20])
    ax.legend()
    ax.set_yscale("log")

    ## Energy in y
    ax = fig.add_subplot(gs[1])
    EnergySqr = sy**2
    ax.scatter(np.linspace(1, Ay.shape[1], Ay.shape[1]), EnergySqr/sum(EnergySqr), color='blue', label='Mode Energy %')
    ax.plot(np.linspace(1, Ay.shape[1], Ay.shape[1]), EnergySqr/sum(EnergySqr), color='blue')
    ax.axvline(x=mode, color='r', linestyle='--', label='cut off mode = 10')
    ax.set_title(r"Energy fraction ($\sigma^{2}_{i}/\Sigma_{j}\sigma^{2}_{j}$) of $\Delta y$")
    ax.set_xlabel("POD Mode")
    ax.set_ylabel("Energy Fraction")
    # ax.set_xlim([0, 20])
    ax.legend()
    ax.set_yscale("log")

    fig.savefig('POD_mode_frac.png')
    plt.close()

    ##
    coeff_x = Sigmax.dot(VxT)
    coeff_y = Sigmay.dot(VyT)

    return Ux, coeff_x, Uy, coeff_y