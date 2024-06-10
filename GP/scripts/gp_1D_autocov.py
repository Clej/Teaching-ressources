"""
Plot 1D kernel functions as autocovariances.
"""

import numpy as np
from matplotlib import pyplot as plt
from mogptk.gpr.singleoutput import SquaredExponentialKernel,\
    LocallyPeriodicKernel, MaternKernel, SincKernel, PeriodicKernel
from mogptk.gpr.kernel import MulKernel, AddKernel
from mogptk.gpr.mean import LinearMean, ConstantMean
from utils import get_figsize, kernel_autocov_1D
from constants import columnwidth

#----------------------#
# 1D inputs generation
#----------------------#
a, b = 0.0, 3.0
nt = 10**3
x = np.linspace(a, b, num=nt)

#-------------------#
#   Set kernels
#-------------------#
kernels = [SquaredExponentialKernel(input_dims=1),
           MaternKernel(input_dims=1, nu=1.5),
           PeriodicKernel(input_dims=1),
           SincKernel(input_dims=1)
           ]

#-------------------#
#   Plot
#-------------------#
fig, ax = plt.subplots(
    1, 1,
    sharey=True,
    figsize=get_figsize(columnwidth=columnwidth)
)
for i, ker in enumerate(kernels):

    K_autocov = kernel_autocov_1D(kernel=ker, x=x)

    ax.plot(
        x,
        K_autocov,
        linestyle='solid', linewidth=1.0,
        label=ker.name().replace('Kernel', ''),
        alpha=0.80
        # label=r"$p = {{{:.2f}}}, \ell = 1$".format(l)
    )

    ax.set_ylim((-1.25, 1.25))
    # ax.set_title(r"$p = {{{:.2f}}}, \ell = 1$".format(l))
    ax.set_xlabel(r'$|t - t^\prime|$', fontsize='x-large')
    ax.set_ylabel(r'$k(t, 0)$', fontsize='x-large')

    ax.legend(
        loc='lower right', bbox_to_anchor=(0.99, 0.99),
        labelspacing=0.1, ncols=2,
        frameon=False, fontsize='xx-small'
    )

fig.savefig(
    './images/autocov_gp_1D.pdf',
    bbox_inches='tight',
    pad_inches = 0.1
)
