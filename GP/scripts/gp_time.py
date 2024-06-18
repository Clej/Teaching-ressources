"""
Plot 1D random fields (time series) with Gaussian priors with MOGPTK.
Help to understand Gaussian prior structure and kernel functions.
"""

import numpy as np
from matplotlib import pyplot as plt
from mogptk import Model as Model_mogptk
from mogptk import Exact
from mogptk import DataSet
from mogptk.gpr.singleoutput import SquaredExponentialKernel,\
    MaternKernel, PeriodicKernel, SincKernel, SpectralMixtureKernel, WhiteKernel
from mogptk.gpr.kernel import MulKernel, AddKernel
from mogptk.gpr.mean import LinearMean, ConstantMean
from torch import manual_seed
from utils import get_figsize, gen_1D_gp
from constants import columnwidth

manual_seed(3004666)

#-------------------#
# Data generation
#-------------------#
a, b = 0.0, 5.0
nt = 10**3
tt = np.linspace(a, b, num=nt)
N_gp = 10 # number of GP draws

d = 1
f = lambda ti: np.sin(np.abs(ti))
n = 900 # number of samples of a single signal
t_samples = np.random.uniform(a, b, (n, 1))
z_samples = np.array(
    [f(t) for t in t_samples[:,0]]
)[:,None]

# additive noise
z_samples = z_samples[:, 0]
z_samples +=  np.random.normal(loc=0.0, scale=0.05, size=n)
mogptk_dataset = DataSet([t_samples], [z_samples])

#----------------------#
#   Plot prior samples
#----------------------#
print('Plotting...')

# Main loop
## for squared_exponential
lengthscales = [2.0, 1.0, 0.1]

## for Locally periodic
per = [1.25, 0.5]

## for whitenoise
magnitude = [2.0]

for idx, l in enumerate(magnitude):

    #-------------------#
    #   Set a model
    #-------------------#
    # kernel = SincKernel(input_dims=1, active_dims=[0])
    # kernel.bandwidth.assign(value=[1.0], train=False)

    # kernel = MaternKernel(input_dims=1, active_dims=[0], nu=1.5)
    # kernel.lengthscale.assign(value=[l], train=False)

    # kernel = PeriodicKernel(input_dims=1, active_dims=[0])
    # kernel.period.assign(value=[l], train=False)

    # kernel = SquaredExponentialKernel(input_dims=1, active_dims=[0])
    # kernel.lengthscale.assign(value=[l], train=False)

    # kernel = SpectralMixtureKernel(input_dims=1, Q=2)

    kernel = WhiteKernel(input_dims=1)
    kernel.magnitude.assign(value=1.0, train=False)

    # kernel = SincKernel(input_dims=1, active_dims=[0])
    # kernel.bandwidth.assign(value=[l], train=False)

    model = Model_mogptk(mogptk_dataset, inference=Exact(), kernel=kernel, mean=ConstantMean())
    model.gpr.likelihood.scale.assign(value=1.0, train=False)

    #-------------------#
    #   Prior sampling
    #-------------------#
    prior_samples = gen_1D_gp(model=model, x=tt, N=N_gp)

    #-------------------#
    #   Plot
    #-------------------#
    # GP samples
    fig, ax = plt.subplots(
        1, 1,
        sharey=True,
        figsize=get_figsize(columnwidth=columnwidth)
    )
    ax.plot(
        tt,
        prior_samples,
        linestyle='solid', linewidth=0.75,
        alpha=0.75
    )

    ax.set_xlim(tt.min(), tt.max())
    ax.set_ylim(-3.0, 3.0)
    # ax.set_title(r"$\ell = {{{:.2f}}}$".format(l))
    ax.set_title(r"$ \sigma = {{{:.2f}}} $".format(l))
    ax.set_xlabel(r'$t$', fontsize='x-large')
    ax.set_ylabel(r'$y(t)$', fontsize='x-large')

    fig.savefig(
        './images/{:2d}_gp_time_{}_{:.2f}.pdf'.format(N_gp, kernel.name(), l),
        bbox_inches='tight',
        pad_inches = 0.1
    )