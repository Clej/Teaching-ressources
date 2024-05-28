"""
Plot 1D Gaussian histogram and density.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(seed=3004666)

#----------------------#
#       Data gen
#----------------------#
n = 10**4
mu = 0.
std = 1.5
univ_gauss = norm(loc=mu, scale=std)
y = univ_gauss.rvs(size=n)

#----------------------#
#       Plot
#----------------------#
fig, ax = plt.subplots(1, 1)

# density
t = np.linspace(-6.0, 6.0, num=n)
ax.plot(
    t,
    univ_gauss.pdf(t),
    color='blue'
)
ax.vlines(x=mu, ymin=0, ymax=0.3, linewidth=3.5)

# histrogram samples
binwidth = 0.15
xymax = max(np.max(np.abs(y)), np.max(np.abs(y)))
lim = (int(xymax/binwidth) + 1) * binwidth
bins = np.arange(-lim, lim + binwidth, binwidth)
ax.hist(
    y,
    bins=bins,
    density=True,
    color='purple'
)

# legend
ax.set_xlabel(r'y', fontsize='x-large')
ax.text(x=-6, y=0.225, s=rf"$\mu = ${mu}", fontsize='x-large', color='C0')
ax.text(x=-6, y=0.20, s=rf"$\sigma = ${std}", fontsize='x-large', color='blue')

#
fig.savefig('./images/gaussian_1d.pdf')
