"""
Plot:
- a 2D Gaussian point cloud with a highgly correlated covariance,
- an nD Gaussian point cloud ad "values vs. index dimension" plot.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics.pairwise import rbf_kernel
from utils import scatter_hist, numpy_to_latex

np.random.seed(seed=3004666)
n = 10**4

#----------------------#
#       Data gen
#----------------------#
# small d
d = 5
mu = [0]*d
variance = 1.5
sigma_high = np.array(
    [[variance, 0.99, 0.98, 0.96, 0.94],
     [0.99, variance, 0.99, 0.98, 0.96],
     [0.98, 0.99, variance, 0.99, 0.98],
     [0.96, 0.98, 0.99, variance, 0.99],
     [0.94, 0.96, 0.98, 0.99, variance]]
)

gauss_anis_high = multivariate_normal(mean=mu, cov=sigma_high)
Y = gauss_anis_high.rvs(size=n)

#----------------------#
#       Plot
# 1st plot:
### A 2D Gaussian point clouds with N samples randomly selected
### restricted to two of their marginals (dimensions): y_1 vs. y_2
fig_2d, ax_2d = scatter_hist(
    x=Y[:, 0], y=Y[:, 1],
    margin_hist=False,
    fig=None
)

# Take N samples in Y, zoom-in and color them
N = 8
idx_rnd = np.arange(N) # easy but arbitrary, should be random
colors_N = [f'C{j}' for j in range(N)]

for i, idx in enumerate(idx_rnd):

    # 2D scatter plot: first and second dimension
    ax_2d.plot(
        Y[idx, 0], Y[idx, 1],
        linestyle='None',
        markersize=10,
        color=colors_N[i],
        label=f'Sample #{idx + 1}',
        marker='o', markeredgecolor='black'
    )

# legend
fig_2d.legend(loc='upper center', frameon=False, fontsize='small', ncol=4)
fig_2d.savefig(
    './images/gaussian_2d_2outof5.pdf',
    bbox_inches='tight', pad_inches = 0
)

# 2nd plot:
### Same N samples with all their marginals: 
### ordonate are values, abscissa are dimension index
fig_5d, ax_5d = plt.subplots(1, 1, figsize=fig_2d.get_size_inches())

# nD scatter plot: "values vs. index dimension"
for i, idx in enumerate(idx_rnd):
    ax_5d.plot(
        np.arange(d),
        Y[idx, :],
        linestyle='solid',
        color=colors_N[i],
        marker='x'
    )

ax_5d.set_ylim((-5, 5))
tcks = np.arange(d)
ax_5d.set_xticks(tcks, labels=[rf'$y_{j}$' for j in tcks], fontsize='x-large')
ax_5d.set_xlabel('Dimension')

fig_5d.savefig(
    f'./images/gaussian_5d_valuevsindex.pdf',
    bbox_inches='tight', pad_inches = 0
)

# 2nd plot (bis): same as above but with dimension 1 and 2 only
fig_5d, ax_5d = plt.subplots(1, 1, figsize=fig_2d.get_size_inches())

for i, idx in enumerate(idx_rnd):
    ax_5d.plot(
        np.arange(2),
        Y[idx, :2],
        linestyle='solid',
        color=colors_N[i],
        marker='x'
    )

ax_5d.set_ylim((-5, 5))
tcks = np.arange(d)
ax_5d.set_xticks(tcks, labels=[rf'$y_{j}$' for j in tcks], fontsize='x-large')
ax_5d.set_xlabel('Dimensions')

fig_5d.savefig(
    f'./images/gaussian_2d_valuevsindex.pdf',
    bbox_inches='tight', pad_inches = 0
)

# 3rd plot, d=50:
### Same N samples with all their marginals: 
### ordonate are values, abscissa are dimension index
d = 50
a = 8
t = np.linspace(-a, a, num=d)
mu = [0]*d
sigma_rbf = rbf_kernel(t.reshape(-1, 1), gamma=1.0)
# sigma_rbf[np.diag_indices(d)] *= variance
# sigma_rbf /= 10.0

# print(numpy_to_latex(sigma_rbf, 3))

gauss_rbf = multivariate_normal(mean=mu, cov=sigma_rbf)
Y = gauss_rbf.rvs(size=N)

fig_100d, ax_100d = plt.subplots(1, 1, figsize=fig_2d.get_size_inches())

for i in range(N):
    ax_100d.plot(
        np.arange(d),
        Y[i, :],
        linestyle='solid',
        color=colors_N[i],
        marker='x'
    )

# ax_nd.set_ylim((-5, 5))
tcks = np.arange(d)[::5]

ax_100d.set_xticks(
    tcks,
    labels=[r"$y_{{{:2d}}}$".format(j) for j in tcks],
    fontsize='x-large'
)
ax_100d.set_xlabel('Dimensions')
fig_100d.savefig(
    f'./images/gaussian_{d}d_valuevsindex.pdf',
    bbox_inches='tight', pad_inches = 0
)