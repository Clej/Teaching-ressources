"""
Plot 2D Gaussian point clouds with multiple covariances.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm

np.random.seed(seed=3004666)
n = 10**4
d = 2

#----------------------#
#       Data gen
#----------------------#
mu = [0]*d
rho_low, rho_high = -0.3, 1.1
variances = [1.5]*d
sigma_high = np.array(
    [[variances[0], rho_high],
    [rho_high, variances[1]]]
)
sigma_low = np.array(
    [[variances[0], rho_low],
    [rho_low, variances[1]]]
)

gauss_2d_anis_low = multivariate_normal(mean=mu, cov=sigma_low)
gauss_2d_anis_high = multivariate_normal(mean=mu, cov=sigma_high)
gauss_2d_iso = multivariate_normal(mean=mu, cov=np.eye(d)*variances[0])

#----------------------#
#     Plot functions
#----------------------#
from utils import scatter_hist

t = np.linspace(-6.0, 6.0, num=n)

def plot_pdf_joint(pdf_fun, ax, fig):
    """Plot (full) 2d density."""


def plot_pdf_marginals(ax_x, ax_y, fig):
    """Plot marginal densities."""

    ax_x.plot(t, norm.pdf(t, loc=mu[0], scale=np.sqrt(variances[0])), color='blue', label='marginal')
    # add vlines at mu and sigma
    ax_y.plot(norm.pdf(t, loc=mu[1], scale=np.sqrt(variances[1])), t, color='blue')
    return fig

def confidence_ellipse(mu, cov, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    """

    from matplotlib.patches import Ellipse
    from matplotlib import transforms

    if cov.shape != (2,2):
        raise ValueError("cov must be a nump array with shape (2, 2).")

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_x, mean_y = mu[0], mu[1]

    # scale to our covariance matrix
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)

    #draft
    _, eigvect = np.linalg.eig(cov)
    maj_ax, min_ax = eigvect[0], eigvect[1]
    ax.quiver(
        tuple(mu), tuple(mu),
        min_ax, maj_ax,
        scale=1/n_std,
        scale_units='xy', angles='xy',
        color='red'
    )

    return ax

#----------------------#
#     Main loop
#----------------------#
for rho_str, gauss_2d in zip(
    ['rho_high', 'rho_low', 'rho_null'],
    [gauss_2d_anis_high, gauss_2d_anis_low, gauss_2d_iso]
    ):

    # sampling
    Y = gauss_2d.rvs(size=n)
    mu, cov = gauss_2d.mean, gauss_2d.cov
    rho = gauss_2d.cov[0, 1]

    # plot sampled data
    fig, ax_scatt, ax_x, ax_y = scatter_hist(
        x=Y[:, 0],
        y=Y[:, 1]
    )

    # add marginal (true) densities
    fig = plot_pdf_marginals(ax_x, ax_y, fig)

    # add ellipse and covariance eigeven vectors
    ax_anis = confidence_ellipse(
        mu=mu, cov=cov, n_std=2.0,
        ax=ax_scatt, edgecolor='red'
    )

    # add mean distribution point
    ax_scatt.scatter(mu[0], mu[1], s=2**6, color='C0')
    
    # legend
    ax_scatt.set_xlim((-7.0, 7.0))
    ax_scatt.set_ylim((-7.0, 7.0))
    ax_scatt.text(x=-5.0, y=4, s=rf"$\mu = {mu}^\top$", fontsize='x-large', color='C0')
    ax_scatt.text(x=-5.0, y=5, s=rf"$\rho = {rho}$", fontsize='x-large', color='red')
    ax_scatt.text(x=-5.0, y=6, s=rf"diag($\Sigma) = ${np.diag(cov)}", fontsize='x-large', color='blue')

    # fig_anis.legend(loc='upper left')
    fig.savefig(f'./images/gaussian_2d_{rho_str}.pdf')
