"""
Simulate 2D random fields with Gaussian priors with MOGPTK.
Help to understand Gaussian prior structure and kernel functions.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import CenteredNorm
from mogptk import Model as Model_mogptk
from mogptk import Exact
from mogptk import DataSet
from mogptk.gpr.singleoutput import SquaredExponentialKernel
from mogptk.gpr.mean import ConstantMean
from torch import from_numpy, tensor, detach
from constants import columnwidth
from utils import get_figsize

#-------------------#
# Data generation
#-------------------#
# create a meshgrid in 2D
a, b = -np.pi/2, np.pi/2
nx, ny = 60, 60
[xx, yy] = np.meshgrid(
    np.linspace(a, b, nx),
    np.linspace(a, b, ny),
    indexing='xy'
)

# fig_grid, axs = plt.subplots(1, 2)
# axs[0].matshow(xx, cmap='RdBu_r'); axs[0].set_title('xx')
# im = axs[1].matshow(yy, cmap='RdBu_r'); axs[1].set_title('yy')
# fig_grid.colorbar(im, ax=axs[1])
# fig_grid.savefig('./xx_yy_grid.png')

# useless: just for the model __init__
f = lambda xi, xj: np.sin(xi) * np.tan(xj)
n = 500
xy_samples = np.random.uniform(a, b, (n, 2))
z_samples = np.array(
    [f(x1,x2) for (x1,x2) in zip(xy_samples[:,0], xy_samples[:,1])]
)[:,None]
# additive noise
z_samples = z_samples[:, 0]
z_samples +=  np.random.normal(loc=0.0, scale=0.05, size=n)

mogptk_dataset = DataSet([xy_samples], [z_samples])

#-------------------#
#       Model
#-------------------#
# Choose a kernel
kernel = SquaredExponentialKernel(input_dims=2, active_dims=[0, 1])
print(kernel)
kernel.lengthscale.assign(value=[0.5, 0.5], train=False)

print(f'input dims kernels {kernel.input_dims}')

# Choose a likelihood inference
model = Model_mogptk(mogptk_dataset, inference=Exact(), kernel=kernel, mean=ConstantMean())
model.gpr.likelihood.scale.assign(value=1.0, train=False)

print(f'#inputs {model.dataset.get_input_dims()}')
print(model.print_parameters())

#-------------------#
#   Prior sampling
#-------------------#
# convert to 2D array of (2D) coordinates
Z = np.vstack([xx.flatten(), yy.flatten()]).T
print(f"Z shape {Z.shape}")
print("Sampling...")
prior_samples = model.gpr.sample_f(Z=Z, n=None, prior=True)
print(f"prior samples shape {prior_samples.shape}")

# #-------------------#
# #   Kernel
# #-------------------#
# print('Kernel evaluation...')

# K = kernel.K(X1=from_numpy(Z), X2=from_numpy(np.zeros((1, 2))))
# K = detach(K).numpy().reshape(xx.shape)
# print(f'K shape {K.shape}')

# # tau = kernel.distance(from_numpy(Z[:4]))
# # tau = tau.numpy()
# # print(f'tau {tau}')

# K_mat = kernel.K(X1=from_numpy(Z).detach(), X2=None)
# K_mat = detach(K_mat).numpy()
# print(f'K_mat shape {K_mat.shape}')

#-------------------#
#   Plot prior
#-------------------#
print('Plot...')
fig, ax = plt.subplots(1, 1, figsize=get_figsize(columnwidth=columnwidth, wf=0.5, hf=0.5))
im = ax.pcolormesh(
    xx, yy,
    prior_samples.reshape(xx.shape),
    cmap='PiYG', norm=CenteredNorm(vcenter=0.0)
)
ax.set_xticklabels([]); ax.set_yticklabels([])
ax.set_xlabel(r'$\mathcal{X}_1$'); ax.set_ylabel(r'$\mathcal{X}_2$')
ax.set_title(r"Un GP sur $\mathcal{X}_1 \times \mathcal{X}_2 = \mathbb{R}^2 \to \mathbb{R}$", fontsize="small")
fig.colorbar(im, ax=ax).set_label(label="Température", fontsize="small")


# #-------------------#
# #   Plot kernel
# #-------------------#
# im = ax[1].matshow(
#     K_mat,
#     cmap='RdBu_r', norm=CenteredNorm(vcenter=0.0)
# )
# ax[1].set_title('Kernel matrix')

# im = ax[2].pcolormesh(
#     xx, yy,
#     K,
#     shading='nearest',
#     cmap='RdBu_r', norm=CenteredNorm(vcenter=0.0)
# )
# # im = ax[1].contour(
# #     xx, yy,
# #     K
# # )
# fig.colorbar(im, ax=ax[2])
# ax[2].set_title('Kernel function')

# for axis in ax:
#     axis.set_aspect('equal', adjustable='box')

fig.savefig(
    './images/{:1d}_gp_space_{}.pdf'.format(1, kernel.name()),
    bbox_inches='tight',
    pad_inches = 0.1
)