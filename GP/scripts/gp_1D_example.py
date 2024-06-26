import numpy as np
from matplotlib import pyplot as plt
from mogptk import Model as Model_mogptk
from mogptk import Exact
from mogptk import LoadFunction
from mogptk.gpr.mean import ConstantMean
from mogptk.gpr.singleoutput import MaternKernel, SquaredExponentialKernel
from torch import manual_seed
from utils import get_figsize, gen_1D_gp
from constants import columnwidth

manual_seed(3004666)

#-------------------#
# Data generation
#-------------------#
# input space
a, b = 0.0, 1.5
b_forecast = 3.0
nt = 10**3
tt = np.linspace(a, b, num=nt)
tt_forecast = np.r_[tt, np.arange(b, b_forecast, step=tt[1]-tt[0])]
N_gp = 10 # number of GP draws

# output space
g = lambda ti: -1 * np.cos(3*np.pi*ti) + 0.25 * np.sin(8*np.pi*ti)
n = 50
mogptk_dataset = LoadFunction(
    f=g,
    start=a, end=b,
    n=n,
    var=0.0,
    name='sin+cos'
)

mogptk_dataset.remove_randomly(pct=0.70)
mogptk_dataset.set_prediction_data(X=[0.80])

x_tr, y_tr = mogptk_dataset.get_train_data()
n_tr = y_tr.size
x_0 = mogptk_dataset.get_prediction_data()
x_0 = x_0[0]
y_0 = g(x_0)

#-------------------#
#   Train/predict
#-------------------#
mean = ConstantMean()
mean.bias.assign(value=[0.0], train=False)

kernel = SquaredExponentialKernel(input_dims=1, active_dims=[0])
model = Model_mogptk(
    mogptk_dataset,
    inference=Exact(data_variance=[0.0]*n_tr), # noise-free data
    kernel=kernel,
    mean=mean
)

# train this very simple model
model.train(method='lbfgs', lr=1e-2, verbose=True)
model.print_parameters()

# prior samples
yy_prior = gen_1D_gp(model=model, x=tt, N=N_gp)

# posterior evaluated at a single value x_0
y0_pred_samples = model.gpr.sample_f(Z=x_0, n=10**2, prior=False)
print(f'y0 = {y0_pred_samples.median()} +/- {0.5 * y0_pred_samples.std()}')

# posterior (prediction) on the whole input space (x)
yy_pred = model.gpr.sample_f(Z=tt, n=N_gp, prior=False)

# posterior mean anv variance
post_mean, post_var = model.gpr.predict_f(tt, full=False)
post_mean = post_mean.ravel()
post_var = post_var.ravel()

#posterior forecast
post_mean_forecast, post_var_forecast = model.gpr.predict_f(tt_forecast, full=False)
post_mean_forecast = post_mean_forecast.ravel()
post_var_forecast = post_var_forecast.ravel()

#-------------------#
#       Plot
#-------------------#
fig, ax = plt.subplots(1, 1, figsize=get_figsize(columnwidth))

#--------#
# train and 
# test points
#--------#
ax.plot(
    x_tr, y_tr,
    linestyle='None',
    marker='x', markersize=2,
    color='red',
    label=r'$[y_i, x_i]$ (train, n={:2d})'.format(n_tr)
)
true_fun_plot = ax.plot(
    tt, g(tt),
    linestyle='dashed', linewidth=1.0,
    color='blue',
    label=r'$g$'
)
x0_vline = ax.axvline(
    x_0,
    linestyle='dotted',
    color='tab:cyan',
    label=r'$x_0$ (test)'
)

# decoration
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_xlim(a - (b-a)*0.1, b + (b-a)*0.1)
ax.set_ylim(-2, 2)
legend = ax.legend(
    loc='lower right', bbox_to_anchor=(0.99, 0.99),
    labelspacing=0.1, ncols=2,
    frameon=False, fontsize='xx-small'
)

fig.savefig(
    './images/gp_1D_example_noisefree_data.pdf',
    bbox_inches='tight'
)

#--------#
# add prior samples
# f(x_0)|train
#--------#
# for k in range(N_gp):
fig_prior, ax_prior = plt.subplots(1, 1, figsize=get_figsize(columnwidth))

for k in range(N_gp):
    ax_prior.plot(
        tt, yy_prior.T[k],
        linestyle='solid', linewidth=0.5,
        color='purple', alpha=0.2,
        marker=None,
        label=r'$f_{\theta} \sim GP(0, k_\theta)$' if k ==1 else None
    )

ax_prior.set_xlabel(r'$x$')
ax_prior.set_ylabel(r'$y$')
ax_prior.set_xlim(a - (b-a)*0.1, b + (b-a)*0.1)
ax_prior.set_ylim(-2, 2)
ax_prior.legend(
    loc='center', bbox_to_anchor=(0.90, 1.05),
    labelspacing=0.1, ncols=2,
    frameon=False, fontsize='xx-small'
)

fig_prior.savefig(
    './images/gp_1D_example_noisefree_data_prior.pdf',
    bbox_inches='tight'
)

#--------#
# add posterior samples from
# f(x_0)|train
#--------#
[l.remove() for l in true_fun_plot]
x0_vline.remove()
legend.remove()

# point to predict
ax.plot(
    x_0, y_0,
    linestyle='None',
    marker='x', markersize=5,
    color='tab:cyan',
    label=r'$[y_0, x_0]$'
)

# posterior samples on the whole input space
for k in range(N_gp):
    ax.plot(
        tt, yy_pred[k],
        linestyle='solid', linewidth=0.5,
        color='orange', alpha=0.2,
        marker=None,
        label=r'$f_{\theta} | y, x  \sim GP(m_{*}, k_{*,\theta}) $' if k ==1 else None
    )

# decoration
legend = ax.legend(
    loc='lower right', bbox_to_anchor=(0.99, 0.99),
    labelspacing=0.1, ncols=2,
    frameon=False, fontsize='xx-small'
)

fig.savefig(
    './images/gp_1D_example_noisefree_pred.pdf',
    bbox_inches='tight'
)

#--------#
# posterior
# mean and variance
# on the whole input space
#--------#
post_mean_plot = ax.plot(
    tt, post_mean,
    linestyle='solid', linewidth=1.0,
    color='tab:cyan', alpha=0.75,
    label=r'$m_{*}(x)$'
)

post_var_plot = ax.fill_between(
    tt,
    post_mean - np.sqrt(post_var), post_mean + np.sqrt(post_var),
    color='tab:cyan', alpha=0.150,
    label=r'$k_{*,\theta}(x, x)$'
)

legend.remove()
legend = ax.legend(
    loc='lower right', bbox_to_anchor=(0.99, 0.99),
    labelspacing=0.1, ncols=2,
    frameon=False, fontsize='xx-small'
)

fig.savefig(
    './images/gp_1D_example_noisefree_pred_meanvar.pdf',
    bbox_inches='tight'
)
#--------#
# posterior forecast x>1.5
#--------#
[l.remove() for l in post_mean_plot]
[post_var_plot.remove()]

ax.set_xlim(a - 0.1, b_forecast + 0.1)

ax.plot(
    tt_forecast, post_mean_forecast,
    linestyle='solid', linewidth=1.0,
    color='tab:cyan', alpha=0.75,
    label=r'$m_{*}(x)$'
)

ax.fill_between(
    tt_forecast,
    post_mean_forecast - np.sqrt(post_var_forecast),
    post_mean_forecast + np.sqrt(post_var_forecast),
    color='tab:cyan', alpha=0.150,
    label=r'$k_{*,\theta}(x, x)$'
)

true_fun_plot = ax.plot(
    tt_forecast, g(tt_forecast),
    linestyle='dashed', linewidth=0.75,
    color='blue',
    label=r'$g$'
)

legend.remove()
legend = ax.legend(
    loc='lower right', bbox_to_anchor=(0.99, 0.99),
    labelspacing=0.1, ncols=2,
    frameon=False, fontsize='xx-small'
)

fig.savefig(
    './images/gp_1D_example_noisefree_pred_meanvar_forecast.pdf',
    bbox_inches='tight'
)