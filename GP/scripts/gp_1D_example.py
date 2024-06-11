import numpy as np
from matplotlib import pyplot as plt
from mogptk import Model as Model_mogptk
from mogptk import Exact
from mogptk import DataSet
from mogptk.gpr.mean import LinearMean, ConstantMean
from mogptk.gpr.singleoutput import MaternKernel
from torch import manual_seed
from utils import get_figsize, gen_1D_gp
from constants import columnwidth

manual_seed(3004666)

#-------------------#
# Data generation
#-------------------#
# input space
a, b = 0.0, 2.0
nt = 10**3
tt = np.linspace(a, b, num=nt)
N_gp = 10 # number of GP draws

# output space
d = 1
f = lambda ti: -1 * np.cos(2*np.pi*ti) + 0.5 * np.sin(6*np.pi*ti)
n_tr = 50 # number of samples of a single signal
n_tst = 10

# training samples
t_tr = np.random.uniform(a, b, (n_tr, 1))
y_tr = np.array(
    [f(t) for t in t_tr[:,0]]
)[:,None]

# additive noise
y_tr = y_tr[:, 0]
# z_samples +=  np.random.normal(loc=0.0, scale=0.01, size=n)
mogptk_dataset = DataSet([t_tr], [y_tr])

# # here we cheat, we sample from a known GP
# kernel = MaternKernel(nu=1.5, input_dims=1, active_dims=[0])
# model = Model_mogptk(
#     mogptk_dataset,
#     inference=Exact(),
#     kernel=kernel,
#     mean=ConstantMean()
# )

fig, ax = plt.subplots(1, 1)
ax.set_xlim(a, b)
ax.set_ylim(-2, 2)

ax.plot(

)