import matplotlib.pyplot as plt
import numpy as np
from constants import columnwidth

#----------------------#
#------ Plot ----------#
#----------------------#
def scatter_hist(x, y, margin_hist=True, fig=None):
    """Scatter plot with histogram marginals."""

    if fig is None:
        fig = plt.figure(
            # want a square
            figsize=(4, 4)
            # get_figsize(columnwidth, 0.5)
        )

    color = 'purple'
    # marginal distributions
    if margin_hist:

        # Add a gridspec with two rows and two columns and 
        # a ratio of 1 to 4 between the size of the marginal Axes
        gs = fig.add_gridspec(
            2, 2,
            width_ratios=(4, 1), height_ratios=(1, 4),
            left=0.1, right=0.99, bottom=0.1, top=0.99,
            wspace=0.0, hspace=0.0
        )

        # Create the Axes.
        ax = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

        # first marginal
        ax_histx.axis('off')
        # second marginal
        ax_histy.axis('off')

        # imits by hand
        binwidth = 0.15
        xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
        lim = (int(xymax/binwidth) + 1) * binwidth
        bins = np.arange(-lim, lim + binwidth, binwidth)

        ax_histx.hist(
            x,
            bins='fd',
            density=True,
            color=color
        )
        ax_histy.hist(
            y,
            bins='fd', density=True,
            color=color, orientation='horizontal'
        )

        # does not work
        # ax_histx.set_box_aspect(1/4)
        # ax_histy.set_box_aspect(4/1)
        # ax.set_box_aspect(1)
    else:
        gs = fig.add_gridspec(
            1, 1,
            left=0.1, right=0.99, bottom=0.1, top=0.99,
            wspace=0.0, hspace=0.0
        )
        ax = fig.add_subplot(gs[0, 0])

    # 2d data
    ax.scatter(x, y, c=color, marker='.', s=1.0, alpha=0.6)
    ax.set_xlabel(r'$y_1$')
    ax.set_ylabel(r'$y_2$')

    if margin_hist:
        return fig, ax, ax_histx, ax_histy
    
    return fig, ax

#----------------------#
#--------- GP ---------#
#----------------------#
def gen_1D_gp(model, x, N):
    """Draw N GPs with given kernel and mean."""

    # model.gpr.likelihood.scale.assign(value=1.0, train=False)
    d = 1
    # convert to 2D array of (1D) grid values
    return model.gpr.sample_f(
        Z=np.copy(x).reshape((-1, d)),
        n=N,
        prior=True
    ).numpy().T

def kernel_autocov_1D(kernel, x):
    """Return 1D autocovariance values from kernel."""
    
    from torch import from_numpy, detach

    Z = np.copy(x).reshape(-1, 1)
    K = kernel.K(X1=from_numpy(Z), X2=from_numpy(np.zeros((1, 1))))
    K = detach(K).numpy().reshape(-1, 1)

    return K / kernel.K_diag(
        X1=from_numpy(np.zeros((1, 1)))
    ).detach().numpy()

#----------------------#
#--- Latex useful -----#
#----------------------#
def numpy_to_latex(matrix, n_decimals=2):
    """Convert a numpy array into a LaTeX matrix."""

    if not isinstance(matrix, np.ndarray):
        raise ValueError("Must be a numpy array.")

    factor = 10**n_decimals
    matrix = np.trunc(matrix * factor) / factor
    
    lines = str(matrix).replace('[', '').replace(']', '').splitlines()
    latex_matrix = "\\begin{bmatrix}\n"
    latex_matrix += " \\\\\n".join([" & ".join(line.split()) for line in lines])
    latex_matrix += "\n\\end{bmatrix}"

    return latex_matrix

def get_figsize(columnwidth, wf=0.5, hf=((5.0**0.5)-1.0)/2.0):
    """
    Return figure size in inches given the column with of a LaTeX template.
        columnwidth [float]: width of the column in latex. Get this from LaTeX 
            using "\\showthe\\columnwidth".
        wf [float]:  width fraction in columnwidth units
        hf [float]:  height fraction in columnwidth units.
            Set by default to golden ratio.

    Returns:  [fig_width,fig_height]: that should be given to matplotlib
    """
    fig_width_pt = columnwidth*wf
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*hf      # height in inches
    return [fig_width, fig_height]