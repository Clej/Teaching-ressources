import matplotlib.pyplot as plt
import numpy as np

#----------------------#
#------ Plot ----------#
#----------------------#
def scatter_hist(x, y, margin_hist=True, fig=None):
    """Scatter plot with histogram marginals."""

    if fig is None:
        fig = plt.figure(figsize=(6, 6))

    color = 'purple'
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
    else:
        gs = fig.add_gridspec(
            1, 1,
            left=0.1, right=0.99, bottom=0.1, top=0.99,
            wspace=0.0, hspace=0.0
        )
        ax = fig.add_subplot(gs[0, 0])

    # 2d data
    ax.scatter(x, y, c=color, marker='.', s=1.0)
    ax.set_xlabel(r'$y_1$', fontsize='x-large')
    ax.set_ylabel(r'$y_2$', fontsize='x-large')

    if margin_hist:
        return fig, ax, ax_histx, ax_histy
    
    return fig, ax

#----------------------#
#------ Misc ----------#
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
