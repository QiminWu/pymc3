from matplotlib import pyplot as plt
import pymc3 as pm
from pymc3 import plots as pmp
import numpy as np


def correlation_plot_hist(mtrace, varnames=None,
        transform=lambda x: x, figsize=None, hist_color='orange', cmap=None,
        grid=50, chains=None, ntickmarks=2, point=None,
        point_style='.', point_color='red', point_size='6', alpha=0.35):
    """
    Plot 2d marginals (with kernel density estimation) showing the correlations
    of the model parameters. In the main diagonal is shown the parameter
    histograms.

    Parameters
    ----------
    mtrace : :class:`pymc3.base.MutliTrace`
        Mutlitrace instance containing the sampling results
    varnames : list of variable names
        Variables to be plotted, if None all variable are plotted
    transform : callable
        Function to transform data (defaults to identity)
    figsize : figure size tuple
        If None, size is (12, num of variables * 2) inch
    cmap : matplotlib colormap
    hist_color : str or tuple of 3
        color according to matplotlib convention
    grid : resolution of kernel density estimation
    chains : int or list of ints
        chain indexes to select from the trace
    ntickmarks : int
        number of ticks at the axis labels
    point : dict
        Dictionary of variable name / value  to be overplotted as marker
        to the posteriors e.g. mean of posteriors, true values of a simulation
    point_style : str
        style of marker according to matplotlib conventions
    point_color : str or tuple of 3
        color according to matplotlib convention
    point_size : str
        marker size according to matplotlib conventions

    Returns
    -------
    fig : figure object
    axs : subplot axis handles
    """

    if varnames is None:
        varnames = mtrace.varnames

    nvar = len(varnames)

    if figsize is None:
        figsize = mpl_papersize('a4', 'landscape')

    fig, axs = plt.subplots(nrows=nvar, ncols=nvar, figsize=figsize,
            subplot_kw={'adjustable': 'box-forced'})

    d = dict()

    for var in varnames:
        d[var] = transform(mtrace.get_values(
                var, chains=chains, combine=True, squeeze=True))

    for k in range(nvar):
        v_namea = varnames[k]
        a = d[v_namea]

        for l in range(k, nvar):
            v_nameb = varnames[l]
            pm._log.debug('%s, %s' % (v_namea, v_nameb))
            if l == k:
                if point is not None:
                    if v_namea in point.keys():
                        reference = point[v_namea]
                        axs[l, k].axvline(
                            x=reference, color=point_color,
                            lw=int(point_size) / 4.)
                    else:
                        reference = None
                else:
                    reference = None

                histplot_op(
                    axs[l, k], pmp.utils.make_2d(a), alpha=alpha,
                    color='orange', tstd=0., reference=reference,
                    ntickmarks=ntickmarks)
                axs[l, k].get_yaxis().set_visible(False)

                xticks = axs[l, k].get_xticks()
            else:
                b = d[v_nameb]

                kde2plot(
                    a, b, grid=grid, ax=axs[l, k], cmap=cmap, aspect='auto')

                bmin = b.min()
                bmax = b.max()

                if point is not None:
                    if v_namea and v_nameb in point.keys():
                        axs[l, k].plot(point[v_namea], point[v_nameb],
                            color=point_color, marker=point_style,
                            markersize=point_size)

                        bmin = np.minimum(bmin, point[v_nameb])
                        bmax = np.maximum(bmax, point[v_nameb])

                ytickmarks = get_tickmarks(bmin, bmax, ntickmarks=ntickmarks)
                axs[l, k].set_xticks(xticks)
                axs[l, k].set_yticks(ytickmarks)

            if l != nvar - 1:
                axs[l, k].get_xaxis().set_ticklabels([])

            if k == 0:
                axs[l, k].set_ylabel(v_nameb)
            else:
                axs[l, k].get_yaxis().set_ticklabels([])

            axs[l, k].tick_params(direction='in')

        axs[l, k].set_xlabel(v_namea)

    for k in range(nvar):
        for l in range(k):
            fig.delaxes(axs[l, k])

    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    return fig, axs

