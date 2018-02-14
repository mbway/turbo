#!/usr/bin/env python3
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#TODO: plot objective function value over time. Should this file be named something different?

def plot_error_over_time(rec, true_best, log_scale=False, plot_best=True, fig_ax=None):
    '''
    plot a line graph showing the difference between the known optimal value
    and the optimiser's best guess at each step.

    Args:
        rec (PlottingRecorder): the recorder which observed the run of an optimiser
        true_best (float): the globally optimal value to compare to
        log_scale: whether to plot on a logarithmic or a linear scale
        plot_best: whether to plot a marker showing the trial with the overall best cost
        fig_ax: the figure and axes to plot to in a tuple (optional)
    '''
    fig, ax = fig_ax if fig_ax is not None else plt.subplots(figsize=(10, 4)) # inches

    trials = rec.get_sorted_trials()
    xs = [n for n, t in trials]
    errors = [(true_best - t.y if rec.optimiser.is_maximising() else t.y - true_best)
              for n, t in trials]
    color_dict = {
        'pre_phase': 'violet',
        'bayes': '#4c72b0',
        'fallback': 'red'
    }
    colors = [color_dict[t.selection_info['type']] for n, t in trials]

    assert errors, 'no trials'
    if any(e < 0 for e in errors):
        print('warning: some of the trials are better than true_best!')

    ax.margins(0.01, 0.05)
    ax.axhline(y=0, linestyle='--', color='grey', linewidth=0.8)
    ax.plot(xs, errors, color='#4c72b0', label='error')
    ax.scatter(xs, errors, s=16, color=colors, zorder=10)

    if plot_best:
        best_i = np.argmin(errors)

        # move the marker out of the way.
        offset = -4.5 # pt
        offset = mpl.transforms.ScaledTranslation(0, offset/fig.dpi, fig.dpi_scale_trans)
        trans = ax.transData + offset

        ax.plot(xs[best_i], errors[best_i], marker='^', color='#55a868',
                linestyle='', markersize=10, zorder=10, markeredgecolor='black',
                markeredgewidth=1, label='best cost', transform=trans)
        ax.margins(0.01, 0.1) # need more margins to fit the marker in

    ax.set_title('Error Over Time', fontsize=14)
    ax.set_xlabel('trial num')
    ax.set_ylabel('error')

    if log_scale:
        ax.set_yscale('log')

    if len(xs) < 50:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2.0))
    elif len(xs) < 100:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5.0))
    ax.legend()


