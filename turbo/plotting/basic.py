#!/usr/bin/env python3
'''
Basic plotting functionality for `Optimiser` objects.
'''
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_error_over_time(opt, true_best, log_scale=False, ax=None):
    '''
    plot a line graph showing the difference between the known optimal value
    and the optimiser's best guess at each step.

    Args:
        opt (Optimiser): the optimiser with the data to plot
        true_best (float): the globally optimal value to compare to
        log_scale: whether to plot on a logarithmic or a linear scale
        ax: the axes to plot to (optional)
    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 10)) # inches

    rt = opt.rt
    n = rt.num_trials()
    xs = range(1, n+1)
    errors = [(true_best - s if opt.is_maximising() else s - true_best)
              for s in rt.trial_ys]

    if any(e < 0 for e in errors):
        print('warning: some of the trials are better than true_best!')

    ax.axhline(y=0, linestyle='--', color='grey', linewidth=0.8)
    ax.plot(xs, errors, marker='o', markersize=4, color='#4c72b0', label='error')

    ax.set_title('Error Over Time', fontsize=14)
    ax.set_xlabel('trials')
    ax.set_ylabel('error')

    if log_scale:
        ax.set_yscale('log')

    ax.margins(0.01, 0.05)
    if n < 50:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2.0))
    elif n < 100:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5.0))
    ax.legend()

