#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# local imports
import turbo as tb
import turbo.modules as tm
from .config import Config

def plot_surrogate_likelihood(rec, fig_ax=None):
    assert not rec.has_unfinished_trials()
    fig, ax = fig_ax if fig_ax is not None else plt.subplots(figsize=Config.fig_sizes['overview'])
    trial_nums = [t[0] for t in sorted(rec.trials.items())]
    # only interested in the Bayesian optimisation steps
    trial_nums = [t for t in trial_nums if 'model' in rec.trials[t].selection_info]
    models = [rec.trials[t].selection_info['model'] for t in trial_nums]
    assert models, 'no Bayesian optimisation iterations recorded'

    # assume the surrogate models all have the same type
    m0 = models[0]
    assert all(type(m) == type(m0) for m in models)
    hyper_params = np.array([m.get_hyper_params() for m in models])
    likelihoods = np.array([m.get_log_likelihood() for m in models])

    ax.plot(trial_nums, likelihoods, linestyle='-', marker='o', markersize=4, label='log likelihood')
    ax.set_xlabel('trial number')
    ax.set_ylabel('surrogate model log likelihood')
    ax.set_title('Surrogate Likelihood Over Time', fontsize=14)
    ax.legend()


def plot_surrogate_hyper_params_1D(rec, param_index, trial_nums=None,
                                   use_param_bounds=False, size=25,
                                   axes=('trial_num', 'param', 'likelihood'),
                                   fig_ax=None, log_scale=False):
    '''plot the surrogate hyperparameters for each of the given trials

    the scatter plot is colored based on the to the data log-likelihood of the model.

    Args:
        rec (PlottingRecorder): the recorder which observed the run of an optimiser
        param_index: index into `Surrogate.get_hyper_params()` for the hyperparameter to use
        trial_nums (list): either a list of trial numbers to use for the plot, or None to use all of them
        use_param_bounds: whether to set the boundaries of the plot to match the boundaries of the hyperparameter space
        size (float): the size of the points of the scatter plot
        axes: the values to use for the (x, y, color) axes. each axis can be one of 'trial_num', 'param', 'likelihood'
        fig_ax: the matplotlib figure and axes to plot onto
    '''
    assert not rec.has_unfinished_trials()
    fig, ax = fig_ax if fig_ax is not None else plt.subplots(figsize=Config.fig_sizes['overview'])
    trial_nums = trial_nums or [t[0] for t in sorted(rec.trials.items())]
    # only interested in the Bayesian optimisation steps
    trial_nums = [t for t in trial_nums if 'model' in rec.trials[t].selection_info]
    models = [rec.trials[t].selection_info['model'] for t in trial_nums]
    assert models, 'no Bayesian optimisation iterations recorded'

    # assume the surrogate models all have the same type
    m0 = models[0]
    assert all(type(m) == type(m0) for m in models)
    hyper_params = np.array([m.get_hyper_params() for m in models])
    # assume the kernel (and so the bounds and names) don't change
    names = m0.get_hyper_param_names()
    assert all(len(names) == len(ps) for ps in hyper_params)
    likelihoods = np.array([m.get_log_likelihood() for m in models])

    if isinstance(m0, tm.SciKitGPSurrogate.ModelInstance):
        if use_param_bounds:
            # k.bounds is log transformed
            ax.set_ylim(np.exp(k.bounds[param_index, 0]), np.exp(k.bounds[param_index, 1]))
    elif isinstance(m0, tm.GPySurrogate.ModelInstance):
        if use_param_bounds:
            pass # don't know the parameter bounds
    else:
        raise NotImplementedError()

    values = {
        'trial_num': (trial_nums, 'trial number'),
        'likelihood': (likelihoods, 'log likelihood'),
        'param': (hyper_params[:, param_index], names[param_index])
    }
    x, x_label = values[axes[0]]
    y, y_label = values[axes[1]]
    c, c_label = values[axes[2]]

    s = ax.scatter(x, y, c=c, cmap='viridis', s=size, alpha=0.8)
    cbar = fig.colorbar(s, ticks=trial_nums if axes[2] == 'trial_num' else None, ax=ax)
    cbar.set_label(c_label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title('Surrogate Hyperparameters', fontsize=14)
    if log_scale:
        ax.set_yscale('log')

    return fig


def plot_surrogate_hyper_params_2D(rec, param_indices=(0, 1), trial_nums=None,
                                   size_limits=(5, 50), use_param_bounds=False,
                                   fig_ax=None, log_scale=False):
    '''plot the surrogate hyperparameters for each of the given trials

    the scatter plot is colored based on the trial number and the sizes
    correspond to the data log-likelihood of the model.

    Args:
        rec (PlottingRecorder): the recorder which observed the run of an optimiser
        param_indices: indices into `Surrogate.get_hyper_params()` for the hyperparameter to use for the x and y axes
        trial_nums (list): either a list of trial numbers to use for the plot, or None to use all of them
        size_limits: the (min, max) size to use for one of the points. set
            min=max to have every point the same size (don't size by likelihood)
        use_param_bounds: whether to set the boundaries of the plot to match the boundaries of the hyperparameter space
        fig_ax: the matplotlib figure and axes to plot onto
    '''
    assert not rec.has_unfinished_trials()
    fig, ax = fig_ax if fig_ax is not None else plt.subplots(figsize=Config.fig_sizes['2D_surrogate'])
    x_param, y_param = param_indices
    trial_nums = trial_nums or [t[0] for t in sorted(rec.trials.items())]
    # only interested in the Bayesian optimisation steps
    trial_nums = [t for t in trial_nums if 'model' in rec.trials[t].selection_info]
    models = [rec.trials[t].selection_info['model'] for t in trial_nums]
    assert models, 'no Bayesian optimisation iterations recorded'

    # assume the surrogate models all have the same type
    m0 = models[0]
    assert all(type(m) == type(m0) for m in models)
    hyper_params = np.array([m.get_hyper_params() for m in models])
    # assume the kernel (and so the bounds and names) don't change
    names = m0.get_hyper_param_names()
    likelihoods = np.array([m.get_log_likelihood() for m in models])

    if isinstance(m0, tm.SciKitGPSurrogate.ModelInstance):
        if use_param_bounds:
            # k.bounds is log transformed
            ax.set_xlim(np.exp(k.bounds[x_param, 0]), np.exp(k.bounds[x_param, 1]))
            ax.set_ylim(np.exp(k.bounds[y_param, 0]), np.exp(k.bounds[y_param, 1]))
    elif isinstance(m0, tm.GPySurrogate.ModelInstance):
        if use_param_bounds:
            pass # don't know param bounds
    else:
        raise NotImplementedError()

    sizes = tb.utils.remap(likelihoods, (min(likelihoods), max(likelihoods)), size_limits)
    s = ax.scatter(hyper_params[:,x_param], hyper_params[:,y_param], c=trial_nums, cmap='viridis', s=sizes, alpha=0.8, linewidth=0.5, edgecolors='k')
    ticks = None if len(trial_nums) > 50 else trial_nums
    c = fig.colorbar(s, ticks=ticks, ax=ax)
    c.set_label('trial number')

    ax.set_title('Surrogate Hyperparameters', fontsize=14)
    ax.set_xlabel(names[x_param])
    ax.set_ylabel(names[y_param])
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    return fig

