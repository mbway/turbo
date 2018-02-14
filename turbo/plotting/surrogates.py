#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# local imports
import turbo as tb
import turbo.modules as tm

def plot_surrogate_likelihood_over_time(rec, fig_ax=None):
    fig, ax = fig_ax if fig_ax is not None else plt.subplots(figsize=(10, 4)) # inches
    trial_nums = [t[0] for t in sorted(rec.trials.items())]
    # only interested in the Bayesian optimisation steps
    trial_nums = [t for t in trial_nums if 'model' in rec.trials[t].selection_info]
    models = [rec.trials[t].selection_info['model'] for t in trial_nums]
    assert models, 'no Bayesian optimisation iterations recorded'

    # assume the surrogate models all have the same type
    if isinstance(models[0], tm.SciKitGPSurrogate):
        likelihoods = np.array([m.model.log_marginal_likelihood() for m in models]) # log likelihood

        ax.plot(trial_nums, likelihoods, linestyle='-', marker='o', markersize=4, label='log likelihood')
        ax.set_xlabel('trial number')
        ax.set_ylabel('surrogate model log likelihood')
        ax.set_title('Surrogate Likelihood Over Time', fontsize=14)
        ax.legend()
    else:
        raise NotImplementedError()


def plot_surrogate_hyper_params_1D(rec, param_index, trial_nums=None,
                                   use_param_bounds=False, size=25,
                                   axes=('trial_num', 'param', 'likelihood'),
                                   fig_ax=None):
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
    fig, ax = fig_ax if fig_ax is not None else plt.subplots(figsize=(8, 6)) # inches
    trial_nums = trial_nums or [t[0] for t in sorted(rec.trials.items())]
    # only interested in the Bayesian optimisation steps
    trial_nums = [t for t in trial_nums if 'model' in rec.trials[t].selection_info]
    models = [rec.trials[t].selection_info['model'] for t in trial_nums]
    assert models, 'no Bayesian optimisation iterations recorded'

    # assume the surrogate models all have the same type
    if isinstance(models[0], tm.SciKitGPSurrogate):
        # assume the kernel (and so the bounds and names) don't change
        k = models[0].model.kernel
        names = [h.name for h in k.hyperparameters]
        if use_param_bounds:
            # k.bounds is log transformed
            ax.set_ylim(k.bounds[param_index, 0], k.bounds[param_index, 1])

        likelihoods = np.array([m.model.log_marginal_likelihood() for m in models]) # log likelihood
        hyper_params = np.array([m.get_hyper_params() for m in models]) # log transformed

        values = {
            'trial_num': (trial_nums, 'trial number'),
            'likelihood': (likelihoods, 'log likelihood'),
            'param': (hyper_params[:, param_index], 'log({})'.format(names[param_index]))
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
    else:
        raise NotImplementedError()


def plot_surrogate_hyper_params_2D(rec, param_indices=(0, 1), trial_nums=None,
                                   size_limits=(5, 50), use_param_bounds=False,
                                   fig_ax=None):
    '''plot the surrogate hyperparameters for each of the given trials

    the scatter plot is colored based on the trial number and the sizes
    correspond to the data log-likelihood of the model.

    Args:
        rec (PlottingRecorder): the recorder which observed the run of an optimiser
        param_indices: indices into `Surrogate.get_hyper_params()` for the hyperparameter to use for the x and y axes
        trial_nums (list): either a list of trial numbers to use for the plot, or None to use all of them
        size_limits: the (min, max) size to use for one of the points
        use_param_bounds: whether to set the boundaries of the plot to match the boundaries of the hyperparameter space
        fig_ax: the matplotlib figure and axes to plot onto
    '''
    fig, ax = fig_ax if fig_ax is not None else plt.subplots(figsize=(8, 6)) # inches
    x_param, y_param = param_indices
    trial_nums = trial_nums or [t[0] for t in sorted(rec.trials.items())]
    # only interested in the Bayesian optimisation steps
    trial_nums = [t for t in trial_nums if 'model' in rec.trials[t].selection_info]
    models = [rec.trials[t].selection_info['model'] for t in trial_nums]
    assert models, 'no Bayesian optimisation iterations recorded'

    # assume the surrogate models all have the same type
    if isinstance(models[0], tm.SciKitGPSurrogate):
        # assume the kernel (and so the bounds and names) don't change
        k = models[0].model.kernel
        names = [h.name for h in k.hyperparameters]
        if use_param_bounds:
            # k.bounds is log transformed
            ax.set_xlim(k.bounds[x_param, 0], k.bounds[x_param, 1])
            ax.set_ylim(k.bounds[y_param, 0], k.bounds[y_param, 1])

        likelihoods = np.array([m.model.log_marginal_likelihood() for m in models]) # log likelihood
        hyper_params = np.array([m.get_hyper_params() for m in models]) # log transformed

        sizes = tb.utils.remap(likelihoods, (min(likelihoods), max(likelihoods)), size_limits)
        s = ax.scatter(hyper_params[:,x_param], hyper_params[:,y_param], c=trial_nums, cmap='viridis', s=sizes, alpha=0.8)
        c = fig.colorbar(s, ticks=trial_nums, ax=ax)
        c.set_label('trial number')

        ax.set_title('Surrogate Hyperparameters', fontsize=14)
        ax.set_xlabel('log({})'.format(names[x_param]))
        ax.set_ylabel('log({})'.format(names[y_param]))
    else:
        raise NotImplementedError()
