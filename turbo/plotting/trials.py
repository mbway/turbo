#!/usr/bin/env python3
'''
Plotting individual iterations of Bayesian optimisation.
'''
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.transforms

# local imports
import turbo.modules as tm
from turbo.utils import row2D, unique_rows_close


#TODO: could use k-means to choose N locations to plot the surrogate through to get the best coverage of interesting regions while using as few plots as possible

class PlottingRecorder(tm.Listener):
    class Trial:
        __slots__ = ('trial_num', 'x', 'y', 'extra_data')

    def __init__(self):
        self.trials = {}

    def selection_started(self, trial_num):
        assert trial_num not in self.trials.keys()
        t = PlottingRecorder.Trial()
        t.trial_num = trial_num
        self.trials[trial_num] = t

    def eval_started(self, trial_num, x, extra_data):
        t = self.trials[trial_num]
        t.x = x
        t.extra_data = extra_data

    def eval_finished(self, trial_num, y):
        t = self.trials[trial_num]
        t.y = y

    def get_data_for_trial(self, trial_num):
        #TODO: for async cannot assume that finished == all trials before trial_num
        finished = [self.trials[n] for n in range(trial_num)]
        trial = self.trials[trial_num]
        return finished, trial


def vary_param(point, param_index, param_range):
    '''generate a matrix of points (as rows) which are based on the given point,
    but varies along the given range for a single given parameter.
    '''
    param_range = param_range.flatten()
    # create many duplicates of the given point
    points = np.repeat(row2D(point), len(param_range), axis=0)
    points[:,param_index] = param_range
    return points

def choose_predict_locations(trial_x, f_xs, param_index):
    '''return a matrix with the points to predict through as rows

    avoid drawing predictions of the same place more than once, so
    avoid duplicate configurations which are identical to another
    except for the value of 'param', since the plot varies this
    parameter: the resulting plot will be the same in both cases.
    '''
    # a copy of the current samples with the focused parameter zeroed
    # start with s.next_x since that is a point which is guaranteed to
    # have a prediction plotted through it
    param_zeroed = np.vstack([row2D(trial_x)] + f_xs)
    param_zeroed[:, param_index] = 0
    param_zeroed = unique_rows_close(param_zeroed, close_tolerance=1e-3)
    param_zeroed = param_zeroed[1:, :] # exclude the trial point (first row)
    return param_zeroed


def plot_trial_1D(opt, rec, param, trial_num, true_objective=None,
                  divisions=200, n_sigma=2, predict_through_all=True,
                  log_scale=False):
    r'''Plot the state of Bayesian optimisation (perturbed along a single
    parameter) at the time that the given trial was starting its evaluation.

    The intuition for the case of a 1D space is trivial: the plot is simply the
    parameter value and the corresponding objective value and acquisition
    values. In 2D, visualise the surface plot of the two parameters against the
    objective value (as the height). This plot takes a 1D cross section of that
    surface along the specified axis and passing through the point of the next
    point to test to show how the acquisition function varies along that
    dimension. The same holds for higher dimensions but is harder to visualise.

    Args:
        opt (turbo.Optimiser): the `Optimiser` who's data to plot
        rec (PlottingRecorder): the recorder which observed the run of `opt`
        param (str): the name of the parameter to perturb to obtain the graph
        trial_num (int): the number of the trial to plot
        true_objective: true objective function (or array of pre-computed cost
            values corresponding to the number of divisions) (None to omit)
        n_sigma (float): the number of standard deviations from the mean to plot
            the uncertainty confidence interval.

            .. note::
                for a normal distribution (i.e. GP surrogate):
                :math:`1\implies 68\%,\;2\implies 95\%,\;3\implies 99\%`
        predict_through_all: whether to plot a surrogate prediction through
            every sample or just through the location of the point chosen this
            iteration.
        log_scale: whether to use a log scale for the `x` axis
    '''
    ################
    # Extract Data
    ################
    rt = opt.rt
    assert trial_num >= 0 and trial_num < rt.max_trials, 'invalid iteration number'
    assert opt.async_eval is None, 'this function does not support asynchronous optimisation runs'

    param_range = opt.latent_space.evenly_spaced_range(param, divisions)
    latent_range = opt.latent_space.evenly_spaced_range(param, divisions)
    param_index = opt.bounds.get_param_index(param)

    # f_ => finished trial data
    # _xs => latent space point containing all parameter values
    # _ps => value for the chosen parameter _only_ (in the input (unwarped) space)
    finished, trial = rec.get_data_for_trial(trial_num)

    if trial.extra_data['type'] == 'pre_phase':
        print('pre-phase trial {}'.format(trial_num))
        return # TODO

    f_xs = [f.x for f in finished]
    f_ps = [opt.latent_space.from_latent(p)[param_index] for p in f_xs]
    f_ys = [f.y for f in finished]
    trial_p = opt.latent_space.from_latent(trial.x)[param_index]
    # _i => index
    best_i = np.argmax(f_ys) if opt.is_maximising() else np.argmin(f_ys)

    model = trial.extra_data['model']
    acq_fun = opt.acq_func_factory(trial_num, model, opt.desired_extremum)
    # it makes sense to plot the acquisition function through the slice
    # corresponding to the current trial.
    trial_perturbed = vary_param(trial.x, param_index, latent_range)
    trial_ac = acq_fun(trial_perturbed)


    ##################
    # Plotting Setup
    ##################
    fig = plt.figure(figsize=(16, 10)) # inches
    grid = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2, 1])
    ax1, ax2 = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])

    title = 'Bayesian Optimisation Trial {}'.format(trial_num)

    fig.suptitle(title, fontsize=14)
    ax1.margins(0.005, 0.05)
    ax2.margins(0.005, 0.05)
    if log_scale:
        ax1.set_xscale('log')
        ax2.set_xscale('log')
    fig.subplots_adjust(hspace=0.3)


    ######################
    # Plotting Surrogate
    ######################
    ax1.set_ylabel('objective')
    ax1.set_title('Surrogate Objective Function')

    ### Plot True Objective
    if true_objective is not None:
        # true cost is either the cost function, or pre-computed costs as an array
        if callable(true_objective):
            true_ys = [true_objective(v) for v in param_range]
        else:
            assert len(true_objective) == len(param_range)
            true_ys = true_objective
        ax1.plot(param_range, true_ys, '--', color='#2f2f2f',
                 label='true objective', linewidth=1.0, alpha=0.6)

    ### Plot Samples
    # plot samples projected onto the `param` axis
    # exclude the best sample
    mask = np.ones((len(f_ys),), bool)
    mask[best_i] = False
    ax1.plot(np.vstack(f_ps)[mask,:], np.array(f_ys)[mask], 'o', markersize=6,
             label='finished trials', color='#213bcb', zorder=5)

    ax1.plot(f_ps[best_i], f_ys[best_i], '*', markersize=10,
                color='deepskyblue', zorder=10, label='best trial')

    ### Plot Surrogate Function
    def plot_prediction_through(point, label, alphas):
        perturbed = vary_param(point, param_index, latent_range)
        mus, sigmas = model.predict(perturbed, return_std_dev=True)
        if label:
            mu_label = r'surrogate $\mu$'
            sigma_label = r'uncertainty ${}\sigma$'.format(n_sigma)
        else:
            mu_label, sigma_label = (None, None)
        mu_alpha, sigma_alpha = alphas
        l = ax1.plot(param_range, mus, '-', color='#b544ee', label=mu_label, alpha=mu_alpha, linewidth=1.0)
        ax1.fill_between(param_range, mus - n_sigma*sigmas, mus + n_sigma*sigmas,
                         alpha=sigma_alpha, color='#c465f3',
                         label=sigma_label)

    #TODO: fit the view to the cost function, don't expand to fit in the uncertainty

    plot_prediction_through(trial.x, label=True, alphas=(1, 0.25))

    # plot the predictions through each sample
    if predict_through_all:
        locs = choose_predict_locations(trial.x, f_xs, param_index)
        if len(locs) > 0:
            # cap to make sure they don't become invisible
            alpha = max(0.4/locs.shape[0], 0.015)
            for p in locs:
                plot_prediction_through(row2D(row), label=False, alphas=(alpha, alpha))

    ### Plot Vertical Bars
    bar_width, bar_color = 1, '#3590ce'
    ax1.axvline(x=trial_p, linewidth=bar_width, color=bar_color)
    ax1.plot(trial_p, trial.y, 'bo', markersize=6, alpha=0.4, label='this trial')

    ax1.legend()

    #################################
    # Plotting Acquisition Function
    #################################
    ax2.set_xlabel('parameter {}'.format(param))
    ax2.set_ylabel(acq_fun.get_name())
    ax2.set_title('Acquisition Function')

    ax2.plot(param_range, trial_ac, '-', color='g', linewidth=1.0,
             label='acquisition function')
    ax2.fill_between(param_range, np.zeros_like(param_range), trial_ac,
                     alpha=0.3, color='palegreen')

    ax2.axvline(x=trial_p, linewidth=bar_width, color=bar_color)
    ax2.plot(trial_p, trial.extra_data['ac_x'], '^', color='black',
             markersize=7, zorder=10, label='this trial')

    ax2.legend()
    return fig
