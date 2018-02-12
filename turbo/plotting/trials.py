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
import turbo.gui as tg
from turbo.utils import row2D, unique_rows_close


#TODO: could use k-means to choose N locations to plot the surrogate through to get the best coverage of interesting regions while using as few plots as possible

def _vary_param(point, param_index, range_):
    '''generate a matrix of points (as rows) which are based on the given point,
    but varies along the given range for a single given parameter.
    '''
    range_ = range_.flatten()
    # create many duplicates of the given point
    points = np.repeat(row2D(point), len(range_), axis=0)
    points[:,param_index] = range_
    return points

def _choose_predict_locations(trial_x, f_xs, param_index):
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

def interactive_plot_trial_1D(rec, param=None, trial_num=None, *args, **kwargs):
    '''choose the param and trial_num for the trial plot interactively

    param and trial_num can be left as None to be chosen interactively, or can be specified

    Args:
        rec (PlottingRecorder): the recorder which observed the run of an optimiser
        param: None to be chosen interactively, or can be specified
        trial_num: None to be chosen interactively, or can be specified
    '''
    # choose interactively
    if param is None or trial_num is None:
        opt = rec.optimiser
        # partially apply by specifying all the non-interactive parameters
        plot_params = (args, kwargs) # these names have special meanings, so store them under a different name
        plot = lambda param, trial_num, latent: \
            plot_trial_1D(rec, param, trial_num, *plot_params[0], plot_in_latent_space=latent, **plot_params[1])
        # store the plots so they only have to be rendered once
        memoised = tg.PlotMemoization(plot)

        # updated by the widgets, or left alone if not None
        def update_plot(param, trial_num):
            key = (param, trial_num)
            if None not in key: # all args set
                latent = param.startswith('latent_')
                param = param[len('latent_'):] if latent else param
                memoised.show(key, {'param': param, 'trial_num': trial_num, 'latent': latent})

        if param is None:
            params = [b[0] for b in opt.bounds.ordered]
            params = params + [b[0] for b in opt.latent_space.get_latent_bounds().ordered if b[0] not in params]
            param = tg.dropdown(params, description='Param:', initial=0)
        else:
            param = tg.widgets.fixed(param)

        if trial_num is None:
            trial_num = tg.slider(opt.rt.finished_trials, description='Trial:', initial=-1)
        else:
            trial_num = tg.widgets.fixed(trial_num)

        tg.widgets.interact(update_plot, param=param, trial_num=trial_num)
    else:
        plot_trial_1D(rec, param, trial_num, *args, **kwargs)



def plot_trial_1D(rec, param, trial_num, true_objective=None,
                  plot_in_latent_space=True, divisions=200, n_sigma=2,
                  predict_through_all=True, log_scale=False, fig=None):
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
        rec (PlottingRecorder): the recorder which observed the run of an optimiser
        param (str): the name of the parameter to perturb to obtain the graph.
        trial_num (int): the number of the trial to plot.
            <0 => index from the end/last trial
        true_objective: true objective function (or array of pre-computed cost
            values corresponding to the number of divisions) (None to omit)
        plot_in_latent_space: whether to plot in latent space or input space
        divisions (int): the resolution of the plot / number of points along the
            parameter domain to plot at. (higher => slightly better quality but
            takes longer to render)
        n_sigma (float): the number of standard deviations from the mean to plot
            the uncertainty confidence interval.

            .. note::
                for a normal distribution (i.e. GP surrogate):
                :math:`1\implies 68\%,\;2\implies 95\%,\;3\implies 99\%`
        predict_through_all: whether to plot a surrogate prediction through
            every sample or just through the location of the point chosen this
            iteration.
        log_scale: whether to use a log scale for the `x` axis
        fig: the matplotlib figure to plot onto
    '''
    # to catch errors where the user mistakes this for the interactive version
    assert param is not None and trial_num is not None, 'param and trial_num must be specified'

    ################
    # Extract Data
    ################
    opt = rec.optimiser
    rt = opt.rt
    rt.check_consistency()
    # allow negative trial numbers for referring to the end
    trial_num = rt.max_trials + trial_num if trial_num < 0 else trial_num
    assert trial_num >= 0 and trial_num < rt.max_trials, 'invalid iteration number'
    assert opt.async_eval is None, 'this function does not support asynchronous optimisation runs'

    if plot_in_latent_space:
        # linear in the latent space, transform this back to get the points in the input space
        latent_range = opt.latent_space.linear_latent_range(param, divisions) # the input space range
        input_range = np.array([opt.latent_space.param_from_latent(param, p) for p in latent_range])
        plot_range = latent_range # the x axis of the plot
    else:
        # linear in the input space, transform this to get the points in the latent space
        input_range = np.linspace(*opt.bounds.get(param), num=divisions) # the input space range
        latent_range = np.array([opt.latent_space.param_to_latent(param, p) for p in input_range])
        plot_range = input_range # the x axis of the plot

    # f_ => finished trial data
    # _xs => latent space point containing all parameter values
    # _ps => value for the chosen parameter _only_
    finished, trial = rec.get_data_for_trial(trial_num)

    if trial.extra_data['type'] == 'pre_phase':
        print('pre-phase trial {}'.format(trial_num))
        return # TODO

    param_index = opt.bounds.get_param_index(param)
    f_xs = [f.x for f in finished]
    f_ps = [x[param_index] for x in f_xs]
    f_ys = [f.y for f in finished]
    trial_p = trial.x[param_index]
    if not plot_in_latent_space:
        f_ps = [opt.latent_space.param_from_latent(param, p) for p in f_ps]
        trial_p = opt.latent_space.param_from_latent(param, trial_p)
    # _i => index
    best_i = np.argmax(f_ys) if opt.is_maximising() else np.argmin(f_ys)

    model = trial.extra_data['model']
    acq_fun = rec.get_acquisition_function(trial_num)
    # it makes sense to plot the acquisition function through the slice
    # corresponding to the current trial.
    trial_perturbed = _vary_param(trial.x, param_index, latent_range)
    trial_ac = acq_fun(trial_perturbed)


    ##################
    # Plotting Setup
    ##################
    fig = fig or plt.figure(figsize=(16, 10)) # inches
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
            true_ys = [true_objective(v) for v in input_range]
        else:
            assert len(true_objective) == divisions
            true_ys = true_objective
        ax1.plot(plot_range, true_ys, '--', color='#2f2f2f',
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
        perturbed = _vary_param(point, param_index, latent_range)
        mus, sigmas = model.predict(perturbed, return_std_dev=True)
        if label:
            mu_label = r'surrogate $\mu$'
            sigma_label = r'uncertainty ${}\sigma$'.format(n_sigma)
        else:
            mu_label, sigma_label = (None, None)
        mu_alpha, sigma_alpha = alphas
        l = ax1.plot(plot_range, mus, '-', color='#b544ee', label=mu_label, alpha=mu_alpha, linewidth=1.0)

        ax1.fill_between(plot_range, mus - n_sigma*sigmas, mus + n_sigma*sigmas,
                         alpha=sigma_alpha, color='#c465f3',
                         label=sigma_label)

    #TODO: fit the view to the cost function, don't expand to fit in the uncertainty

    plot_prediction_through(trial.x, label=True, alphas=(1, 0.25))

    # plot the predictions through each sample
    if predict_through_all:
        locs = _choose_predict_locations(trial.x, f_xs, param_index)
        if len(locs) > 0:
            # cap to make sure they don't become invisible
            alpha = max(0.4/locs.shape[0], 0.015)
            for p in locs:
                plot_prediction_through(row2D(p), label=False, alphas=(alpha, alpha))

    ### Plot Vertical Bars
    bar_width, bar_color = 1, '#3590ce'
    ax1.axvline(x=trial_p, linewidth=bar_width, color=bar_color)
    ax1.plot(trial_p, trial.y, 'bo', markersize=6, alpha=0.4, label='this trial')

    ax1.legend()

    #################################
    # Plotting Acquisition Function
    #################################
    ax2.set_xlabel('parameter {}{}'.format(param, ' (latent space)' if plot_in_latent_space else ''))
    ax2.set_ylabel('{}({})'.format(acq_fun.get_name(), param))
    ax2.set_title('Acquisition Function ({})'.format(acq_fun.get_name()))

    ax2.plot(plot_range, trial_ac, '-', color='g', linewidth=1.0,
             label='acquisition function')
    ax2.fill_between(plot_range, np.zeros_like(plot_range), trial_ac,
                     alpha=0.3, color='palegreen')

    ax2.axvline(x=trial_p, linewidth=bar_width, color=bar_color)
    ax2.plot(trial_p, trial.extra_data['ac_x'], '^', color='black',
             markersize=7, zorder=10, label='this trial')

    ax2.legend()
    return fig


#TODO 2D plot

