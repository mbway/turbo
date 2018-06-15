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
import turbo.gui.jupyter as tg#TODO: need to sort out this dependency
from turbo.utils import row_2d, unique_rows_close
from .config import Config

#TODO: instead of calling these 'trial' plots, how about 'parameter plots'


#TODO: could use k-means to choose N locations to plot the surrogate through to get the best coverage of interesting regions while using as few plots as possible
#TODO: plots could use PlottingRecorder.unfinished_trial_nums to ignore any unfinished trials. This would allow for running the optimiser in the background while plotting before the optimiser has finished.
#TODO: color points by their trial number


#TODO: allow this functionality to be specified by passing in the necessary information and having a list of (point, opacity) returned. Then pass in what to plot through as a parameter
def _choose_predict_locations(trial_x, finished_xs, param_index):
    '''return a matrix with the points to predict through as rows

    avoid drawing predictions of the same place more than once, so
    avoid duplicate configurations which are identical to another
    except for the value of 'param', since the plot varies this
    parameter: the resulting plot will be the same in both cases.

    Args:
        trial_x: the point for the input of the current trial
        finished_xs: points for the inputs of finished trials
        param_index: the index into the space which finished_xs resides in, for
            the desired parameter
    '''
    # a copy of the current samples with the focused parameter zeroed
    # start with s.next_x since that is a point which is guaranteed to
    # have a prediction plotted through it
    param_zeroed = np.vstack([row_2d(trial_x)] + finished_xs)
    param_zeroed[:, param_index] = 0
    param_zeroed = unique_rows_close(param_zeroed, close_tolerance=1e-3)
    param_zeroed = param_zeroed[1:, :] # exclude the trial point (first row)
    return param_zeroed

#TODO: move the interactive stuff to turbo.gui.jupyter where it belongs
#TODO: instead of providing/not providing parameters to add/remove widgets. Always display the widgets but change the default value based on what is passed in
def interactive_plot_trial_1D(rec, param=None, trial_num=None, plot_in_latent_space=None, *args, **kwargs):
    '''choose the param and trial_num for the trial plot interactively

    param and trial_num can be left as None to be chosen interactively, or can be specified

    Args:
        rec (PlottingRecorder): the recorder which observed the run of an optimiser
        param: None to be chosen interactively, or can be specified
        trial_num: None to be chosen interactively, or can be specified
        plot_in_latent_space: None to be chosen interactively, or can be specified
    '''
    opt = rec.optimiser
    max_trial_num = max(rec.trials.keys())
    # partially apply by specifying all the non-interactive parameters
    plot_params = (args, kwargs) # these names have special meanings, so store them under a different name
    plot = lambda param, trial_num, latent: \
        plot_trial_1D(rec, param, trial_num, *plot_params[0], plot_in_latent_space=latent, **plot_params[1])
    # store the plots so they only have to be rendered once
    memoised = tg.PlotMemoization(plot)

    # updated by the widgets, or left alone if not None
    def update_plot(param, trial_num, space):
        key = (param, trial_num, space)
        if None not in key: # all args set
            latent = bool(space == 'Latent')
            memoised.show(key, {'param': param, 'trial_num': trial_num, 'latent': latent})

    if param is None:
        params = [b[0] for b in opt.bounds.ordered]
        param_w = tg.dropdown(params, description='Param:', initial=0)
    else:
        param_w = tg.widgets.fixed(param)

    if plot_in_latent_space is None:
        space_w = tg.widgets.ToggleButtons(options=['Input', 'Latent'], description='Space:', value='Latent')
    else:
        space_w = tg.widgets.fixed('Latent')

    if trial_num is None:
        trial_num_w = tg.slider(max_trial_num+1, description='Trial:', initial=-1)
    else:
        trial_num_w = tg.widgets.fixed(trial_num)

    # interactive is a widget, specifically a subclass of VBox
    controls = tg.widgets.interactive(update_plot, param=param_w, space=space_w, trial_num=trial_num_w)
    if param is None and plot_in_latent_space is None:
        output = controls.children[-1]
        param_row = tg.widgets.HBox([param_w, space_w]) if param is None else space_w
        if trial_num is None:
            controls.children = (param_row, trial_num_w, output)
        else:
            controls.children = (param_row, output)
    tg.display(controls)

def interactive_plot_trial_2D(rec, x_param=None, y_param=None, trial_num=None, plot_in_latent_space=None, *args, **kwargs):
    '''choose the param and trial_num for the trial plot interactively

    param and trial_num can be left as None to be chosen interactively, or can be specified

    Args:
        rec (PlottingRecorder): the recorder which observed the run of an optimiser
        x_param: None to be chosen interactively, or can be specified
        y_param: None to be chosen interactively, or can be specified
        trial_num: None to be chosen interactively, or can be specified
        plot_in_latent_space: None to be chosen interactively, or can be specified
    '''
    opt = rec.optimiser
    max_trial_num = max(rec.trials.keys())
    # partially apply by specifying all the non-interactive parameters
    plot_params = (args, kwargs) # these names have special meanings, so store them under a different name
    plot = lambda x_param, y_param, trial_num, x_latent, y_latent: \
        plot_trial_2D(rec, x_param, y_param, trial_num, *plot_params[0], plot_in_latent_space=(x_latent, y_latent), **plot_params[1])
    # store the plots so they only have to be rendered once
    memoised = tg.PlotMemoization(plot)

    # updated by the widgets, or left alone if not None
    def update_plot(x_param, y_param, trial_num, x_space, y_space):
        if x_param == y_param:
            print('error: x == y')
            return
        key = (x_param, y_param, trial_num, x_space, y_space)
        if None not in key: # all args set
            memoised.show(key, {
                'x_param': x_param, 'y_param': y_param,
                'trial_num': trial_num,
                'x_latent': bool(x_space == 'Latent'), 'y_latent': bool(y_space == 'Latent')
            })

    params = [b[0] for b in opt.bounds.ordered]

    x_param_w = tg.dropdown(params, description='X Param:', initial=0) if x_param is None else tg.widgets.fixed(x_param)
    y_param_w = tg.dropdown(params, description='Y Param:', initial=1) if y_param is None else tg.widgets.fixed(y_param)

    x_space_w = tg.widgets.ToggleButtons(options=['Input', 'Latent'], description='X Space:', value='Latent') \
            if plot_in_latent_space is None else tg.widgets.fixed('Latent')

    y_space_w = tg.widgets.ToggleButtons(options=['Input', 'Latent'], description='Y Space:', value='Latent') \
            if plot_in_latent_space is None else tg.widgets.fixed('Latent')

    trial_num_w = tg.slider(max_trial_num+1, description='Trial:', initial=-1) if trial_num is None else tg.widgets.fixed(trial_num)

    # interactive is a widget, specifically a subclass of VBox
    controls = tg.widgets.interactive(update_plot,
                                        x_param=x_param_w, y_param=y_param_w,
                                        x_space=x_space_w, y_space=y_space_w,
                                        trial_num=trial_num_w)
    # interactively selecting latent space => need custom rows
    if plot_in_latent_space is None:
        output = controls.children[-1]
        x_param_row = tg.widgets.HBox([x_param_w, x_space_w]) if x_param is None else x_space_w
        y_param_row = tg.widgets.HBox([y_param_w, y_space_w]) if y_param is None else y_space_w
        if trial_num is None:
            controls.children = (x_param_row, y_param_row, trial_num_w, output)
        else:
            controls.children = (x_param_row, y_param_row, output)
    tg.display(controls)



class DecodedTrial:
    ''' A utility for collecting data regarding a trial for use with plotting '''
    def __init__(self, trial_num, rec, plot_through):
        opt = rec.optimiser
        self.finished_trials, self.trial = rec.get_data_for_trial(trial_num)

        info = self.trial.selection_info
        trial_type = info['type']
        assert trial_type in ('pre_phase', 'bayes', 'fallback'), 'unknown trial type: {}'.format(trial_type)
        self.is_pre_phase = bool(trial_type == 'pre_phase')
        self.is_bayes = bool(trial_type == 'bayes')
        self.is_fallback = bool(trial_type == 'fallback')

        if self.is_fallback:
            self.fallback_reason = info['fallback_reason']
            if self.fallback_reason == 'too_close':
                # the point chosen by Bayesian optimisation that was too close
                # to an already tested point and so was discarded.
                self.bayes_x = info['bayes_x']

        self.has_surrogate = self.is_bayes or (self.is_fallback and self.fallback_reason == 'too_close')
        if self.has_surrogate:
            self.model = info['model']
            self.acq_fun = rec.get_acquisition_function(trial_num)
            self.acq_x = info['maximisation_info']['max_acq']

        self.finished_xs = [f.x.flatten() for f in self.finished_trials] # in latent space
        self.finished_costs = [f.y for f in self.finished_trials]

        if len(self.finished_trials) > 0:
            costs = self.finished_costs
            self.incumbent_index = np.argmax(costs) if opt.is_maximising() else np.argmin(costs)
            self.incumbent_mask = np.ones_like(costs, dtype=bool)
            self.incumbent_mask[self.incumbent_index] = False

        if isinstance(plot_through, int):
            self.plot_through = rec.trials[plot_through].x
        elif plot_through == 'trial':
            self.plot_through = self.trial.x
        elif plot_through == 'incumbent':
            self.plot_through = row_2d(self.finished_xs[self.incumbent_index])
        else:
            raise ValueError(plot_through)

        # used for the plot title
        if self.is_bayes:
            self.extra_text = ''
            self.title_color = 'black'
        elif self.is_pre_phase:
            self.extra_text = ' (pre_phase)'
            self.title_color = 'violet'
        elif self.is_fallback:
            self.extra_text = ' (fallback, reason: {})'.format(self.fallback_reason)
            self.title_color = 'red'

class DecodedParam:
    '''A utility for collecting data regarding a single parameter for use with plotting

    Attributes:
        name: the name of the parameter (in the input space Bounds)
        latent_name: the name of the parameter in latent space corresponding to `self.name`
        index: the index of the parameter in the input space `Bounds`
        latent_index: the index of latent_name in the latent space `Bounds`
        latent_range: the latent range for the parameter (may be linear in input or latent space)
        input_range: the input space range for the parameter (may be linear in input or latent space)
        plot_range: the range used for plotting (either latent or input)
        plot_bounds: the bounds for this param in the space used for plotting (either latent or input)
        finished_vals: a list of the values of this parameter for the finished trials (in the space being plotted)
        trial_val: the value of this parameter for the current trial (in the space being plotted)
        line_through_trial: an array of points formed by moving from the minimum
            to the maximum of the bounds of this parameter in latent space,
            passing through the current trial.
        label: used to describe the parameter on an axis label
    '''
    def __init__(self, param_name, rec, divisions, plot_in_latent_space, finished_xs, trial):
        '''
        Args:
            param_name: the name of the parameter to decode the information for
            rec: the recorder to pull data from
            plot_in_latent_space: whether this parameter is to be plotted in latent or input space
            finished_xs: the input points for the finished trials (1D array)
            trial (Trial): the current trial
        '''
        self._rec = rec
        self._plot_in_latent_space = plot_in_latent_space

        opt = rec.optimiser
        self.name = param_name
        latent_bounds = opt.latent_space.get_latent_bounds()
        latent_names = opt.latent_space.get_latent_param_names(self.name)
        assert len(latent_names) == 1 # assumption, have to rework otherwise
        self.latent_name = latent_names[0]

        self.index = opt.bounds.get_param_index(self.name)
        self.latent_index = latent_bounds.get_param_index(self.latent_name)

        if plot_in_latent_space:
            # linear in the latent space, transform this back to get the points in the input space
            self.latent_range = opt.latent_space.linear_latent_range(param_name, divisions) # the input space range
            self.input_range = np.array([opt.latent_space.param_from_latent(param_name, p) for p in self.latent_range])
        else:
            # linear in the input space, transform this to get the points in the latent space
            self.input_range = np.linspace(*opt.bounds.get(param_name), num=divisions) # the input space range
            self.latent_range = np.array([opt.latent_space.param_to_latent(param_name, p) for p in self.input_range])

        self.plot_range = self.latent_range if plot_in_latent_space else self.input_range
        self.plot_bounds = latent_bounds.get(self.latent_name) if plot_in_latent_space else opt.bounds.get(param_name)
        self.finished_vals = [self.get_plot_val(x) for x in finished_xs]
        self.trial_val = self.get_plot_val(trial.x)
        self.label = 'parameter {}{}'.format(self.name, ' (latent space)' if plot_in_latent_space else '')

    def get_plot_val(self, latent_point):
        '''extract the value of this parameter from the given latent space point and return the value in the space being plotted '''
        val = latent_point.flatten()[self.latent_index]
        return val if self._plot_in_latent_space else self._rec.optimiser.latent_space.param_from_latent(self.name, val)

    def latent_line_through(self, point):
        '''generate a list of points which span the latent range of this
        parameter, forming a line and passing through the given point.

        Returns:
            a matrix with the points as rows
        '''
        latent_range = self.latent_range.flatten()
        # create many duplicates of the given point
        points = np.repeat(row_2d(point), len(latent_range), axis=0)
        points[:,self.latent_index] = latent_range
        return points


def plot_trial_1D(rec, param, trial_num, true_objective=None,
                  plot_in_latent_space=True, divisions=200, n_sigma=2,
                  predict_through_all=True, plot_through='trial', ylim=None, fig=None):
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
        true_objective: true objective function which takes an input space as an
            argument (or array of pre-computed cost values corresponding to the
            number of divisions) (None to omit)
        plot_in_latent_space: whether to plot in latent space or input space
        divisions (int): the resolution of the plot / number of points along the
            parameter domain to plot at. (higher => slightly better quality but
            takes longer to render)
        n_sigma (float): the number of standard deviations from the mean to plot
            the uncertainty confidence interval.

            pass 'beta' to use the beta parameter of UCB/LCB (only possible when
            using this acquisition function)

            .. note::
                for a normal distribution (i.e. GP surrogate):
                :math:`1\implies 68\%,\;2\implies 95\%,\;3\implies 99\%`
        predict_through_all: whether to plot a surrogate prediction through
            every sample or just through the location of the point chosen this
            iteration.
        plot_through: the point which the main (most significant) prediction
            plot passes through. either 'trial' (current trial) or 'incumbent
            (best trial so far) or a trial number (which can be after the trial
            being plotted). The acquisition function always passes through the
            current trial since there would not be much use in doing anything
            else.
        ylim: when specified, set the limits of the y/cost axis to
            get a better detailed look at that range of values (optional)
        fig: the matplotlib figure to plot onto
    '''
    # to catch errors where the user mistakes this for the interactive version
    assert param is not None and trial_num is not None, 'param and trial_num must be specified'

    assert not rec.has_unfinished_trials()
    opt = rec.optimiser
    assert opt.async_eval is None, 'this function does not support asynchronous optimisation runs'
    max_trial_num = max(rec.trials.keys())
    # allow negative trial numbers for referring to the end
    trial_num = max_trial_num+1 + trial_num if trial_num < 0 else trial_num
    assert trial_num >= 0 and trial_num <= max_trial_num, 'invalid trial number'

    ################
    # Extract Data
    ################
    t = DecodedTrial(trial_num, rec, plot_through)
    param = DecodedParam(param, rec, divisions, plot_in_latent_space, t.finished_xs, t.trial)

    if t.has_surrogate:
        #TODO: like predictions_through_all, might want to plot the acquisition function on multiple lines
        # it makes sense to plot the acquisition function through the slice
        # corresponding to the current trial.
        acq_through_trial = t.acq_fun(param.latent_line_through(t.trial.x))
        if n_sigma == 'beta':
            assert isinstance(opt.acq_func_factory, tm.UCB.Factory), \
                'n_sigma == "beta" only possible when using the UCB/LCB acquisition function'
            n_sigma = rec.trials[trial_num].selection_info['acq_info']['beta']

    if t.is_fallback and t.fallback_reason == 'too_close':
        # the value of the parameter for the point selected by Bayesian
        # optimisation that was discarded
        bayes_val = param.get_plot_val(t.bayes_x)

    # TODO: can use plot_through to provide all the parameters to the function when plotting a multi-dimensional function
    if true_objective is not None:
        # true cost is either the cost function, or pre-computed costs as an array
        if callable(true_objective):
            true_costs = [true_objective(p) for p in param.input_range]
        else:
            assert len(true_objective) == divisions, 'true_objective has the wrong length'
            true_costs = true_objective

    ##################
    # Plotting Setup
    ##################
    fig = fig or plt.figure(figsize=Config.fig_sizes['1D_trial'])
    grid = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2, 1])
    ax1, ax2 = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])

    fig.suptitle('Bayesian Optimisation Trial {}{}'.format(trial_num, t.extra_text), fontsize=14, color=t.title_color)
    for ax in (ax1, ax2):
        # lim and margins are mutually exclusive, so have to calculate margins manually...
        xmin, xmax = param.plot_bounds
        margin = (xmax-xmin)*0.005
        ax.set_xlim((xmin-margin, xmax+margin))
        ax.margins(y=0.04)
    if ylim is not None:
        ax1.set_ylim(ylim)
    fig.subplots_adjust(hspace=0.3)

    ax1.set_ylabel('objective')
    ax1.set_title('Surrogate Objective Function')

    ax2.set_xlabel(param.label)
    if t.has_surrogate:
        ax2.set_ylabel('{}({})'.format(t.acq_fun.get_name(), param.name))
        ax2.set_title('Acquisition Function ({})'.format(t.acq_fun.get_name()))

    bar_width, bar_color = 1.5, '#3590ce'

    ##########
    # Top Axes
    ##########
    if true_objective is not None:
        ax1.plot(param.plot_range, true_costs, '--', color='#2f2f2f',
                label='true objective', linewidth=1.0, alpha=0.6)

    # plot samples projected onto the `param` axis
    # exclude the incumbent since that is plotted separately
    if len(t.finished_trials) > 0:
        ax1.plot(np.vstack(param.finished_vals)[t.incumbent_mask,:], np.array(t.finished_costs)[t.incumbent_mask],
                'o', markersize=6, label='finished trials', color=Config.trial_marker_colors['bayes'], zorder=5)

        ax1.plot(param.finished_vals[t.incumbent_index], t.finished_costs[t.incumbent_index],
                '*', markersize=10, color=Config.trial_marker_colors['incumbent'], zorder=10, markeredgewidth=0.5, markeredgecolor=Config.trial_edge_colors['incumbent'], label='incumbent')

    ax1.axvline(x=param.trial_val, linewidth=bar_width, color=bar_color)
    if t.is_fallback and t.fallback_reason == 'too_close':
        ax1.axvline(x=bayes_val, linewidth=bar_width, color='orange')
    ax1.plot(param.trial_val, t.trial.y, 'bo', markersize=6, alpha=0.4, markeredgecolor=Config.trial_edge_colors['bayes'], label='this trial')

    if not t.has_surrogate:
        # finish early
        ax1.legend()
        return fig

    #TODO: pull out
    def plot_prediction_through(point, label, mu_alpha, sigma_alpha):
        line = param.latent_line_through(point)
        mus, sigmas = t.model.predict(line, return_std_dev=True)
        mu_label = r'$\mu$' if label else None
        sigma_label = r'${}\sigma$'.format(n_sigma) if label else None

        ax1.plot(param.plot_range, mus, '-', color='#b055de', label=mu_label, alpha=mu_alpha, linewidth=1.0)
        ax1.fill_between(param.plot_range, mus - n_sigma*sigmas, mus + n_sigma*sigmas,
                        alpha=sigma_alpha, color='#c465f3', label=sigma_label)

    #TODO: fit the view to the cost function, don't expand to fit in the uncertainty

    # the most significant prediction plot
    plot_prediction_through(t.plot_through, label=True, mu_alpha=1, sigma_alpha=0.25)

    # plot the predictions through each sample
    #TODO: change name since not predicting through all any more
    if predict_through_all:
        #TODO: more intelligent choosing for plotting locations
        locs = _choose_predict_locations(t.trial.x, t.finished_xs, param.latent_index)
        if len(locs) > 0:
            # cap to make sure they don't become invisible
            alpha = max(0.4/locs.shape[0], 0.015)
            for p in locs:
                plot_prediction_through(row_2d(p), label=False, mu_alpha=alpha, sigma_alpha=alpha)

    #############
    # Bottom Axes
    #############
    ax2.plot(param.plot_range, acq_through_trial, '-', color='g', linewidth=1.0,
            label='acquisition function')
    ax2.fill_between(param.plot_range, np.zeros_like(param.plot_range), acq_through_trial,
                    alpha=0.3, color='palegreen')

    ax2.axvline(x=param.trial_val, linewidth=bar_width, color=bar_color)
    if t.is_fallback and t.fallback_reason == 'too_close':
        ax2.axvline(x=bayes_val, linewidth=bar_width, color='orange')
        ax2.plot(bayes_val, t.acq_x, '^', color='orange',
                markersize=7, zorder=10)
    elif t.is_bayes:
        ax2.plot(param.trial_val, t.acq_x, '^', color='black',
                markersize=7, zorder=10, label='this trial')


    ax1.legend()
    ax2.legend()
    return fig


class Decoded2Params:
    ''' A utility for collecting data for 2 decoded parameters for use with 2D plotting '''
    def __init__(self, x_param, y_param, plot_through, divisions):
        self.x, self.y = x_param, y_param
        self.grid_size = divisions # (x_divisions, y_divisions)

        self.finished_vals = np.column_stack((self.x.finished_vals, self.y.finished_vals))

        # used for constructing the latent_plane_points, which the model uses to predict with
        self.latent_grid = np.meshgrid(self.x.latent_range, self.y.latent_range)
        self.latent_points = self.grid_to_points(self.latent_grid)

        # used for passing to the true objective function
        self.input_grid = np.meshgrid(self.x.input_range, self.y.input_range)
        self.input_points = self.grid_to_points(self.input_grid)

        # used by the model for prediction
        self.latent_plane_points = self.latent_plane_through(plot_through)

    def grid_to_points(self, grid):
        ''' take a grid generated with `np.meshgrid` and return every point on that grid as a row of a matrix '''
        # vstack then transpose is different to just hstack because the stacking behaves differently because of the shape
        return np.vstack((grid[0].ravel(), grid[1].ravel())).T
    def points_to_grid(self, points):
        ''' take a matrix of points generated with grid_to_points and return it to a grid'''
        return points.reshape(*self.grid_size)

    def latent_plane_through(self, point):
        '''generate a list of points which span the latent range of both
        parameters, forming a plane and passing through the given point.

        Returns:
            a matrix with the points as rows
        '''
        # create many duplicates of the given point
        points = np.repeat(row_2d(point), self.latent_points.shape[0], axis=0)
        points[:,self.x.latent_index] = self.latent_points[:,0]
        points[:,self.y.latent_index] = self.latent_points[:,1]
        return points




#TODO: documentation
def plot_trial_2D(rec, x_param, y_param, trial_num, true_objective=None,
                  plot_in_latent_space=(True, True), divisions=(100, 100),
                  plot_through='trial', mu_norm=None, fig=None):
    '''
    TODO:
    Args:
        true_objective: either a function which takes an input space point as an
            argument, or a grid of pre-computed cost values corresponding to `divisions`
        plot_through: The point which the plane being plotted passes through.
            either 'trial' (current trial) or 'incumbent' (best trial so far) or
            a trial number (which can be after the trial being plotted).
    '''
    # to catch errors where the user mistakes this for the interactive version
    assert x_param is not None and y_param is not None and trial_num is not None, 'params and trial_num must be specified'
    assert x_param != y_param, 'x_param and y_param must be different'
    if tg.in_jupyter():
        assert not tg.using_svg_backend(), 'don\'t use the SVG backend with 2D plots. The results are unmanageably large files'

    assert not rec.has_unfinished_trials()
    opt = rec.optimiser
    assert opt.async_eval is None, 'this function does not support asynchronous optimisation runs'
    max_trial_num = max(rec.trials.keys())
    # allow negative trial numbers for referring to the end
    trial_num = max_trial_num+1 + trial_num if trial_num < 0 else trial_num
    assert trial_num >= 0 and trial_num <= max_trial_num, 'invalid trial number'

    ################
    # Extract Data
    ################
    t = DecodedTrial(trial_num, rec, plot_through)
    x_param = DecodedParam(x_param, rec, divisions[0], plot_in_latent_space[0], t.finished_xs, t.trial)
    y_param = DecodedParam(y_param, rec, divisions[1], plot_in_latent_space[1], t.finished_xs, t.trial)

    params = Decoded2Params(x_param, y_param, t.plot_through, divisions)

    if t.has_surrogate:
        mus_points, sigmas_points = t.model.predict(params.latent_plane_points, return_std_dev=True)
        mus, sigmas = params.points_to_grid(mus_points), params.points_to_grid(sigmas_points)

        acq_points = t.acq_fun(params.latent_plane_points)
        acq = params.points_to_grid(acq_points)

    if t.is_fallback and t.fallback_reason == 'too_close':
        # the value of the parameter for the point selected by Bayesian
        # optimisation that was discarded
        bayes_x_val = x_param.get_plot_val(t.bayes_x)
        bayes_y_val = y_param.get_plot_val(t.bayes_x)

    if true_objective is not None:
        # true cost is either the cost function, or pre-computed costs as an array
        if callable(true_objective):
            # when using positional arguments it is easier to get confused by
            # passing arguments in the wrong order based on which is the x and y
            # axes of the plot.
            objective_args = [{x_param.name: p[0], y_param.name: p[1]} for p in params.input_points]
            true_cost_points = np.array([true_objective(**a) for a in objective_args])
            true_costs = params.points_to_grid(true_cost_points)
        else:
            assert true_objective.shape == divisions, 'true_objective has the wrong shape'
            true_costs = true_objective

    ##################
    # Plotting Setup
    ##################
    #TODO: profiling (with time.time()) shows that setting up the axes takes longer than plotting or setup!
    fig = fig or plt.figure(figsize=Config.fig_sizes['2D_trial'])
    grid = gridspec.GridSpec(nrows=2, ncols=2)
    # layout: ax1 ax2
    #         ax3 ax4
    ax1, ax2 = fig.add_subplot(grid[0]), fig.add_subplot(grid[1]) if true_objective is not None else None
    ax3, ax4 = fig.add_subplot(grid[2]), fig.add_subplot(grid[3])

    for ax in fig.axes:
        ax.set_xlim(x_param.plot_bounds)
        ax.set_ylim(y_param.plot_bounds)
        ax.grid(False)

    fig.suptitle('Bayesian Optimisation Trial {}{}'.format(trial_num, t.extra_text), fontsize=20, color=t.title_color)
    # need to specify rect so that the suptitle isn't cut off
    fig.tight_layout(h_pad=3, w_pad=8, rect=[0, 0, 1, 0.96]) # [left, bottom, right, top] 0-1


    ############
    # Plotting
    ############
    #TODO: pull out
    def plot_trials(ax):
        ax.set_xlabel(x_param.label)
        ax.set_ylabel(y_param.label)
        if len(t.finished_trials) > 0:
            ax.plot(np.vstack(x_param.finished_vals)[t.incumbent_mask,:],
                    np.vstack(y_param.finished_vals)[t.incumbent_mask,:],
                    'ro', markersize=4, linestyle='None', label='finished trials')

            ax.plot(x_param.finished_vals[t.incumbent_index], y_param.finished_vals[t.incumbent_index],
                    '*', markersize=15, color='deepskyblue', zorder=10,
                    markeredgecolor='black', markeredgewidth=1.0, label='incumbent')

        ax.plot(x_param.trial_val, y_param.trial_val, marker='X', color='red',
                markeredgecolor='black', markeredgewidth=1.0, markersize=10, zorder=11,
                linestyle='None', label='this trial')

        if t.is_fallback and t.fallback_reason == 'too_close':
            ax.plot(bayes_x_val, bayes_y_val, marker='s', color='orange',
                markeredgecolor='black', markeredgewidth=1.0, markersize=10,
                linestyle='None', label='Bayes suggestion')


    def plot_heatmap(ax, data, colorbar, cmap, norm=None):
        # pcolormesh is better than imshow because: no need to fiddle around
        # with extents and aspect ratios because the x and y values can be
        # fed in and so just works. This also prevents the problem of the
        # origin being in the wrong place. It is compatible with log scaled
        # axes unlike imshow. There is no interpolation by default unlike
        # imshow.
        im = ax.pcolormesh(x_param.plot_range, y_param.plot_range, data, cmap=cmap, norm=norm)
        if colorbar:
            levels = norm.levels() if norm is not None else None
            c = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.051, boundaries=levels)
            c.set_label('cost')

    title_size = 16
    cmap = 'viridis'
    # reverse the color map if minimising
    cmap_match_direction = cmap if opt.is_maximising() else cmap + '_r' # reversed

    if not t.has_surrogate:
        # different plot entirely
        if true_objective is not None:
            ax2.set_title('True Objective', fontsize=title_size)
            plot_heatmap(ax2, true_costs, colorbar=True, cmap=cmap_match_direction)
            ax = ax2
        else:
            # ax2 won't be available
            ax1.set_title(r'Trials', fontsize=title_size)
            ax = ax1
        plot_trials(ax)
        legend = ax.legend(frameon=True, fancybox=True, loc='lower left')
        legend.get_frame().set_facecolor('#ffffff')
        legend.get_frame().set_alpha(0.5)
        return fig

    ax1.set_title(r'Surrogate $\mu$', fontsize=title_size)
    plot_heatmap(ax1, mus, colorbar=True, cmap=cmap_match_direction, norm=mu_norm)
    plot_trials(ax1)

    if true_objective is not None:
        ax2.set_title('True Objective', fontsize=title_size)
        plot_heatmap(ax2, true_costs, colorbar=True, cmap=cmap_match_direction, norm=mu_norm)
        plot_trials(ax2)

    ax3.set_title(r'Surrogate $\sigma$', fontsize=title_size)
    plot_heatmap(ax3, sigmas, colorbar=True, cmap=cmap)
    plot_trials(ax3)

    ax4.set_title('Acquisition Function ({})'.format(t.acq_fun.get_name()), fontsize=title_size)
    plot_heatmap(ax4, acq, colorbar=True, cmap=cmap)
    plot_trials(ax4)

    if true_objective is None:
        ax4.legend(bbox_to_anchor=(0, 1.01), loc='lower left', borderaxespad=0.0)
    else:
        legend = ax2.legend(frameon=True, fancybox=True, loc='lower left')
        legend.get_frame().set_facecolor('#ffffff')
        legend.get_frame().set_alpha(0.5)

    return fig

