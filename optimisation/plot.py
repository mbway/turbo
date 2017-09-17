#!/usr/bin/env python3
'''
Plotting methods for Optimiser objects (extracted from the main class
definitions because these methods are long and just add noise)
'''
# python 2 compatibility
from __future__ import (absolute_import, division, print_function, unicode_literals)
from .py2 import *

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.transforms

from collections import defaultdict
from itertools import groupby

# local modules
from .utils import *
from .bayesian_utils import *
from . import plot3D

# note: Mixins have to be inherited in reverse, eg MyClass(Plotting, SuperClass)

class OptimiserPlotting:
    '''
    A mixin for providing generic plotting capabilities to an Optimiser object
    (applicable to Grid/Random/Bayes).
    '''

    def group_by_param(self, param_name):
        '''
        return [value, [sample]] for each unique value of the given parameter
        '''
        param_key = lambda sample: sample.config[param_name]
        data = []
        # must be sorted before grouping
        for val, samples in groupby(sorted(self.samples, key=param_key), param_key):
            data.append((val, list(samples)))
        return data

    def group_by_params(self, param_a, param_b):
        '''
        return [(value_a, value_b), [sample]] for each unique pair of values of the given parameters
        '''
        params_key = lambda sample: (sample.config[param_a], sample.config[param_b])
        data = []
        # must be sorted before grouping
        for val, samples in groupby(sorted(self.samples, key=params_key), params_key):
            data.append((val, list(samples)))
        return data

    def plot_cost_over_time(self, plot_each=True, plot_best=True, true_best=None):
        '''
        plot a line graph showing the progress that the optimiser makes towards
        the optimum as the number of samples increases.
        plot_each: plot the cost of each sample
        plot_best: plot the running-best cost
        true_best: if available: plot a horizontal line for the best possible cost
        '''
        fig, ax = plt.subplots(figsize=(16, 10)) # inches

        xs = range(1, len(self.samples)+1)
        costs = [s.cost for s in self.samples]

        if true_best is not None:
            ax.axhline(true_best, color='black', label='true best')

        if plot_best:
            chooser = max if self.maximise_cost else min
            best_cost = [chooser(costs[:x]) for x in xs]
            ax.plot(xs, best_cost, color='#55a868')

            best_x, best_cost = chooser(zip(xs, costs), key=lambda t: t[1])

            # move the marker out of the way.
            offset = 4.5 if self.maximise_cost else -4.5 # pt
            offset = matplotlib.transforms.ScaledTranslation(0, offset/fig.dpi,
                                                             fig.dpi_scale_trans)
            trans = ax.transData + offset

            marker = 'v' if self.maximise_cost else '^'
            ax.plot(best_x, best_cost, marker=marker, color='#55a868',
                    markersize=10, zorder=10, markeredgecolor='black',
                    markeredgewidth=1, label='best cost', transform=trans)

        if plot_each:
            ax.plot(xs, costs, marker='o', markersize=4, color='#4c72b0', label='cost')

        ax.set_title('Cost Over Time', fontsize=14)
        ax.set_xlabel('samples')
        ax.set_ylabel('cost')
        ax.margins(0.0, 0.15)
        if len(self.samples) < 50:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2.0))
        elif len(self.samples) < 100:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5.0))
        ax.legend()

        return fig

    def plot_param(self, param_name, plot_boxplot=True, plot_samples=True,
                   plot_means=True, log_axes=(False, False)):
        '''
        plot a boxplot of parameter values against cost
        plot_boxplot: whether to plot boxplots
        plot_samples: whether to plot each sample as a point
        plot_means: whether to plot a line through the mean costs
        log_axes: (xaxis,yaxis) whether to display the axes with a logarithmic scale
        '''
        values = []
        costs = []
        means = []
        for val, samples in self.group_by_param(param_name):
            values.append(val)
            c = [s.cost for s in samples]
            costs.append(c)
            means.append(np.mean(c))
        labels = ['{:.2g}'.format(v) for v in values]

        plt.figure(figsize=(16, 8))

        if plot_means:
            plt.plot(values, means, 'r-', linewidth=1, alpha=0.5)
        if plot_samples:
            xs, ys = zip(*[(s.config[param_name], s.cost) for s in self.samples])
            plt.plot(xs, ys, 'go', markersize=5, alpha=0.6)
        if plot_boxplot:
            plt.boxplot(costs, positions=values, labels=labels)

        plt.margins(0.1, 0.1)
        plt.xlabel('parameter: ' + param_name)
        plt.ylabel('cost')
        if log_axes[0]:
            plt.xscale('log')
        if log_axes[1]:
            plt.yscale('log')
        plt.autoscale(True)
        plt.show()

    #TODO: instead of 'interactive' pass an argument of how many points to show, then deal with the slider business outside of optimisation.py and plot3D.py
    def scatter_plot(self, param_a, param_b, interactive=True, color_by='cost',
                     log_axes=(False, False, False)):
        '''
            interactive: whether to display a slider for changing the number of
                samples to display
            color_by: either 'cost' or 'age'
            log_axes: whether to display the x,y,z axes with a logarithmic scale
        '''
        assert color_by in ['cost', 'age']

        xs, ys, costs, texts = [], [], [], []
        for i, s in enumerate(self.samples):
            xs.append(s.config[param_a])
            ys.append(s.config[param_b])
            costs.append(s.cost)
            texts.append('sample {:03}, config: {}, cost: {}'.format(i+1, config_string(s.config), s.cost))

        xs, ys, costs, texts = map(np.array, (xs, ys, costs, texts))
        color = 'z' if color_by == 'cost' else 'age'
        axes_names = ['param: ' + param_a, 'param: ' + param_b, 'cost']

        plot3D.scatter3D(xs, ys, costs, interactive=interactive, color_by=color,
                         markersize=4, tooltips=texts, axes_names=axes_names,
                         log_axes=log_axes)

    def surface_plot(self, param_a, param_b, log_axes=(False, False, False)):
        '''
        plot the surface of different values of param_a and param_b and how they
        affect the cost (z-axis). If there are multiple configurations with the
        same combination of param_a,param_b values then the minimum is taken for
        the z/cost value.

        This method does not require that in self.samples there is complete
        coverage of all param_a and param_b values, or that the samples have a
        particular ordering.

        If there are gaps where a param_a,param_b combination has not yet been
        evaluated, the cost for that point will be 0.

        log_axes: whether to display the x,y,z axes with a logarithmic scale
        '''
        # get all the x and y values found in any of the samples (may not equal self.ranges[...])
        xs = np.array(sorted(set([val for val, samples in self.group_by_param(param_a)])))
        ys = np.array(sorted(set([val for val, samples in self.group_by_param(param_b)])))
        costs = defaultdict(float) # if not all combinations of x and y are available: cost = 0
        texts = defaultdict(lambda: 'no data')
        for val, samples_for_val in self.group_by_params(param_a, param_b):
            sample = min(samples_for_val, key=lambda s: s.cost)
            costs[val] = sample.cost
            texts[val] = 'config: {}, cost: {}'.format(config_string(sample.config), sample.cost)
        xs, ys = np.meshgrid(xs, ys)
        costs = np.vectorize(lambda x, y: costs[(x, y)])(xs, ys)
        texts = np.vectorize(lambda x, y: texts[(x, y)])(xs, ys)
        axes_names = ['param: ' + param_a, 'param: ' + param_b, 'cost']
        plot3D.surface3D(xs, ys, costs, tooltips=texts, axes_names=axes_names, log_axes=log_axes)




class BayesianOptimisationOptimiserPlotting:
    '''
    a mixin for providing Bayesian Optimisation specific plotting capabilities
    to a BayesianOptimisationOptimiser
    '''

    def _points_vary_one(self, point, param, xs):
        '''
        points generated by fixing all but one parameter to match the given
        config, and varying the remaining parameter.
        point: the configuration represented in point space to base the points on
        param: the name of the parameter to vary
        xs: the values to provide in place of config[param]
        '''
        assert len(xs.shape) == 1 # not 2D
        # create many duplicates of the point version of the given configuration
        points = np.repeat(point, len(xs), axis=0)
        param_index = self.point_space.param_indices[param] # column to swap out for xs
        points[:,param_index] = xs
        return points

    def plot_step_1D(self, param, step, true_cost=None, n_sigma=2, gp_through_all=True):
        '''
        plot a Bayesian optimisation step, perturbed along a single parameter.

        the intuition for the case of a 1D configuration space is trivial: the
        plot is simply the parameter value and the corresponding cost and
        acquisition values. in 2D, imagine the surface plot of the two
        parameters against cost (as the height). This plot takes a cross section
        of that surface along the specified axis and passing through the point
        of the next configuration to test to show how the acquisition function
        varies along that dimension. The same holds for higher dimensions but
        is harder to visualise.

        param: the name of the parameter to perturb to obtain the graph
        step: the number of the step to plot
        true_cost: true cost function (or array of pre-computed cost values
            corresponding to self.ranges[param]) (None to omit)
        n_sigma: the number of standard deviations from the mean to plot the
            uncertainty confidence interval.
            Note 1=>68%, 2=>95%, 3=>99% (for a normal distribution, which this is)
        gp_through_all: whether to plot a gp prediction through every sample or
            just through the location of the next point to be chosen
        '''
        assert step in self.step_log, 'step not recorded in the log'

        range_ = self.point_space.ranges[param]
        assert range_.type_ in [RangeType.Linear, RangeType.Logarithmic]

        s = self.step_log[step]
        num_suggestions = len(s.suggestions)
        all_xs = range_.values

        # cases for different types of steps. Only certain cases are accounted
        # for, because only these cases should happen under normal circumstances.
        if not (num_suggestions == 1 or num_suggestions == 2):
            raise ValueError(num_suggestions)

        # extract and process the data from the first suggestion
        sg1 = s.suggestions[0]
        if isinstance(sg1, Step.MaxAcqSuggestion):
            # combination of the concrete samples (sx, sy) and the hypothesised samples
            # (hx, hy) if there are any
            xs = np.vstack((s.sx, s.hx))
            ys = np.vstack((s.sy, sg1.hy))

            # restore from just the hyperparameters to a GP which can be queried
            gp_model = restore_GP(sg1.gp, self.gp_params, xs, ys)

            acq_fun = self._get_acq_fun(gp_model, ys)
            sims = False # no simulations for this strategy

            # the GP is trained in point space
            gp_xs = self.point_space.param_to_point_space(all_xs, param)

            # the values for the chosen parameter for the concrete and
            # hypothesised samples in config space.
            concrete_xs = self.point_space.param_to_config_space(s.sx, param).flatten()
            concrete_ys = s.sy
            hypothesised_xs = self.point_space.param_to_config_space(s.hx, param).flatten()
            hypothesised_ys = sg1.hy

            # the current best concrete sample (x is only the value along the
            # chosen parameter in config space)
            best_i = np.argmax(concrete_ys) if self.maximise_cost else np.argmin(concrete_ys)
            best_concrete_x = concrete_xs[best_i]
            best_concrete_y = concrete_ys[best_i]

            # next point chosen by the acquisition function to be evaluated
            acq_chosen_point = sg1.x
            acq_chosen_x = self.point_space.param_to_config_space(acq_chosen_point, param)
            acq_chosen_ac = sg1.ac

            # the final choice of the step, replaced by the random fallback
            # choice if there was one
            chosen_point = acq_chosen_point
            chosen_x = acq_chosen_x

        elif isinstance(sg1, Step.MC_MaxAcqSuggestion):
            # combination of the concrete samples (sx, sy) and the hypothesised samples
            # (hx, hy) if there are any
            xs = np.vstack((s.sx, s.hx))

            # restore from just the hyperparameters to a GP which can be queried
            gp_model = restore_GP(sg1.gp, self.gp_params, s.sx, s.sy)

            sims = True # simulations used in this strategy
            sim_gps = []
            sim_ac_funs = [] # the acquisition functions for each simulation
            for hy, sim_gp in sg1.simulations:
                ys = np.vstack((s.sy, hy))
                sim_gp = sim_gp if sim_gp is not None else sg1.gp
                # fit the GP to the points of this simulation
                sim_gp = restore_GP(sim_gp, self.gp_params, xs, ys)
                acq = self._get_acq_fun(sim_gp, ys) # partially apply
                sim_ac_funs.append(acq)
                sim_gps.append(sim_gp)

            # average acquisition across every simulation
            acq_fun = lambda xs: 1.0/len(sg1.simulations) * np.sum(acq(xs) for acq in sim_ac_funs)

            # the GP is trained in point space
            gp_xs = self.point_space.param_to_point_space(all_xs, param)

            # the values for the chosen parameter for the concrete and
            # hypothesised samples in config space.
            concrete_xs = self.point_space.param_to_config_space(s.sx, param).flatten()
            concrete_ys = s.sy
            hypothesised_xs = self.point_space.param_to_config_space(s.hx, param).flatten()
            hypothesised_xs = np.vstack([make2D(hypothesised_xs)] * len(sg1.simulations))
            hypothesised_ys = np.vstack([hy for hy,sim_gp in sg1.simulations])

            # the current best concrete sample (x is only the value along the
            # chosen parameter in config space)
            best_i = np.argmax(concrete_ys) if self.maximise_cost else np.argmin(concrete_ys)
            best_concrete_x = concrete_xs[best_i]
            best_concrete_y = concrete_ys[best_i]

            # next point chosen by the acquisition function to be evaluated
            acq_chosen_point = sg1.x
            acq_chosen_x = self.point_space.param_to_config_space(acq_chosen_point, param)
            acq_chosen_ac = sg1.ac

            # the final choice of the step, replaced by the random fallback
            # choice if there was one
            chosen_point = acq_chosen_point
            chosen_x = acq_chosen_x

        else:
            raise NotImplementedError()

        # extract and process the data from the second suggestion if there is one
        if num_suggestions == 2:
            sg2 = s.suggestions[1]
            if isinstance(sg2, Step.RandomSuggestion):
                random_fallback = True

                random_chosen_point = sg2.x
                random_chosen_x = self.point_space.param_to_config_space(random_chosen_point, param)

                chosen_point = random_chosen_point
                chosen_x = random_chosen_x

            else:
                raise NotImplementedError()
        else:
            random_fallback = False

        if acq_fun:
            # it makes sense to plot the acquisition function through the slice
            # corresponding to the next point to be chosen, whether it is the
            # suggestion by the acquisition function maximisation or the random
            # suggestion.
            gp_perturbed_points = self._points_vary_one(chosen_point, param, gp_xs)
            ac = acq_fun(gp_perturbed_points)

            if sims:
                sim_ac = [acq(gp_perturbed_points) for acq in sim_ac_funs]
                sim_mus = [gp.predict(gp_perturbed_points).flatten() for gp in sim_gps]

        fig = plt.figure(figsize=(16, 10)) # inches
        grid = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2, 1])
        ax1, ax2 = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])

        title = 'Bayesian Optimisation step {}'.format(
            step-self.strategy.pre_phase_steps)
        if random_fallback:
            title += ' (Random Fallback)'

        fig.suptitle(title, fontsize=14)
        ax1.margins(0.005, 0.05)
        ax2.margins(0.005, 0.05)
        if range_.type_ == RangeType.Logarithmic:
            ax1.set_xscale('log')
            ax2.set_xscale('log')
        fig.subplots_adjust(hspace=0.3)

        ax1.set_ylabel('cost')
        ax1.set_title('Surrogate objective function')

        ### Plot True Cost
        if true_cost is not None:
            # true cost is either the cost function, or pre-computed costs as an array
            true_ys = true_cost(all_xs) if callable(true_cost) else true_cost
            ax1.plot(all_xs, true_ys, '--', color='#2f2f2f', label='true cost', linewidth=1.0)

        ### Plot Samples
        # plot samples projected onto the `param` axis
        ax1.plot(concrete_xs, concrete_ys, 'bo', markersize=6, label='samples', zorder=5)

        if len(hypothesised_xs) > 0:
            # there are some hypothesised samples
            ax1.plot(hypothesised_xs, hypothesised_ys, 'o', color='tomato',
                     markersize=6, label='hypothesised samples', zorder=5)

        ax1.plot(best_concrete_x, best_concrete_y, '*', markersize=15,
                 color='deepskyblue', zorder=10, label='best sample')

        ### Plot Surrogate Function
        def plot_gp_prediction_through(point, mu_label, sigma_label, mu_alpha, sigma_alpha):
            # points with all but the chosen parameter fixed to match the given
            # config, but the chosen parameter varies
            perturbed = self._points_vary_one(point, param, gp_xs)
            mus, sigmas = gp_model.predict(perturbed, return_std=True)
            mus = mus.flatten()
            ax1.plot(all_xs, mus, 'm-', label=mu_label, alpha=mu_alpha, linewidth=1.0)
            ax1.fill_between(all_xs, mus - n_sigma*sigmas, mus + n_sigma*sigmas, alpha=sigma_alpha,
                            color='mediumpurple', label=sigma_label)

        #TODO: fit the view to the cost function, don't expand to fit in the uncertainty

        if sims:
            for mus in sim_mus:
                ax1.plot(all_xs, mus, 'm:', linewidth=1.0)

        plot_gp_prediction_through(chosen_point,
            mu_label='surrogate cost', sigma_label='uncertainty ${}\\sigma$'.format(n_sigma),
            mu_alpha=1, sigma_alpha=0.25)

        # plot the predictions through each sample
        def predictions_through_all_samples():
            # avoid drawing predictions of the same place more than once, so
            # avoid duplicate configurations which are identical to another
            # except for the value of 'param', since the plot varies this
            # parameter: the resulting plot will be the same in both cases.
            param_index = self.point_space.param_indices[param]
            # a copy of the current samples with the focused parameter zeroed
            # start with s.next_x since that is a point which is guaranteed to
            # have a prediction plotted through it
            param_zeroed = np.vstack((chosen_point, xs))
            param_zeroed[0,param_index] = 0
            param_zeroed = unique_rows_close(param_zeroed, close_tolerance=1e-3)
            param_zeroed = param_zeroed[1:,:] # exclude chosen_point

            if param_zeroed.shape[0] > 0:
                # cap to make sure they don't become invisible
                alpha = max(0.4/param_zeroed.shape[0], 0.015)
                for row in param_zeroed:
                    plot_gp_prediction_through(make2D_row(row),
                        mu_label=None, sigma_label=None,
                        mu_alpha=alpha, sigma_alpha=alpha)

        if gp_through_all:
            predictions_through_all_samples()


        ### Plot Vertical Bars
        bar_width = 1
        ax1.axvline(x=chosen_x, linewidth=bar_width)
        if random_fallback:
            # in this case: chosen_x is the random choice
            ax1.axvline(x=acq_chosen_x, color='y', linewidth=bar_width)

        ax1.legend()

        ### Plot Acquisition Function
        ax2.set_xlabel('parameter {}'.format(param))
        ax2.set_ylabel(self.strategy.get_name(self.maximise_cost))
        ax2.set_title('acquisition function')
        # can be useful for observing the gradients of acquisition functions
        # with very thin spikes.
        #ax2.set_yscale('log')

        if sims:
            for s_ac in sim_ac:
                ax2.plot(all_xs, s_ac, ':', color='g', linewidth=1.0)

        ax2.plot(all_xs, ac, '-', color='g', linewidth=1.0, label='acquisition function')
        ax2.fill_between(all_xs, np.zeros_like(all_xs), ac.flatten(), alpha=0.3, color='palegreen')

        if random_fallback:
            ax2.axvline(x=random_chosen_x, label='next sample', linewidth=bar_width)
            label='$\\mathrm{{argmax}}\\; {}$'.format(
                self.strategy.get_name(self.maximise_cost))
            ax2.axvline(x=acq_chosen_x, color='y', label=label, linewidth=bar_width)
        else:
            ax2.axvline(x=acq_chosen_x, linewidth=bar_width)
            ax2.plot(acq_chosen_x, acq_chosen_ac, 'b^', markersize=7,
                     zorder=10, label='next sample')

        ax2.legend()
        return fig



    def plot_step_2D(self, x_param, y_param, step, true_cost=None,
                     plot_through='next', force_view_linear=False):
        '''
        x_param: the name of the parameter to place along the x axis
        y_param: the name of the parameter to place along the y axis
        step: the number of the step to plot
        true_cost: a function (that takes x and y arguments) or meshgrid
            containing the true cost values
        plot_through: unlike the 1D step plotting, in 2D the heatmaps cannot be
            easily overlaid to get a better understanding of the whole space.
            Instead, a configuration can be provided to vary x_pram and y_param
            but leave the others constant to produce the graphs.
            Pass 'next' to signify the next sample (the one chosen by the
            current step) or 'best' to signify the current best sample as-of the
            current step. a configuration dict can also be passed
        force_view_linear: force the images to be displayed with linear axes
            even if the parameters are logarithmic
        '''
        assert step in self.step_log, 'step not recorded in the log'

        x_range = self.point_space.ranges[x_param]
        y_range = self.point_space.ranges[y_param]

        assert all(type_ in (RangeType.Linear, RangeType.Logarithmic)
                   for type_ in (x_range.type_, y_range.type_))
        x_is_log = x_range.type_ == RangeType.Logarithmic
        y_is_log = y_range.type_ == RangeType.Logarithmic

        s = self.step_log[step]
        num_suggestions = len(s.suggestions)
        all_xs, all_ys = x_range.values, y_range.values

        # cases for different types of steps. Only certain cases are accounted
        # for, because only these cases should happen under normal circumstances.
        if not (num_suggestions == 1 or num_suggestions == 2):
            raise ValueError(num_suggestions)

        # extract and process the data from the first suggestion
        sg1 = s.suggestions[0]
        if isinstance(sg1, Step.MaxAcqSuggestion):
            # combination of the concrete samples (sx, sy) and the hypothesised samples
            # (hx, hy) if there are any
            xs = np.vstack((s.sx, s.hx))
            ys = np.vstack((s.sy, sg1.hy))

            # restore from just the hyperparameters to a GP which can be queried
            gp_model = restore_GP(sg1.gp, self.gp_params, xs, ys)

            acq_fun = self._get_acq_fun(gp_model, ys)

            # the GP is trained in point space
            gp_xs = self.point_space.param_to_point_space(all_xs, x_param)
            gp_ys = self.point_space.param_to_point_space(all_ys, y_param)
            gp_X, gp_Y = np.meshgrid(gp_xs, gp_ys)
            grid_size = (len(all_xs), len(all_ys)) # shape of gp_X and gp_Y
            assert grid_size == gp_X.shape #TODO: remove if accurate
            # all combinations of x and y values, each point as a row
            #TODO: would hstack work instead of transposing?
            gp_points = np.vstack((gp_X.ravel(), gp_Y.ravel())).T # ravel squashes to 1D

            # the values for the chosen parameter for the concrete and
            # hypothesised samples in config space.
            concrete_xs = self.point_space.param_to_config_space(s.sx, x_param).flatten()
            concrete_ys = self.point_space.param_to_config_space(s.sx, y_param).flatten()
            concrete_zs = s.sy

            hypothesised_xs = self.point_space.param_to_config_space(s.hx, x_param).flatten()
            hypothesised_ys = self.point_space.param_to_config_space(s.hx, y_param).flatten()
            hypothesised_zs = sg1.hy

            # the current best concrete sample
            best_i = np.argmax(concrete_ys) if self.maximise_cost else np.argmin(concrete_ys)
            best_concrete_x = concrete_xs[best_i]
            best_concrete_y = concrete_ys[best_i]
            best_concrete_z = concrete_zs[best_i]

            # next point chosen by the acquisition function to be evaluated
            acq_chosen_point = sg1.x
            acq_chosen_x = self.point_space.param_to_config_space(acq_chosen_point, x_param)
            acq_chosen_y = self.point_space.param_to_config_space(acq_chosen_point, y_param)
            acq_chosen_ac = sg1.ac

            chosen_point = acq_chosen_point
            chosen_x = acq_chosen_x
            chosen_y = acq_chosen_y

        # extract and process the data from the second suggestion if there is one
        if num_suggestions == 2:
            sg2 = s.suggestions[1]
            if isinstance(sg2, Step.RandomSuggestion):
                random_fallback = True

                random_chosen_point = sg2.x
                random_chosen_x = self.point_space.param_to_config_space(random_chosen_point, x_param)
                random_chosen_y = self.point_space.param_to_config_space(random_chosen_point, y_param)

                chosen_point = random_chosen_point
                chosen_x = random_chosen_x
                chosen_y = random_chosen_y

            else:
                raise NotImplementedError()
        else:
            random_fallback = False


        # determine the point to plot through
        if isinstance(plot_through, dict):
            plot_through = self.config_to_point(plot_through)
        else:
            plot_through = plot_through.lower()
            if plot_through == 'next':
                plot_through = chosen_point
            elif plot_through == 'best':
                best_i = np.argmax(concrete_ys) if self.maximise_cost else np.argmin(concrete_ys)
                plot_through = make2D_row(s.sx[best_i])
            else:
                raise ValueError(plot_through)

        # points with all but the chosen parameter fixed to match the given
        # config, but the focused parameters vary
        gp_perturbed_points = self._points_vary_one(plot_through, x_param, gp_points[:,0])
        y_index = self.point_space.param_indices[y_param]
        gp_perturbed_points[:,y_index] = gp_points[:,1]

        if acq_fun:
            mus, sigmas = gp_model.predict(gp_perturbed_points, return_std=True)
            mus = mus.reshape(*grid_size)
            sigmas = sigmas.reshape(*grid_size)
            ac = acq_fun(gp_perturbed_points)
            ac = ac.reshape(*grid_size)


        fig = plt.figure(figsize=(16, 16)) # inches
        grid = gridspec.GridSpec(nrows=2, ncols=2)
        # layout:
        # ax1 ax2
        # ax3 ax4
        ax1      = fig.add_subplot(grid[0])
        ax3, ax4 = fig.add_subplot(grid[2]), fig.add_subplot(grid[3])
        ax2 = fig.add_subplot(grid[1]) if true_cost is not None else None
        axes = (ax1, ax2, ax3, ax4) if true_cost is not None else (ax1, ax3, ax4)

        for ax in axes:
            ax.set_xlim(x_range.bounds)
            if x_is_log and not force_view_linear:
                ax.set_xscale('log')
            ax.set_ylim(y_range.bounds)
            if y_is_log and not force_view_linear:
                ax.set_yscale('log')
            ax.grid(False)

        #TODO: decrease h_pad
        # need to specify rect so that the suptitle isn't cut off
        fig.tight_layout(h_pad=4, w_pad=8, rect=[0, 0, 1, 0.96]) # [left, bottom, right, top] 0-1

        title = 'Bayesian Optimisation step {}'.format(step-self.strategy.pre_phase_steps)
        if random_fallback:
            title += ' (Random Fallback)'

        fig.suptitle(title, fontsize=20)

        def plot_heatmap(ax, data, colorbar, cmap):
            # pcolormesh is better than imshow because: no need to fiddle around
            # with extents and aspect ratios because the x and y values can be
            # fed in and so just works. This also prevents the problem of the
            # origin being in the wrong place. It is compatible with log scaled
            # axes unlike imshow. There is no interpolation by default unlike
            # imshow.
            im = ax.pcolormesh(all_xs, all_ys, data, cmap=cmap)
            if colorbar:
                c = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.051)
                c.set_label('cost')
            ax.set_xlabel('parameter {}'.format(x_param))
            ax.set_ylabel('parameter {}'.format(y_param))
            ax.plot(best_concrete_x, best_concrete_y, '*', markersize=15,
                    color='deepskyblue', zorder=10, linestyle='None',
                    label='best sample')
            ax.plot(concrete_xs, concrete_ys, 'ro', markersize=4,
                    linestyle='None', label='samples')
            if len(hypothesised_xs) > 0:
                ax1.plot(hypothesised_xs, hypothesised_ys, 'o', color='tomato',
                         linestyle='None', label='hypothesised samples')

            ax.plot(chosen_x, chosen_y, marker='d', color='orangered',
                    markeredgecolor='black', markeredgewidth=1.5, markersize=10,
                    linestyle='None', label='next sample')

        title_size = 16
        cmap = 'viridis'
        # reverse the color map if minimising
        cmap_match_direction = cmap if self.maximise_cost else cmap + '_r' # reversed

        ax1.set_title('Surrogate $\\mu$', fontsize=title_size)
        im = plot_heatmap(ax1, mus, colorbar=True, cmap=cmap_match_direction)

        ax3.set_title('Surrogate $\\sigma$', fontsize=title_size)
        plot_heatmap(ax3, sigmas, colorbar=True, cmap=cmap)

        if true_cost is not None:
            ax2.set_title('True Cost', fontsize=title_size)
            plot_heatmap(ax2, true_cost, colorbar=True, cmap=cmap_match_direction)

        ax4.set_title('Acquisition Function', fontsize=title_size)
        plot_heatmap(ax4, ac, colorbar=True, cmap=cmap)

        if random_fallback:
            label='$\\mathrm{{argmax}}\\; {}$'.format(
                self.strategy.get_name(self.maximise_cost))
            ax4.axvline(x=acq_chosen_x, color='y')
            ax4.axhline(y=acq_chosen_y, color='y', label=label)

        ax4.legend(bbox_to_anchor=(0, 1.01), loc='lower left', borderaxespad=0.0)

        return fig

    def num_randomly_chosen(self):
        count = 0
        for s in self.samples:
            is_pre_sample = s.job_ID <= self.strategy.pre_phase_steps
            is_random = (s.job_ID in self.step_log and
                isinstance(self.step_log[s.job_ID].suggestions[-1], Step.RandomSuggestion))
            if is_pre_sample or is_random:
                count += 1
        return count

    def plot_cost_over_time(self, plot_each=True, plot_best=True,
                            true_best=None, plot_random=True):
        '''
        plot a line graph showing the progress that the optimiser makes towards
        the optimum as the number of samples increases.
        plot_each: plot the cost of each sample
        plot_best: plot the running-best cost
        true_best: if available: plot a horizontal line for the best possible cost
        plot_random: whether to plot markers over the samples which were chosen randomly
        '''
        fig = OptimiserPlotting.plot_cost_over_time(self, plot_each, plot_best, true_best)
        ax = fig.axes[0]

        if plot_random:
            random_sample_nums = []
            random_sample_costs = []
            for i, s in enumerate(self.samples):
                is_pre_sample = s.job_ID <= self.strategy.pre_phase_steps
                is_random = s.job_ID in self.step_log and self.step_log[s.job_ID].chosen_at_random()
                if is_pre_sample or is_random:
                    random_sample_nums.append(i+1)
                    random_sample_costs.append(s.cost)
            ax.plot(random_sample_nums, random_sample_costs, 'ro', markersize=5, label='randomly chosen')
            ax.margins(0.0, 0.18)
            ax.legend()

        def sample_num_to_bayes_step(s_num):
            i = int(s_num)-1
            if i >= 0 and i < len(self.samples):
                s = self.samples[i]
                if s.job_ID in self.step_log:
                    return s.job_ID - self.strategy.pre_phase_steps
                else:
                    return ''
            else:
                return ''
        labels = [sample_num_to_bayes_step(s_num) for s_num in ax.get_xticks()]

        ax2 = ax.twiny() # 'twin y'
        ax2.grid(False)
        ax2.set_xlim(ax.get_xlim())
        ax2.xaxis.set_major_locator(ax.xaxis.get_major_locator())
        # convert the labels marked on ax into new labels for the top
        ax2.set_xticklabels(labels)
        ax2.set_xlabel('Bayesian Step')

        # raise the title to get out of the way of ax2
        ax.title.set_position([0.5, 1.08])

        return fig


