#!/usr/bin/env python3
'''
Plotting methods for Optimiser objects (extracted from the main class
definitions because these methods are long and just add noise)
'''
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.transforms

from collections import defaultdict
from itertools import groupby

# local modules
from .utils import *
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

    def _points_vary_one(self, config, param, xs):
        '''
        points generated by fixing all but one parameter to match the given
        config, and varying the remaining parameter.
        config: the configuration to base the points on
        param: the name of the parameter to vary
        xs: the values to provide in place of config[param]
        '''
        assert len(xs.shape) == 1 # not 2D
        # create many duplicates of the point version of the given configuration
        points = np.repeat(self.config_to_point(config), len(xs), axis=0)
        param_index = self._index_for_param(param) # column to swap out for xs
        points[:,param_index] = xs
        return points

    #TODO: rename to plot1D
    def plot_step_slice(self, param, step, true_cost=None, log_ac=False,
                        n_sigma=2, gp_through_all=True):
        '''
        plot a Bayesian optimisation step, perturbed along a single parameter.

        the 1D case is trivial: the plot is simply the parameter value and the
        corresponding cost and acquisition values.
        in 2D, imagine the surface plot of the two parameters against cost (as
        the height). This plot takes a cross section of that surface along the
        specified axis and passing through the point of the next configuration
        to test to show how the acquisition function varies along that dimension.
        The same holds for higher dimensions but is harder to visualise.

        param: the name of the parameter to perturb to obtain the graph
        bayes_step: the job ID to plot (must be in self.step_log)
        true_cost: true cost function (or array of pre-computed cost values corresponding to self.ranges[param]) (None to omit)
        log_ac: whether to display the negative log acquisition function instead
        n_sigma: the number of standard deviations from the mean to plot the
            uncertainty confidence interval.
            Note 1=>68%, 2=>95%, 3=>99% (for a normal distribution, which this is)
        gp_through_all: whether to plot a gp prediction through every sample or
            just through the location of the next point to be chosen
        '''
        assert step in self.step_log.keys(), 'step not recorded in the log'

        type_ = self.range_types[param]
        assert type_ in [RangeType.Linear, RangeType.Logarithmic]
        # whether the range of the focused parameter is logarithmic
        is_log = type_ == RangeType.Logarithmic

        s = self.step_log[step]
        all_xs = self.ranges[param]

        # combination of the true samples (sx, sy) and the hypothesised samples
        # (hx, hy) if there are any
        xs = np.vstack([s.sx, s.hx])
        ys = np.vstack([s.sy, s.hy])

        # training the GP is nondeterministic if there are any parameters to
        # tune so may give a different result here to during optimisation
        # gp_model = gp.GaussianProcessRegressor(**self.gp_params)
        # gp_model.fit(xs, ys)

        fig = plt.figure(figsize=(16, 10)) # inches
        grid = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2, 1])
        ax1, ax2 = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])

        fig.suptitle('Bayesian Optimisation step {}{}'.format(
            step-self.pre_samples,
            (' (chosen at random)' if s.chosen_at_random else '')), fontsize=14)
        ax1.margins(0.01, 0.1)
        ax2.margins(0.01, 0.1)
        if is_log:
            ax1.set_xscale('log')
            ax2.set_xscale('log')
        fig.subplots_adjust(hspace=0.3)

        ax1.set_ylabel('cost')
        ax1.set_title('Surrogate objective function')

        ### Plot True Cost
        if true_cost is not None:
            # true cost is either the cost function, or pre-computed costs as an array
            ys = true_cost(all_xs) if callable(true_cost) else true_cost
            ax1.plot(all_xs, ys, 'k--', label='true cost')

        ### Plot Samples
        param_index = self._index_for_param(param)
        # if logarithmic: value stored as exponent
        get_param = lambda p: np.exp(p[param_index]) if is_log else p[param_index]
        # plot samples projected onto the `param` axis
        sample_xs = [get_param(x) for x in s.sx]
        ax1.plot(sample_xs, s.sy, 'bo', label='samples')

        if len(s.hx) > 0:
            # there are some hypothesised samples
            hypothesised_xs = [get_param(x) for x in s.hx]
            ax1.plot(hypothesised_xs, s.hy, 'o', color='tomato', label='hypothesised samples')

        # index of the best current real sample
        best_i = np.argmax(s.sy) if self.maximise_cost else np.argmin(s.sy)
        ax1.plot(sample_xs[best_i], s.sy[best_i], '*', markersize=15,
                 color='deepskyblue', zorder=10, label='best sample')


        ### Plot Surrogate Function
        def plot_gp_prediction_through(config, mu_label, sigma_label, mu_alpha,
                                       sigma_alpha):
            # if logarithmic: the GP is trained on the exponents of the values
            gp_xs = np.log(all_xs) if is_log else all_xs
            # points with all but the chosen parameter fixed to match the given
            # config, but the chosen parameter varies
            perturbed = self._points_vary_one(config, param, gp_xs)
            mus, sigmas = s.gp.predict(perturbed, return_std=True)
            mus = mus.flatten()
            ax1.plot(all_xs, mus, 'm-', label=mu_label, alpha=mu_alpha)
            ax1.fill_between(all_xs, mus - n_sigma*sigmas, mus + n_sigma*sigmas, alpha=sigma_alpha,
                            color='mediumpurple', label=sigma_label)

        #TODO: fit the view to the cost function, don't expand to fit in the uncertainty

        plot_gp_prediction_through(s.next_x,
            mu_label='surrogate cost', sigma_label='uncertainty ${}\\sigma$'.format(n_sigma),
            mu_alpha=1, sigma_alpha=0.3)

        # plot the predictions through each sample
        def predictions_through_all_samples():
            configs_to_use = []

            # avoid drawing predictions of the same place more than once, so
            # avoid duplicate configurations which are identical to another
            # except for the value of 'param', since the plot varies this
            # parameter: the resulting plot will be the same in both cases.
            param_index = self._index_for_param(param)
            # a copy of the current samples with the focused parameter zeroed
            # start with s.next_x since that is a point which is guaranteed to
            # have a prediction plotted through it
            param_zeroed = self.config_to_point(s.next_x)
            param_zeroed[0,param_index] = 0
            for x in xs:
                x_zeroed = make2D_row(np.array(x))
                x_zeroed[0,param_index] = 0
                if not close_to_any(x_zeroed, param_zeroed, tol=1e-3):
                    configs_to_use.append(self.point_to_config(make2D_row(x)))
                    param_zeroed = np.append(param_zeroed, x_zeroed, axis=0)

            if len(configs_to_use) > 0:
                # cap to make sure they don't become invisible
                alpha = max(0.4/len(configs_to_use), 0.015)
                for cfg in configs_to_use:
                    plot_gp_prediction_through(cfg,
                        mu_label=None, sigma_label=None,
                        mu_alpha=alpha, sigma_alpha=alpha)

        if gp_through_all:
            predictions_through_all_samples()


        ### Plot Vertical Bars
        ax1.axvline(x=s.next_x[param])

        if s.chosen_at_random and s.argmax_acquisition is not None:
            ax1.axvline(x=get_param(s.argmax_acquisition[0]), color='y')

        ax1.legend()

        ax2.set_xlabel('parameter {}'.format(param))
        ax2.set_ylabel(self.acquisition_function_name)
        ax2.set_title('acquisition function')

        ### Plot Acquisition Function
        # it makes sense to plot the acquisition function through the slice
        # corresponding to the next point to be chosen.
        # if logarithmic: the GP is trained on the exponents of the values
        gp_xs = np.log(all_xs) if is_log else all_xs
        perturbed = self._points_vary_one(s.next_x, param, gp_xs)
        ac = self.acquisition_function(perturbed, s.gp, self.maximise_cost,
                                       s.best_sample.cost,
                                       **self.acquisition_function_params)
        #TODO: remove log_ac option
        if log_ac:
            # only useful for EI where ac >= 0 always
            ac[ac == 0.0] = 1e-10
            ac = -np.log(ac)
            label = '-log(acquisition function)'
        else:
            label = 'acquisition function'

        # show close-up on the next sample
        #ax2.set_xlim(s.next_x[param]-1, s.next_x[param]+1)
        #ax2.set_ylim((0.0, s.next_ac))

        ax2.plot(all_xs, ac, '-', color='g', linewidth=1.0, label=label)
        ax2.fill_between(all_xs, np.zeros_like(all_xs), ac.flatten(), alpha=0.3, color='palegreen')

        ax2.axvline(x=s.next_x[param])
        # may not want to plot if chosen_at_random because next_ac will be incorrect (ie 0)
        ax2.plot(s.next_x[param], s.next_ac, 'b^', markersize=10, alpha=0.8, label='next sample')

        # when chosen at random, next_x is different from what the maximisation
        # of the acquisition function suggested as the next configuration to
        # test. So plot both.
        if s.chosen_at_random and s.argmax_acquisition is not None:
            ac_x = get_param(s.argmax_acquisition[0])
            label='$\\mathrm{{argmax}}\\; {}$'.format(self.acquisition_function_name)
            ax2.axvline(x=ac_x, color='y', label=label)

        ax2.legend()

        return fig

    def plot_step_2D(self, x_param, y_param, step, true_cost=None,
                     plot_through='next', force_view_linear=False):
        '''
        x_param: the name of the parameter to place along the x axis
        y_param: the name of the parameter to place along the y axis
        step: the step number to plot
        true_cost: a function (that takes x and y arguments) or meshgrid
            containing the true cost values
        plot_through: unlike the 1D step plotting, in 2D the heatmaps cannot be
            easily overlaid to get a better understanding of the whole space.
            Instead, a Sample object can be provided to vary x_pram and y_param
            but leave the others constant to produce the graphs.
            Pass 'next' to signify the next sample (the one chosen by the
            current step) or 'best' to signify the current best sample as-of the
            current step.
        force_view_linear: force the images to be displayed with linear axes
            even if the parameters are logarithmic
        '''
        assert step in self.step_log.keys(), 'step not recorded in the log'

        x_type, y_type = self.range_types[x_param], self.range_types[y_param]
        assert all(type_ in [RangeType.Linear, RangeType.Logarithmic]
                   for type_ in [x_type, y_type])
        x_is_log = x_type == RangeType.Logarithmic
        y_is_log = y_type == RangeType.Logarithmic

        s = self.step_log[step]

        if plot_through == 'next':
            plot_through = s.next_x
        elif plot_through == 'best':
            plot_through = self.config_to_point(s.best_sample.config)
        elif isinstance(plot_through, Sample):
            plot_through = self.config_to_point(plot_through.config)
        else:
            raise ValueError(plot_through)


        all_xs, all_ys = self.ranges[x_param], self.ranges[y_param]
        # the GP is trained on the exponents of the values if the parameter is logarithmic
        gp_xs = np.log(all_xs) if x_is_log else all_xs
        gp_ys = np.log(all_ys) if y_is_log else all_ys
        gp_X, gp_Y = np.meshgrid(gp_xs, gp_ys)
        # all combinations of x and y values, each point as a row
        gp_all_combos = np.vstack([gp_X.ravel(), gp_Y.ravel()]).T # ravel squashes to 1D

        grid_size = gp_X.shape # both X and Y have shape=(len xs, len ys)

        # combination of the true samples (sx, sy) and the hypothesised samples
        # (hx, hy) if there are any
        xs = np.vstack([s.sx, s.hx])
        ys = np.vstack([s.sy, s.hy])

        fig = plt.figure(figsize=(16, 16)) # inches
        grid = gridspec.GridSpec(nrows=2, ncols=2)
        # layout:
        # ax1 ax2
        # ax3 ax4
        ax1      = fig.add_subplot(grid[0])
        ax3, ax4 = fig.add_subplot(grid[2]), fig.add_subplot(grid[3])
        ax2 = fig.add_subplot(grid[1]) if true_cost is not None else None
        axes = [ax1, ax2, ax3, ax4] if true_cost is not None else [ax1, ax3, ax4]

        for ax in axes:
            ax.set_xlim(self.range_bounds[x_param])
            if x_is_log and not force_view_linear:
                ax.set_xscale('log')
            ax.set_ylim(self.range_bounds[y_param])
            if y_is_log and not force_view_linear:
                ax.set_yscale('log')
            ax.grid(False)

        #TODO: decrease h_pad
        # need to specify rect so that the suptitle isn't cut off
        fig.tight_layout(h_pad=4, w_pad=8, rect=[0, 0, 1, 0.96]) # [left, bottom, right, top] 0-1

        fig.suptitle('Bayesian Optimisation step {}{}'.format(
            step-self.pre_samples,
            (' (chosen at random)' if s.chosen_at_random else '')), fontsize=20)

        config = s.next_x # plot the GP through this config
        # points with all but the chosen parameter fixed to match the given
        # config, but the focused parameters vary
        perturbed = self._points_vary_one(config, x_param, gp_all_combos[:,0])
        y_index = self._index_for_param(y_param)
        perturbed[:,y_index] = gp_all_combos[:,1]

        mus, sigmas = s.gp.predict(perturbed, return_std=True)
        mus = mus.flatten()
        ac = self.acquisition_function(perturbed, s.gp, self.maximise_cost,
                                       s.best_sample.cost,
                                       **self.acquisition_function_params)

        x_param_index = self._index_for_param(x_param)
        y_param_index = self._index_for_param(y_param)
        # if logarithmic: value stored as exponent
        get_x_param = lambda p: np.exp(p[x_param_index]) if x_is_log else p[x_param_index]
        get_y_param = lambda p: np.exp(p[y_param_index]) if y_is_log else p[y_param_index]

        # plot samples projected onto the `x_param` and `y_param' axes
        sample_xs = [get_x_param(x) for x in s.sx]
        sample_ys = [get_y_param(x) for x in s.sx]

        if len(s.hx) > 0:
            # there are some hypothesised samples
            hypothesised_xs = [get_x_param(x) for x in s.hx]
            hypothesised_ys = [get_y_param(x) for x in s.hx]

        next_x, next_y = s.next_x[x_param], s.next_x[y_param]

        best_i = np.argmax(s.sy) if self.maximise_cost else np.argmin(s.sy)
        best_x, best_y = sample_xs[best_i], sample_ys[best_i]

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
            ax.plot(best_x, best_y, '*', markersize=15,
                 color='deepskyblue', zorder=10, linestyle='None', label='best sample')
            ax.plot(sample_xs, sample_ys, 'ro', markersize=4, linestyle='None', label='samples')
            if len(s.hx) > 0:
                ax1.plot(hypothesised_xs, hypothesised_ys, 'o', color='tomato',
                         linestyle='None', label='hypothesised samples')

            ax.plot(next_x, next_y, marker='d', color='orangered',
                    markeredgecolor='black', markeredgewidth=1.5, markersize=10,
                    linestyle='None', label='next sample')

        title_size = 16
        cmap = 'viridis'
        # reverse the color map if minimising
        cmap_match_direction = cmap if self.maximise_cost else cmap + '_r' # reversed

        ax1.set_title('Surrogate $\\mu$', fontsize=title_size)
        mus = mus.reshape(*grid_size)
        im = plot_heatmap(ax1, mus, colorbar=True, cmap=cmap_match_direction)

        ax3.set_title('Surrogate $\\sigma$', fontsize=title_size)
        sigmas = sigmas.reshape(*grid_size)
        plot_heatmap(ax3, sigmas, colorbar=True, cmap=cmap)

        if true_cost is not None:
            ax2.set_title('True Cost', fontsize=title_size)
            plot_heatmap(ax2, true_cost, colorbar=True, cmap=cmap_match_direction)

        ax4.set_title('Acquisition Function', fontsize=title_size)
        ac = ac.reshape(*grid_size)
        plot_heatmap(ax4, ac, colorbar=True, cmap=cmap)

        if s.chosen_at_random and s.argmax_acquisition is not None:
            label='$\\mathrm{{argmax}}\\; {}$'.format(self.acquisition_function_name)
            ax4.axhline(y=get_y_param(s.argmax_acquisition[0]), color='y', label=label)
            ax4.axvline(x=get_x_param(s.argmax_acquisition[0]), color='y')

        ax4.legend(bbox_to_anchor=(0, 1.01), loc='lower left', borderaxespad=0.0)

        return fig

    def num_randomly_chosen(self):
        count = 0
        for s in self.samples:
            is_pre_sample = s.job_ID <= self.pre_samples
            is_random = s.job_ID in self.step_log.keys() and self.step_log[s.job_ID].chosen_at_random
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
                is_pre_sample = s.job_ID <= self.pre_samples
                is_random = s.job_ID in self.step_log.keys() and self.step_log[s.job_ID].chosen_at_random
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
                if s.job_ID in self.step_log.keys():
                    return s.job_ID - self.pre_samples
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


