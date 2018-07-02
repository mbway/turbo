#!/usr/bin/env python3
""" The Bayesian Optimisation specific code """

import numpy as np
import json

# local imports
from .bounds import Bounds
from .optimiser_presets import load_optimiser_preset


class Optimiser:
    def __init__(self, objective, desired_extremum, bounds, pre_phase_trials, settings_preset='default'):
        """
        Args:
            objective: a function to be optimised, which accepts parameters
                corresponding to the given bounds and returns an objective/cost
                value for the input.
                The objective function may either return a float for the cost,
                or a tuple of (cost, eval_info). The contents of eval_info is up
                to the user. The optimiser will only pass it on to its
                listeners.
            desired_extremum: either 'min' or 'max' specifying whether the
                objective function is to be maximised or minimised
            bounds: a list of tuples of (name, min, max) for each parameter of
                the objective function
            settings_preset: the name of the preset optimiser settings to load with
                `load_optimiser_preset()`. Pass None to leave the optimiser
                uninitialised for full customisation.
        """
        self.objective = objective
        #TODO: internally, should use is_maximising or is_minimising where possible
        assert desired_extremum in ('min', 'max'), 'desired_extremum must be either "min" or "max"'
        self.desired_extremum = desired_extremum
        self.bounds = Bounds(bounds)
        assert pre_phase_trials > 0, 'a pre-phase is required'
        self.pre_phase_trials = pre_phase_trials

        # modules
        self.latent_space = None
        self.pre_phase_select = None
        self.fallback = None
        self.aux_optimiser = None  # auxiliary optimiser to maximise the acquisition function
        #self.async_eval = None#TODO
        #self.parallel_strategy = None#TODO make sub-module of async-eval
        self.surrogate = None  # factory for creating surrogate models
        self.acquisition = None  # factory for creating acquisition functions

        # runtime data kept separate from configuration data
        self.rt = Optimiser.Runtime()
        # shouldn't be accessed directly, but through the `register_listener()`
        # and `unregister_listener()` methods
        self._listeners = []

        self.initialised = True  # no more attributes can be set

        if settings_preset is not None:
            load_optimiser_preset(self, settings_preset)

    class Runtime:
        """ holds the data for an optimisation run

        Attributes:
            max_trials: the maximum number of trials in total (not just this
                run) before stopping
            trial_xs: input points (in latent space) for the finished trials
            trial_ys: cost values for the finished trials
        """
        def __init__(self):
            self.running = False
            self.started_trials = 0
            self.finished_trials = 0

            # stopping conditions
            self.max_trials = 0
            #TODO: max_time (maximum wallclock time before stopping)
            #TODO: min_improvement (over last N trials if incumbent hasn't improved by at least min_improvement then stop)

            #TODO (naming): these should be finished_xs and finished_ys
            self.trial_xs = []  # list of row vectors
            self.trial_ys = []  # list of scalars

        def check_consistency(self):
            """ check that the optimiser runtime data makes sense

            Raises:
                AssertionError
            """
            assert self.started_trials >= self.finished_trials
            assert len(self.trial_xs) == len(self.trial_ys)
            assert len(self.trial_ys) == self.finished_trials

        def add_finished_trial(self, x, y):
            self.trial_xs.append(x)
            self.trial_ys.append(y)
            self.finished_trials += 1

    def __setattr__(self, name, value):
        """ limit the ability to set optimiser attributes. New attributes can
        only be created from the constructor.

        This can help prevent typos caused by getting the name of a module wrong
        (or catching when a name changes due to an API update), which would
        otherwise fail silently, especially if default settings are loaded.
        """
        initialised = hasattr(self, 'initialised') and self.initialised
        if hasattr(self, name) or not initialised:
            object.__setattr__(self, name, value)
        else:
            raise AttributeError('Optimiser does not have an "{}" attribute!'.format(name))

    def is_maximising(self):
        return self.desired_extremum == 'max'

    def is_minimising(self):
        return self.desired_extremum == 'min'

    def register_listener(self, listener):
        assert listener not in self._listeners
        self._listeners.append(listener)
        listener.registered(self)

    def unregister_listener(self, listener):
        assert listener in self._listeners
        self._listeners = [l for l in self._listeners if l != listener]
        listener.unregistered()

    def save_JSON(self, filename, include_runtime=True):
        data = self.save(include_runtime)
        with open(filename, 'w') as f:
            json.dump(data, f, indent='    ', sort_keys=True)

    def load_JSON(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            self.load(data)

    def save(self, include_runtime=True):
        """ Save the configuration of the optimiser to a dictionary

        Args:
            include_runtime: Whether to also include the current runtime state of the optimiser

        Note:
            the objective function and attached listeners are not saved
        """
        assert self.initialised
        data = {
            'desired_extremum': self.desired_extremum,
            'bounds': self.bounds.ordered,
            'pre_phase_trials': self.pre_phase_trials,
            'latent_space': self.latent_space.save(include_runtime),
            'pre_phase_select': self.pre_phase_select.save(include_runtime),
            'fallback': self.fallback.save(include_runtime),
            'aux_optimiser': self.aux_optimiser.save(include_runtime),
            'surrogate': self.surrogate.save(include_runtime),
            'acquisition': self.acquisition.save(include_runtime)
        }
        if include_runtime and self.rt is not None:
            data['runtime'] = {
                'running': self.rt.running,
                'started_trials': self.rt.started_trials,
                'finished_trials': self.rt.finished_trials,
                'max_trials': self.rt.max_trials,
                'trial_xs': self.rt.trial_xs[:],  # TODO: may need to make a copy of each elements
                'trial_ys': self.rt.trial_ys[:]
            }
        return data

    def load(self, data):
        pass  #TODO:

    def load_trials(self, trials):
        """
        Args:
            trials (Recorder.Trial): a list of trials to load
        """
        assert all(trial.x.shape == (1, len(self.bounds)) for trial in trials)
        assert not self.rt.running

        self.latent_space._set_input_bounds(self.bounds)
        self._check_settings()
        rt = self.rt # runtime data
        rt.running = True
        rt.max_trials = rt.finished_trials + len(trials)

        self._notify('run_started', rt.finished_trials, rt.max_trials)
        trial_num = self.rt.finished_trials
        for trial in trials:
            rt.started_trials += 1
            self._notify('selection_started', trial_num)
            self._notify('selection_finished', trial_num, trial.x, trial.selection_info)
            self._notify('evaluation_started', trial_num)
            rt.add_finished_trial(trial.x, trial.y)
            self._notify('evaluation_finished', trial_num, trial.y, trial.eval_info)
            trial_num += 1
        rt.running = False
        self._notify('run_finished')

    def get_incumbent(self, as_dict=False):
        """ get the current best trial

        Args:
            as_dict (bool): True => return the trial input as a dict in the input space
                            False => return the trial as a point in the latent space

        Returns:
            (i, x, y)
            i = trial number
            x = the trial input
            y = the trial objective function value
        """
        rt = self.rt
        i = int(np.argmax(rt.trial_ys)) if self.is_maximising() else int(np.argmin(rt.trial_ys))
        x = rt.trial_xs[i]
        if as_dict:
            x = self._point_to_dict(self.latent_space.from_latent(x))
        return i, x, rt.trial_ys[i]

    def run(self, max_trials):
        self.run_sequential(max_trials)
        #if self.async_eval is None:
        #    self.run_sequential(max_trials)
        #else:
        #    raise NotImplementedError()

    def run_sequential(self, max_trials):
        """ Run the Bayesian optimisation for the given number of trials
        """
        self.latent_space._set_input_bounds(self.bounds)
        self._check_settings()
        rt = self.rt  # runtime data
        assert not rt.running
        rt.running = True
        rt.max_trials = max_trials  # TODO: naming (overall vs this run)
        self._notify('run_started', rt.finished_trials, max_trials)

        while rt.finished_trials < max_trials:
            trial_num = rt.started_trials

            x = self._select_trial(trial_num)
            params_dict = self._point_to_dict(self.latent_space.from_latent(x))
            self._notify('evaluation_started', trial_num)

            rt.started_trials += 1
            res = self.objective(**params_dict)
            # the objective function may either return a float, or a tuple of (cost, eval_info)
            y, eval_info = res if isinstance(res, tuple) else (res, None)
            if isinstance(y, int):
                y = float(y)
            assert isinstance(y, float), 'objective function should return a float for the cost, instead: {}'.format(type(y))

            #TODO: assert that y is the correct type and not None, NaN or infinity

            rt.add_finished_trial(x, y)
            self._notify('evaluation_finished', trial_num, y, eval_info)
            rt.check_consistency()

        rt.running = False
        self._notify('run_finished')

    def _check_settings(self):
        """ check that the current optimiser settings make sense

        Raises:
            AssertionError
        """
        pass#TODO

    def _point_to_dict(self, point):
        """ convert the given point (array of values) to a dictionary of parameter names to values

        Note:
            the point should reside in the input space, not the latent space if
            the dictionary is to be fed to the objective function.
        """
        num_params = len(self.bounds)
        assert point.shape == (num_params,) or point.shape == (1, num_params), \
            'invalid point shape: {}'.format(point.shape)
        point = point.flatten()
        return {self.bounds.ordered[i][0] : point[i] for i in range(num_params)}

    def _notify(self, event, *args):
        """ Notify each listener of the given event

        Args:
            event (str): the name of the method to call on each listener in `self.listeners`
            args: the arguments to the method being called

        Note:
            see `turbo.modules.Listener` for the possible methods and arguments
        """
        for l in self._listeners:
            getattr(l, event)(*args)

    def _get_acquisition_function(self, trial_num, model):
        """ instantiate an acquisition function for the given iteration """
        acq_type = self.acquisition.get_type()
        acq_args = [trial_num, model, self.desired_extremum]
        if acq_type == 'optimism':
            pass  # no extra arguments needed
        elif acq_type == 'improvement':
            _, _, incumbent_cost = self.get_incumbent()
            acq_args.append(incumbent_cost)
        else:
            raise NotImplementedError('unsupported acquisition function type: {}'.format(acq_type))
        return self.acquisition.construct_function(*acq_args)

    def _get_trial_type(self, trial_num):
        if trial_num < self.pre_phase_trials:
            return 'pre_phase'
        else:
            if self.fallback.fallback_is_planned(trial_num):
                return 'fallback'
            else:
                return 'bayes'

    def _select_trial(self, trial_num):
        """ Get the next input to evaluate """
        rt = self.rt
        lb = self.latent_space.get_latent_bounds()

        self._notify('selection_started', trial_num)
        trial_type = self._get_trial_type(trial_num)
        selection_info = {'type': trial_type}

        if trial_type == 'pre_phase':
            x = self.pre_phase_select(num_points=1, latent_bounds=lb)

        elif trial_type == 'fallback':
            x = self.fallback.select_trial(self, trial_num)
            selection_info.update({'fallback_reason': 'planned'})

        elif trial_type == 'bayes':
            X, y = np.vstack(rt.trial_xs), np.array(rt.trial_ys)
            model, fitting_info = self.surrogate.construct_model(trial_num, X, y)
            self._notify('surrogate_fitted', trial_num)

            acq_fun, acq_info = self._get_acquisition_function(trial_num, model)
            x, maximisation_info = self.aux_optimiser(lb, acq_fun)
            self._notify('acquisition_maximised', trial_num)

            selection_info.update({'model': model,
                                   'fitting_info': fitting_info,
                                   'acq_info': acq_info,
                                   'maximisation_info': maximisation_info})

            if self.fallback.point_too_close(x, X):
                # keep the selection info from the Bayes selection
                selection_info.update({'type': 'fallback', 'fallback_reason': 'too_close', 'bayes_x': x})
                x = self.fallback.select_trial(self, trial_num)

        else:
            raise ValueError('unknown trial type: {}'.format(trial_type))

        self._notify('selection_finished', trial_num, x, selection_info)
        return x



    #TODO: need to wait for pre_phase to finish completely before continuing when using async.


    '''
    def _next_async_trial(self, iteration):
        if self.async_eval.has_pending_trials():
            if self.plan.in_pre_phase(self.iteration):
                return self.pre_phase_select(n), {}

            # a parallel strategy is required
            #TODO
        else:
            # no pending trials => behaviour the same as when running synchronously
            return self._next_trial()

    def run_async(self, max_trials):
        self._check_configuration()
        started = 0
        trial_xs = []
        trial_ys = []
        while started < max_trials:
            while self.async_eval.get_free_capacity() > 0 and started < max_trials:
                x, extra_data = self._next_async_trial()
                self.async_eval.start_trial(x)
                started += 1

            # wait until more trials can be started
            self.async_eval.wait_for_capacity()
            # then store any trials which finished in the meantime.
            xs, ys = self.async_eval.get_finished_trials(wait=False)
            trial_xs.extend(xs)
            trial_ys.extend(ys)

        # all trials have been started, now wait for them to finish
        xs, ys = self.async_eval.get_finished_trials(wait=True)
        trial_xs.extend(xs)
        trial_ys.extend(ys)
    '''






