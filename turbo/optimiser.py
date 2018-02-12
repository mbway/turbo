#!/usr/bin/env python3
''' The Bayesian Optimisation specific code
'''

import numpy as np

# local imports
from .bounds import Bounds
from .optimiser_presets import load_optimiser_preset


class Optimiser:
    def __init__(self, objective, desired_extremum, bounds, settings_preset='default'):
        '''
        Args:
            objective: a function to be optimised, which accepts parameters
                corresponding to the given bounds and returns an objective/cost
                value for the input.
            desired_extremum: either 'min' or 'max' specifying whether the
                objective function is to be maximised or minimised
            bounds: a list of tuples of (name, min, max) for each parameter of
                the objective function
            preset: the name of the preset optimiser settings to load with
                `load_optimiser_preset()`. Pass None to leave the optimiser
                uninitialised for full customisation.
        '''
        self.objective = objective
        #TODO: internally, should use is_maximising or is_minimising where possible
        assert desired_extremum in ('min', 'max'), 'desired_extremum must be either "min" or "max"'
        self.desired_extremum = desired_extremum
        self.bounds = Bounds(bounds)

        # modules
        self.latent_space = None
        self.plan = None
        self.pre_phase_select = None
        self.maximise_acq = None
        self.async_eval = None
        self.parallel_strategy = None
        self.surrogate_factory = None
        self.acq_func_factory = None

        # runtime data kept separate from configuration data
        self.rt = Optimiser.Runtime()
        # shouldn't be accessed directly, but through the `register_listener()`
        # and `unregister_listener()` methods
        self._listeners = []

        if settings_preset is not None:
            load_optimiser_preset(self, settings_preset)

    class Runtime:
        ''' holds the data for an optimisation run '''
        def __init__(self):
            self.running = False
            self.started_trials = 0
            self.finished_trials = 0
            self.max_trials = 0
            self.trial_xs = [] # list of row vectors
            self.trial_ys = [] # list of scalars
            self.last_model_params = None

        def check_consistency(self):
            '''check that the optimiser runtime data makes sense

            Raises:
                AssertionError
            '''
            assert self.started_trials >= self.finished_trials
            assert len(self.trial_xs) == len(self.trial_ys)
            assert len(self.trial_ys) == self.finished_trials

        def add_finished_trial(self, x, y):
            self.trial_xs.append(x)
            self.trial_ys.append(y)
            self.finished_trials += 1

    def _check_settings(self):
        ''' check that the current optimiser settings make sense

        Raises:
            AssertionError
        '''
        pass#TODO

    def _point_to_dict(self, point):
        ''' convert the given point (array of values) to a dictionary of parameter names to values

        Note:
            the point should reside in the input space, not the latent space if
            the dictionary is to be fed to the objective function.
        '''
        num_params = len(self.bounds)
        assert point.shape == (num_params,) or point.shape == (1, num_params), \
            'invalid point shape: {}'.format(point.shape)
        point = point.flatten()
        return {self.bounds.ordered[i][0] : point[i] for i in range(num_params)}

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

    def _notify(self, event, *args):
        '''Notify each listener of the given event

        Args:
            event (str): the name of the method to call on each listener in `self.listeners`
            args: the arguments to the method being called

        Note:
            see `turbo.modules.Listener` for the possible methods and arguments
        '''
        for l in self._listeners:
            getattr(l, event)(*args)

    def reset(self):
        ''' clear the runtime data but keep the optimiser configuration the same '''
        self.rt = Optimiser.Runtime()
        self._listeners = []

    def get_incumbent(self):
        '''get the current best trial

        Returns:
            (i, x, y)
            i = trial number
            x = the trial input
            y = the trial objective function value
        '''
        rt = self.rt
        i = np.argmax(rt.trial_ys) if self.is_maximising() else np.argmin(rt.trial_ys)
        return (i, rt.trial_xs[i], rt.trial_ys[i])

    def _get_acquisition_function(self, trial_num, model):
        ''' instantiate an acquisition function for the given iteration '''
        acq_type = self.acq_func_factory.get_type()
        acq_args = [trial_num, model, self.desired_extremum]
        if acq_type == 'optimism':
            pass # no extra arguments needed
        elif acq_type == 'improvement':
            _, _, incumbent_cost = self.get_incumbent()
            acq_args.append(incumbent_cost)
        else:
            raise NotImplementedError('unsupported acquisition function type: {}'.format(acq_type))
        return self.acq_func_factory(*acq_args)

    def _select_trial(self, trial_num):
        ''' Get the next input to evaluate '''
        rt = self.rt
        lb = self.latent_space.get_latent_bounds()

        self._notify('selection_started', trial_num)
        if self.plan.in_pre_phase(trial_num):
            x = self.pre_phase_select(num_points=1, latent_bounds=lb)
            selection_details = {'type': 'pre_phase'}
        else:
            X, y = np.vstack(rt.trial_xs), np.array(rt.trial_ys)
            self._notify('fitting_surrogate', trial_num, X, y)
            model = self.surrogate_factory(X, y, hyper_params_hint=rt.last_model_params)
            rt.last_model_params = model.get_hyper_params()
            acq = self._get_acquisition_function(trial_num, model)
            self._notify('maximising_acq', trial_num, acq)
            x, ac_x = self.maximise_acq(lb, acq)
            selection_details = {'type': 'bayes', 'ac_x': ac_x, 'model': model}
        self._notify('selection_finished', trial_num, x, selection_details)
        return x


    def run(self, max_trials):
        ''' Run the Bayesian optimisation for the given number of trials
        '''
        self.latent_space.set_input_bounds(self.bounds)
        self._check_settings()
        rt = self.rt # runtime data
        rt.running = True
        rt.max_trials = max_trials # TODO: naming (overall vs this run)
        self._notify('run_started', rt.finished_trials, max_trials)

        while rt.finished_trials < max_trials:
            trial_num = rt.started_trials

            x = self._select_trial(trial_num)
            params_dict = self._point_to_dict(self.latent_space.from_latent(x))
            self._notify('eval_started', trial_num)

            rt.started_trials += 1
            y = self.objective(**params_dict)
            #TODO: assert that y is the correct type and not None, NaN or infinity

            rt.add_finished_trial(x, y)
            self._notify('eval_finished', trial_num, y)
            rt.check_consistency()

        rt.running = False
        self._notify('run_finished')






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






