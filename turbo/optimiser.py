#!/usr/bin/env python3
''' The Bayesian Optimisation specific code
'''

import numpy as np

# local imports
from .bounds import Bounds
from .optimiser_presets import load_optimiser_preset


class Optimiser:
    def __init__(self, objective, desired_extremum, bounds, pre_phase_trials, settings_preset='default'):
        '''
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
            preset: the name of the preset optimiser settings to load with
                `load_optimiser_preset()`. Pass None to leave the optimiser
                uninitialised for full customisation.
        '''
        self.objective = objective
        #TODO: internally, should use is_maximising or is_minimising where possible
        assert desired_extremum in ('min', 'max'), 'desired_extremum must be either "min" or "max"'
        self.desired_extremum = desired_extremum
        self.bounds = Bounds(bounds)
        self.plan = Optimiser.Plan(pre_phase_trials)

        # modules
        self.latent_space = None
        self.pre_phase_select = None
        self.fallback = None
        self.maximise_acq = None
        self.async_eval = None#TODO
        self.parallel_strategy = None#TODO
        self.surrogate_factory = None
        self.acq_func_factory = None

        # runtime data kept separate from configuration data
        self.rt = Optimiser.Runtime()
        # shouldn't be accessed directly, but through the `register_listener()`
        # and `unregister_listener()` methods
        self._listeners = []

        if settings_preset is not None:
            load_optimiser_preset(self, settings_preset)

    class Plan:
        '''Guides the optimiser's high level decisions such as what technique to use to select the next trial

        Attributes:
            pre_phase_trials: the number of trials to use for the pre-phase
                (before the Bayesian optimisation starts)
            surrogate_train_interval: how often to optimise the hyperparameters
                of the surrogate model. 0 => every iteration. When not training,
                the hyperparameters are reused from the last time they were
                trained.
            hint_with_last_hyper_params: whether to use the last trained
                surrogate hyperparameters as a starting point when optimising
                them the next time.
        '''
        def __init__(self, pre_phase_trials, surrogate_train_interval=0,
                     hint_with_last_hyper_params=True):
            assert pre_phase_trials > 0, 'a pre-phase is required'
            self.pre_phase_trials = pre_phase_trials
            # TODO: maybe the surrogate stuff should be pulled out into a module or put into the surrogate factory?
            self.surrogate_train_interval = surrogate_train_interval
            self.hint_with_last_hyper_params = hint_with_last_hyper_params

    class Runtime:
        ''' holds the data for an optimisation run

        Attributes:
            max_trials: the maximum number of trials in total (not just this
                run) before stopping
            trial_xs: input points (in latent space) for the finished trials
            trial_ys: cost values for the finished trials
            last_trained: the trial_num of the last trial where the surrogate
                model was trained, corresponds to `last_model_params`
        '''
        def __init__(self):
            self.running = False
            self.started_trials = 0
            self.finished_trials = 0
            self.max_trials = 0
            #TODO (naming): these should be finished_xs and finished_ys
            self.trial_xs = [] # list of row vectors
            self.trial_ys = [] # list of scalars
            self.last_trained = -1 # trial_num corresponding to last_model_params
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

    def reset(self):
        ''' clear the runtime data and remove listeners, but keep the optimiser configuration the same '''
        self.rt = Optimiser.Runtime()
        self._listeners = []
        for m in (self.latent_space, self.pre_phase_select, self.fallback,
                  self.maximise_acq, self.async_eval, self.parallel_strategy,
                  self.surrogate_factory, self.acq_func_factory):
            if m is not None:
                m.reset()

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

    def run(self, max_trials):
        if self.async_eval is None:
            self.run_sequential(max_trials)
        else:
            raise NotImplementedError()

    def run_sequential(self, max_trials):
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
            res = self.objective(**params_dict)
            # the objective function may either return a float, or a tuple of (cost, eval_info)
            y, eval_info = res if isinstance(res, tuple) else res, None
            assert isinstance(y, float), 'objective function should return a float for the cost'

            #TODO: assert that y is the correct type and not None, NaN or infinity

            rt.add_finished_trial(x, y)
            self._notify('eval_finished', trial_num, y, eval_info)
            rt.check_consistency()

        rt.running = False
        self._notify('run_finished')



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

    def _get_trial_type(self, trial_num):
        if trial_num < self.plan.pre_phase_trials:
            return 'pre_phase'
        else:
            if self.fallback.fallback_is_planned(trial_num):
                return 'fallback'
            else:
                return 'bayes'

    def _should_train_model(self, trial_num):
        trials_since = trial_num - self.rt.last_trained - 1
        return self.rt.last_trained == -1 or trials_since >= self.plan.surrogate_train_interval

    def _get_hyper_params_hints(self):
        if self.plan.hint_with_last_hyper_params:
            last = self.rt.last_model_params
            return None if last is None else [last]
        else:
            return None

    def _fit_surrogate(self, X, y, trial_num, train):
        model = self.surrogate_factory()
        if train:
            model.fit(X, y, hyper_params_hints=self._get_hyper_params_hints())
            self.rt.last_model_params = model.get_hyper_params()
            self.rt.last_trained = trial_num
        else:
            model.fit(X, y, fixed_hyper_params=self.rt.last_model_params)
        return model

    def _select_trial(self, trial_num):
        ''' Get the next input to evaluate '''
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
            self._notify('fitting_surrogate', trial_num, X, y)
            train = self._should_train_model(trial_num)
            model = self._fit_surrogate(X, y, trial_num, train)
            acq = self._get_acquisition_function(trial_num, model)
            self._notify('maximising_acq', trial_num, acq)
            x, ac_x = self.maximise_acq(lb, acq)
            selection_info.update({'ac_x': ac_x, 'model': model, 'trained': train})

            if self.fallback.point_too_close(x, X):
                # keep the selection info from the Bayes selection
                selection_info.update({'type': 'fallback', 'fallback_reason': 'too_close', 'bayes_x':x})
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






