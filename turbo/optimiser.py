#!/usr/bin/env python3
''' The Bayesian Optimisation specific code
'''

import numpy as np

class Bounds:
    '''Boundaries of a space
    '''
    def __init__(self, ordered):
        #TODO: do checks like duplicates etc
        self.ordered = ordered
        self.params = set([b[0] for b in ordered])
        self.associative = {b[0] : (b[1], b[2]) for b in ordered}
    def __len__(self):
        return len(self.ordered)
    def get(self, param):
        return self.associative[param]
    def get_param_index(self, param):
        '''Get the index within a point where the value for the given parameter
        can be found.

        For example, given a point `p`:
        `opt.point_to_config(p)[param] == p[opt.bounds.get_param_index(param)]`

        '''
        for i, b in enumerate(self.ordered):
            if param == b[0]:
                return i
        raise KeyError()

class OptimiserRuntime:
    def __init__(self):
        self.running = False
        self.started_trials = 0
        self.finished_trials = 0
        self.max_trials = 0
        self.trial_xs = [] # list of row vectors
        self.trial_ys = [] # list of scalars
        self.last_model_params = None

    def num_trials(self):
        return len(self.trial_ys)



class Optimiser:
    def __init__(self, objective, desired_extremum, bounds):
        self.objective = objective
        self.desired_extremum = desired_extremum # TODO: sanitise
        self.bounds = Bounds(bounds)

        self.latent_space = None
        self.plan = None
        self.pre_phase_select = None
        self.maximise_acq = None
        self.async_eval = None
        self.parallel_strategy = None

        self.surrogate_factory = None
        self.acq_func_factory = None

        # shouldn't be accessed directly, but through the `register_listener()`
        # and `unregister_listener()` methods
        self._listeners = []

        # runtime data
        self.rt = OptimiserRuntime()

    def load_settings(self, settings):
        #TODO: this could be the mechanism for setting defaults. Have provided defaults dictionaries which can be loaded?
        #TODO: no duplicate names in bounds
        pass

    def point_to_config(self, point):
        num_params = len(self.bounds)
        assert point.shape == (num_params,) or point.shape == (1, num_params), \
            'invalid point shape: {}'.format(point.shape)

        config = {}
        for i in range(num_params):
            name = self.bounds.ordered[i][0]
            config[name] = point[i]
        return config

    def config_to_point(self, config):
        assert set(config.keys()) == self.bounds.params, 'invalid configuration'
        point = []
        for b in self.bounds.ordered:
            name = b[0]
            point.append(config[name])
        return np.array(point)


    def is_maximising(self):
        return self.desired_extremum == 'max'
    def is_minimising(self):
        return self.desired_extremum == 'min'

    def _check_settings(self):
        ''' check that the current optimiser settings make sense

        Raises:
            AssertionError
        '''
        pass

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

    def restart(self):
        self.rt = OptimiserRuntime()

    def _select_trial(self, trial_num):
        ''' Get the next input to evaluate

        Returns:
            The next trial
        '''
        rt = self.rt
        lb = self.latent_space.get_latent_bounds()

        if self.plan.in_pre_phase(trial_num):
            x = self.pre_phase_select(num_points=1, latent_bounds=lb)
            extra_data = {'type':'pre_phase'}
        else:
            X, y = np.vstack(rt.trial_xs), np.array(rt.trial_ys)
            self._notify('fitting_surrogate', trial_num, X, y)
            model = self.surrogate_factory(X, y)
            acq = self.acq_func_factory(trial_num, model, self.desired_extremum)
            self._notify('maximising_acq', trial_num, acq)
            x, ac_x = self.maximise_acq(lb, acq)
            extra_data = {'type':'bayes', 'ac_x':ac_x, 'model':model}
        return x, extra_data


    def run(self, max_trials):
        ''' Run the Bayesian optimisation for the given number of trials
        '''
        self.latent_space.set_bounds(self.bounds)
        self._check_settings()
        rt = self.rt # runtime data
        rt.running = True
        rt.max_trials = max_trials # TODO: naming
        self._notify('run_started', rt.finished_trials, max_trials)

        while rt.finished_trials < max_trials:
            trial_num = rt.started_trials

            self._notify('selection_started', trial_num)
            #TODO: could call extra data 'selection_details'?
            x, extra_data = self._select_trial(trial_num)
            config = self.point_to_config(self.latent_space.from_latent(x))
            self._notify('eval_started', trial_num, x, extra_data)

            rt.started_trials += 1
            y = self.objective(**config)
            #TODO: assert that y is the correct type and not None, NaN or infinity

            rt.trial_xs.append(x)
            rt.trial_ys.append(y)
            rt.finished_trials += 1
            self._notify('eval_finished', trial_num, y)

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




