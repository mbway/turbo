#!/usr/bin/env python3
'''
The Bayesian Optimisation specific code
'''

import interfaces.initialisation

class OptimiserRuntime:
    __slots__ = (
        'iteration',
        'trial_xs', 'trial_ys'
    )
    def __init__(self):
        self.iteration = 0
        self.trial_xs = []
        self.trial_ys = []
        self.last_model_params = None



class Optimiser:
    def __init__(self, objective, optimal, bounds):
        self.objective = objective
        self.optimal = optimal
        self.bounds = bounds

        self.latent_space = None
        self.plan = None
        self.pre_phase_select = None
        self.maximise_acq = None
        self.async_eval = None
        self.parallel_strategy = None

        self.surrogate_factory = None
        self.acq_func_factory = None

        self.logger = None

        # runtime data
        self.rt = OptimiserRuntime()

    def load_settings(self, settings):
        #TODO: this could be the mechanism for setting defaults. Have provided defaults dictionaries which can be loaded?
        pass

    def _check_settings(self):
        ''' check that the current optimiser settings make sense
        Raises: AssertionError
        '''
        pass

    def _next_trial(self):
        rt = self.rt
        if self.plan.in_pre_phase(rt.it):
            lb = self.latent_space.get_latent_bounds()
            return self.pre_phase_select(num_points=1, latent_bounds=lb), {}
        model = self.surrogate_factory.new_model()
        model.fit(rt.trial_xs, rt.trial_ys)
        acq = self.acq_func_factory.new_function(rt.it, model)
        x, ac_x = self.maximise_acquisition(acq)
        return x, {'ac_x':ac_x, 'model':model}

    def run(self, max_trials):
        self._check_configuration()
        rt = self.rt # runtime data
        while rt.iteration < max_trials:
            x, extra_data = self._next_trial()
            config = self.latent_space.from_latent(x)
            y = self.objective(config)

            rt.trial_xs.append(x)
            rt.trial_ys.append(y)
            rt.iteration += 1






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




