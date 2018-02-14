#!/usr/bin/env python3

from turbo.utils import close_to_any

class Fallback:
    '''Optimiser fallback behaviour

    Manages the behaviour for falling back to some other strategy, either to
    add diversity to the trials or when the trial suggested by Bayesian
    optimisation is too close to an existing sample.

    Attributes:
        probability (float): the probability [0,1] that the trial
            should use the fallback method instead of Bayesian optimisation.
            (planned fallback) (0.0 to disable)
        interval (int): the interval at which the fallback
            method should be used instead of Bayesian optimisation (planned
            fallback) (None to disable)
        close_tolerance: the maximum Euclidean distance considered 'too close',
            causing a Bayesian optimisation trial to be discarded and the
            fallback method used instead (not planned) (0.0 to disable)
    '''
    def __init__(self, probability=0.0, interval=None, close_tolerance=1e-10):
        self.probability = probability
        self.interval = interval
        self.close_tolerance = close_tolerance
        self._last_planned_fallback = -1

    def reset(self):
        ''' called when the optimiser resets '''
        self._last_planned_fallback = -1

    def fallback_is_planned(self, trial_num):
        ''' whether a fallback is planned for this trial (queried before performing Bayesian optimisation) '''
        if self._last_planned_fallback == -1:
            self._last_planned_fallback = trial_num - 1

        if (self.interval is not None and trial_num - self._last_planned_fallback - 1 >= self.interval) or \
           (self.probability != 0.0 and np.random.uniform() < self.probability):

            self._last_planned_fallback = trial_num
            return True
        else:
            return False

    def point_too_close(self, x, X):
        ''' whether the trial point x is too close to any of the points in X '''
        if self.close_tolerance == 0.0:
            return False
        else:
            return close_to_any(x, X, tol=self.close_tolerance)

    def select_trial(self, optimiser, trial_num):
        ''' select a trial using the fallback method '''
        lb = optimiser.latent_space.get_latent_bounds()
        return optimiser.pre_phase_select(num_points=1, latent_bounds=lb)

