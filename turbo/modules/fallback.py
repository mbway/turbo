#!/usr/bin/env python3

import numpy as np

from turbo.utils import close_to_any


class Fallback:
    """ Optimiser fallback behaviour

    Manages the behaviour for falling back to some other strategy, either to
    add diversity to the trials or when the trial suggested by Bayesian
    optimisation is too close to an existing sample.

    Attributes:
        planned_fallback: a function which takes a trial number and the trial
            number of the last planned fallback and returns whether the fallback
            method should be used instead of Bayesian optimisation. (None to disable)

            helper functions are provided: `Fallback.planned_with_probability()`
            and `Fallback.planned_with_interval()` for easily constructing these
            functions.
        close_tolerance: the maximum Euclidean distance considered 'too close',
            causing a Bayesian optimisation trial to be discarded and the
            fallback method used instead (not planned) (<0 to disable)
        selector: the selector to use during fallback. If None then the `pre_phase_selector` is used
    """
    def __init__(self, planned_fallback=None, close_tolerance=1e-10, selector=None):
        self.planned_fallback = planned_fallback
        self.close_tolerance = close_tolerance
        self.selector = selector
        self._last_planned_fallback = -1

    def fallback_is_planned(self, trial_num):
        """ whether a fallback is planned for this trial (queried before performing Bayesian optimisation) """
        if self.planned_fallback is None:
            return False
        else:
            if self._last_planned_fallback == -1:
                # on the first Bayesian optimisation trial, behave like a fallback
                # had just happened so we start counting from now.
                self._last_planned_fallback = trial_num - 1

            planned = self.planned_fallback(trial_num, self._last_planned_fallback)
            if planned:
                self._last_planned_fallback = trial_num
            return planned

    def point_too_close(self, x, X):
        """ whether the trial point x is too close to any of the points in X """
        if self.close_tolerance < 0:
            return False
        else:
            return close_to_any(x, X, tol=self.close_tolerance)

    def select_trial(self, optimiser, trial_num): # TODO: unused argument
        """ select a trial using the fallback method """
        lb = optimiser.latent_space.get_latent_bounds()
        selector = self.selector or optimiser.pre_phase_select
        return selector(num_points=1, latent_bounds=lb)

    @staticmethod
    def planned_with_probability(p):
        """ pass to Fallback constructor to use fallback with the given probability

        Args:
            p (float): the probability [0,1] that the trial should use the
                fallback method instead of Bayesian optimisation.
        """
        return lambda trial_num, last_planned_fallback: np.random.uniform(0, 1) < p

    @staticmethod
    def planned_with_interval(interval):
        """ pass to Fallback constructor to use fallback once every `interval` iterations """
        return lambda trial_num, last_planned_fallback: trial_num - last_planned_fallback >= interval

