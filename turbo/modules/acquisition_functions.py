#!/usr/bin/env python3

import numpy as np
from math import isinf

import sys

# local modules
from turbo.utils import *


class UCB:
    r'''Confidence bound acquisition function

    .. math::
        \begin{align*}
        UCB(\mathbf{x})&=\mu(\mathbf{x})+\beta\sigma(\mathbf{x})\\
        LCB(\mathbf{x})&=-(\mu(\mathbf{x})-\beta\sigma(\mathbf{x}))
        \end{align*}

    Args:
        model: the surrogate model fitted to the past trials.
        desired_extremum: `'max' =>` higher cost is better, `'min' =>` lower
            cost is better.
        beta: parameter which controls the trade-off between exploration and
            exploitation. Larger values favour exploration more. (geometrically,
            the uncertainty is scaled more so is more likely to look better than
            known good locations).
            'often beta=2 is used' (Bijl et al., 2016).

            :math:`\beta=0` `=>` 'Expected Value' (EV) acquisition function

            .. math::
                EV(\mathbf x)=\mu(\mathbf x)

            (pure exploitation, not very useful).

            :math:`\beta=\infty` `=>` pure exploration, taking only the variance
            into account (also not very useful)

    Note:
        Technically this function is the negative lower confidence bound when
        minimising the objective function; however, the name UCB is used for
        simplicity.
    '''

    def __init__(self, beta, model, desired_extremum):
        self.beta = beta
        self.model = model
        self.scale_factor = 1 if desired_extremum == 'max' else -1
        self.name = 'UCB' if desired_extremum == 'max' else '-LCB'

    def get_name(self):
        return self.name

    def __call__(self, X):
        '''
        Args:
            X: the array of points to evaluate at. `shape=(num_points, num_attribs)`
        '''
        mus, sigmas = self.model.predict(X, return_std_dev=True)
        if isinf(self.beta):
            return sigmas # pure exploration
        else:
            # in this form it is clearer that the value is the negative LCB when minimising
            # sf * (mus + sf * beta * sigmas)
            return self.scale_factor * mus + self.beta * sigmas

class UCB_AcquisitionFactory:
    def __init__(self, beta):
        self.beta = beta

    def __call__(self, trial_num, model, desired_extremum):
        if callable(self.beta):
            b = self.beta(trial_num)
        else:
            b = self.beta # beta is a constant
        return UCB(b, model, desired_extremum)

