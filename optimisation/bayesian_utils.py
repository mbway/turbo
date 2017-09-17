#!/usr/bin/env python3
'''
Utilities specifically for Bayesian optimisation
'''
# python 2 compatibility
from __future__ import (absolute_import, division, print_function, unicode_literals)
from .py2 import *

import numpy as np

import sklearn.gaussian_process as gp
from sklearn.base import clone as sk_clone
import scipy.optimize

# local modules
from .utils import *



#TODO: test
def store_GP(gp_model):
    '''
    store the current parameters of the given Gaussian process for efficient
    storage. The GP must have been fitted to some data set before storing.

    The restored GP will be identical to the original if restored with the same
    gp_params and data set. The GP can also be restored with a different data
    set, which has the effect of transferring the hyperparameters without
    re-optimising them for the new data set.
    '''
    return np.copy(gp_model.kernel_.theta)

def restore_GP(stored_params, gp_params, xs, ys):
    '''
    restore a stored GP by constructing a new GP with the same parameters and data set
    stored_params: the result of calling store_GP
    gp_params: the parameters that the original GP was initialised with
    xs, ys: the data set that the original GP was trained on
    returns: a new GP which is identical to the original GP
    '''
    gp_model = gp.GaussianProcessRegressor(**gp_params)
    p = gp_model.get_params()
    kernel = p['kernel']
    opt = p['optimizer']
    gp_model.set_params(optimizer=None)
    # don't want to modify the kernel which is part of gp_params, so modify a clone
    gp_model.set_params(kernel=sk_clone(kernel))
    gp_model.kernel.theta = stored_params
    gp_model.fit(xs, ys)
    gp_model.set_params(kernel=kernel, optimizer=opt)
    return gp_model

#TODO: test
def maximise_function(f, bounds, gen_random, num_random, num_grad_restarts, take_best_random):
    '''
    f: the function to maximise, takes an ndarray of shape=(num_points, num_attribs)
    bounds: a list of tuples of (min, max) for each dimension
    gen_random: a function which takes the number of points to sample and
        returns an ndarray with each point as a row.
    num_random: number of random points to sample to search for the maximum
    num_grad_restarts: number of restarts to allow the gradient-based optimiser
        to search for the maximum.
    take_best_random: number of points from the random phase to use as starting
        points in the gradient-based stage. Included in the num_grad_restarts
        total, the remaining points will be chosen at random.
        should be <= num_random and <= num_grad_restarts
    returns: x, y or None, -inf if maximisation fails
        x: shape=(1, num_attribs): the function input which produces the smallest output
        y: float: the value of f(x)
    '''
    assert take_best_random <= num_random and take_best_random <= num_grad_restarts

    # maximise the function by minimising the negation of the function, then
    # negating the results at the end. This is necessary because scipy only
    # offers a gradient based minimiser.

    # keep track of the current best
    best_x = None
    best_y = inf

    # minimise by random sampling
    if num_random > 0:
        random_x = gen_random(num_random)
        random_y = -f(random_x)

        best_ids = np.argsort(random_y, axis=0).flatten() # sorted indices
        best_random_i = best_ids[0] # smallest
        best_x = make2D_row(random_x[best_random_i])
        best_y = random_y[best_random_i]

    # minimise by gradient-based optimiser
    if num_grad_restarts > 0:
        if num_random > 0:
            # see if gradient-based optimisation can improve upon the best
            # samples from the last stage
            # N random xs from the last step with the smallest y values
            best_starts = random_x[best_ids[:take_best_random]]
            new_starts = gen_random(num_grad_restarts - take_best_random)
            starting_points = np.vstack((best_starts, new_starts))
        else:
            starting_points = gen_random(num_grad_restarts)

        for j in range(num_grad_restarts):
            starting_point = make2D_row(starting_points[j])
            # the minimiser passes x as (num_attribs,) but f wants (1,num_attribs)
            neg_f = lambda x: -f(make2D_row(x))

            # result is an OptimizeResult object
            # optimisation process may trigger warnings
            result = scipy.optimize.minimize(
                fun=neg_f,
                x0=starting_point,
                bounds=bounds,
                method='L-BFGS-B', # Limited-Memory Broyden-Fletcher-Goldfarb-Shanno Bounded
                options=dict(maxiter=15000) # maxiter=15000 is default
            )

            if not result.success:
                warnings.warn('restart {}/{} of gradient-based optimisation failed'.format(
                    j, num_grad_restarts))
                continue

            if result.fun < best_y:
                best_x = result.x   # shape=(num_attribs,)
                best_y = result.fun # shape=(1,1)

    if best_x is None:
        return None, -inf
    else:
        best_x = make2D_row(best_x) # shape=(1, num_attribs)
        # ensure that the chosen value lies within the bounds (which may not
        # be the case due to floating point error)
        low_bounds, high_bounds = zip(*bounds)
        best_x = np.clip(best_x, low_bounds, high_bounds)
        best_y = -np.asscalar(best_y) # undo negation
        return best_x, best_y


class Range(DataHolder):
    __slots__ = ('values', 'type_', 'bounds')

class Step(DataHolder):
    '''
    Data regarding a single optimisation step (which results in a single Job).
    Has the data needed to explain the decision process of the optimiser in
    sufficient detail to plot it.

    job_ID: the ID of the job which evaluated the configuration chosen this step.
    sx, sy: point representation of the concrete samples. `x = config` `y = cost`
    hx: hypothesised/in-progress samples during the step `x = config`
    suggestions: a list of suggestion objects describing the suggestions for the
        next configuration to evaluate, in chronological order. The last
        suggestion is the one that is used. (For example, if the suggested
        configuration resulting from maximising the acquisition function is too
        close to an existing sample, a randomly chosen configuration is used
        instead)
    '''
    __slots__ = ('job_ID', 'sx', 'sy', 'hx', 'suggestions')

    def chosen_at_random(self):
        return isinstance(self.suggestions[-1], Step.RandomSuggestion)

    def chosen_x(self):
        return self.suggestions[-1].x

    class RandomSuggestion(DataHolder):
        '''
        A configuration suggestion chosen at random

        x: a randomly sampled configuration
        '''
        __slots__ = ('x')

    class MaxAcqSuggestion(DataHolder):
        '''
        A configuration suggestion chosen by estimating the cost values for
        hypothesised samples if there are any, then maximising an acquisition
        function.

        hy: estimated cost values for the hypothesised configurations: hx
        gp: a stored GP trained on a data set of concrete samples + hypothesised samples
        x: the suggested configuration (= argmax_x acquisition_function(x))
        ac: the value of the acquisition function evaluated at x (should be the maximum)

        ac_random_state: None when not used, or the state of the RNG when the
            acquisition function was evaluated (only used when the acquisition
            function has a random element. Currently this is only the case with
            Thompson Sampling).
        '''
        __slots__ = ('hy', 'gp', 'x', 'ac', 'ac_random_state')
        __defaults__ = {'ac_random_state' : None}

    class MC_MaxAcqSuggestion(DataHolder):
        '''
        A configuration suggestion chosen by sampling N estimates of cost values
        for hypothesised samples if there are any, then maximising the average
        acquisition function value across each Monte-Carlo simulation.

        gp: a stored GP trained on a data set of only the concrete samples
        simulations: [hy, sim_gp] or [hy, sim_gp, ac_random_state]
            each simulation has different values for hy
            hy: estimated cost values sampled from the posterior of the GP
                trained on only concrete samples
            sim_gp: a stored GP trained on a data set consisting of
                concrete_samples + hypothesised samples for this simulation.
                #TODO:
                #For time concerns, these GPs may have the same parameters as
                the GP trained on only the concrete samples.
            ac_random_state: the state of the RNG when the acquisition function
                was evaluated (only used when the acquisition function has a
                random element. Currently this is only the case with Thompson
                Sampling).
        x: the suggested configuration (= argmax_x 1/N sum(acquisition_function^i(x))
        ac: the average acquisition function value across every simulation
            evaluated as x (should be the maximum)

        '''
        __slots__ = ('gp', 'simulations', 'x', 'ac')


#TODO: choose whether to sample log ranges logarithmically or ignore
class PointSpace:
    '''
    deals with the conversion between 'configuration space' which is specified
    by the given 'ranges', and 'point space' which is the space in which the
    computation is performed
    '''

    def __init__(self, params, ranges, close_tolerance):
        self.params = params
        self.ranges = ranges
        self.close_tolerance = close_tolerance

        # Only provide bounds for the parameters that are included in
        # self.config_to_point. Provide the log(lower), log(upper) bounds for
        # logarithmically spaced ranges.
        # IMPORTANT: use range_bounds when dealing with configs and point_bounds
        # when dealing with points
        self.point_bounds = []
        self.param_indices = {}
        i = 0
        for param in self.params:
            range_ = self.ranges[param]
            if range_.type_ == RangeType.Linear:
                self.point_bounds.append(range_.bounds)
                self.param_indices[param] = i
                i += 1
            elif range_.type_ == RangeType.Logarithmic:
                self.point_bounds.append((np.log(range_.bounds[0]), np.log(range_.bounds[1])))
                self.param_indices[param] = i
                i += 1
            else:
                # don't include parameters which do not appear in the point
                # representation in the point bounds
                self.param_indices[param] = None

        # number of columns in a point in point space
        self.num_attribs = len(self.point_bounds)

    def get_grid_points(self):
        '''
        return an ndarray of vertically stacked points which form an evenly
        spaced grid spanning the whole point space.
        '''
        raise NotImplementedError()

    def unique_random_point(self, different_from, num_attempts=1000):
        ''' generate a random point which is sufficiently different from any of the given points.
        different_from: ndarray of points shape=(num_points, num_attribs)
            which the resulting configuration must not be identical to (within a
            very small tolerance). This is because identical configurations
            would break the GP (it models a function, so each 'x' corresponds to
            exactly one 'y')
        num_attempts: number of re-tries before giving up
        returns: a random point, or None if num_attempts is exceeded
        '''
        for _ in range(num_attempts):
            point = self.random_points(1)
            if not close_to_any(point, different_from, tol=self.close_tolerance):
                return point
        return None

    #TODO: test with num_points = 0
    def random_points(self, num_points):
        '''
        generate an array of vertically stacked configuration points equivalent
        num_points: number of points to generate (height of output)
        returns: numpy array with shape=(num_points,num_attribs)
        '''
        cols = [] # generate points column/parameter-wise
        for param in self.params: # self.params is sorted
            range_ = self.ranges[param]
            low, high = range_.bounds # _not_ point bounds

            if range_.type_ == RangeType.Linear:
                cols.append(np.random.uniform(low, high, size=(num_points, 1)))
            elif range_.type_ == RangeType.Logarithmic:
                # note: NOT log_uniform because that computes a value in the
                # original space but distributed logarithmically. We are looking
                # for just the exponent here, not the value.
                cols.append(np.random.uniform(np.log(low), np.log(high), size=(num_points, 1)))
        return np.hstack(cols)


    def param_to_config_space(self, points, param):
        '''
        extract a single parameter from the given points and convert the values
        to configuration space
        points: ndarray shape=(num_points, num_attribs)
        param: the name of the parameter to extract
        returns: ndarray shape=(num_points, 1)
        '''
        i = self.param_indices[param]
        type_ = self.ranges[param].type_
        if type_ == RangeType.Linear:
            return make2D(points[:,i])
        elif type_ == RangeType.Logarithmic:
            return make2D(np.exp(points[:,i]))

    def param_to_point_space(self, vals, param):
        '''
        convert the given values to point space by mapping them into the
        dimension for 'param'
        vals: ndarray
        '''
        type_ = self.ranges[param].type_
        if type_ == RangeType.Linear:
            return vals
        elif type_ == RangeType.Logarithmic:
            return np.log(vals)


    def config_to_point(self, config):
        '''
        convert a configuration (dictionary of param:val) to a point (numpy
        array) in the parameter space that the Gaussian process uses.

        As a point, constant parameters are ignored, and values from logarithmic
        ranges are the exponents of the values. ie a value of 'n' as a point
        corresponds to a value of e^n as a configuration.

        config: a dictionary of parameter names to values
        returns: numpy array with shape=(1,number of linear or logarithmic parameters)
        '''
        assert set(config.keys()) == set(self.ranges.keys())
        elements = []
        for param in self.params: # self.params is sorted
            type_ = self.ranges[param].type_
            if type_ == RangeType.Linear:
                elements.append(config[param])
            elif type_ == RangeType.Logarithmic:
                elements.append(np.log(config[param]))
        return np.array([elements])

    def point_to_config(self, point):
        '''
        convert a point (numpy array) used by the Gaussian process into a
        configuration (dictionary of param:val).

        As a point, constant parameters are ignored, and values from logarithmic
        ranges are the exponents of the values. ie a value of 'n' as a point
        corresponds to a value of e^n as a configuration.

        returns: a configuration dict with all parameters included
        '''
        assert len(point.shape) == 2, 'must be a 2D point'
        assert point.shape[0] == 1, 'only 1 point can be converted at a time'
        assert point.shape[1] == self.num_attribs, 'wrong number of attributes'
        config = {}
        pi = 0 # current point index
        for param in self.params: # self.params is sorted
            range_ = self.ranges[param]

            if range_.type_ == RangeType.Constant:
                config[param] = range_.values[0] # only 1 choice
            else:
                val = point[0, pi]
                pi += 1

                if range_.type_ == RangeType.Linear:
                    config[param] = val
                elif range_.type_ == RangeType.Logarithmic:
                    config[param] = np.exp(val)
        return dotdict(config)


