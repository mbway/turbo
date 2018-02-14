#!/usr/bin/env python3
'''
For various reasons it may be beneficial to do the optimisation process in a
latent space, different to the space of parameters which the objective function
takes (the 'input space').

For example a common technique with random search is to sample log-uniformly for
some parameters where appropriate. Similarly it may be beneficial for some
parameters to be transformed to log space for the optimisation, and then back to
the input space when passed to the objective function.

More advanced techniques are also possible, such as supporting categorical
parameters by one-hot encoding using multiple dimensions in the latent space for
a single categorical/discrete dimension in the input space.
'''

import math
import numpy as np

# local imports
from turbo.optimiser import Bounds
from turbo.utils import remap


class LatentSpace:
    '''
    Attributes:
        input_bounds: the Bounds for the input space, provided by the optimiser
            at the start of a run
    '''
    def reset(self):
        ''' called when the optimiser is reset '''
        raise NotImplementedError()
    def set_input_bounds(self, input_bounds):
        ''' called by the optimiser to initialise the latent space '''
        raise NotImplementedError()
    def get_latent_bounds(self):
        ''' get a `Bounds` instance for the latent space

        Warning:
            the entries in the latent space bounds may not correspond exactly
            with the input space bounds because a single input space parameter
            could map to 0, 1 or several latent space parameters and the
            ordering could be different.
            Use `get_latent_param_names()` to determine which entries in the latent
            space bounds correspond to an input space parameter.
        '''
        raise NotImplementedError()
    def get_latent_param_names(self, param_name):
        '''get the corresponding latent space parameter names for the given input space parameter name

        see the `get_latent_bounds()` documentation for explanation of why this is required.

        Args:
            param_name (str): an input space parameter name to be queried

        Returns:
            a list of names relating to `get_latent_bounds()`
        '''
        raise NotImplementedError()
    def param_to_latent(self, param_name, param_val):
        ''' convert a value for the given parameter from input space to latent space '''
        raise NotImplementedError()
    def param_from_latent(self, param_name, param_val):
        ''' convert a value for the given parameter from latent space to input space '''
        raise NotImplementedError()
    def to_latent(self, point):
        ''' convert a point from input space to latent space '''
        raise NotImplementedError()
    def from_latent(self, point):
        ''' convert a point from latent space to input space '''
        raise NotImplementedError()

    def linear_latent_range(self, param, divisions):
        '''Get a range of values evenly spaced in the latent space spanning the
        entire range of the given parameter.

        picture the latent space as being the 'natural' or untransformed space,
        draw a line in that space spanning the whole range of the parameter,
        then perform the inverse mapping (latent to input space mapping) on
        every point of that line and see where it ends up. This function does
        the same thing but with a finite number of points.

        As far as I know, the mapping *does not* need any special property such
        as distance preservation (isometry), bijectivity, or monotonicity for
        this method to apply.

        For example, if the warping function is the natural logarithm, then the
        evenly spaced range will be equivalent to
        `np.logspace(np.log(min), np.log(max), num=divisions, base=np.e)`

        Args:
            param (str): the parameter who's range will be returned
            divisions (int): the number of points to use for the list
        '''
        # not using latent_bounds since there is not a guaranteed 1:1
        # relationship between parameters in the input and latent spaces, and so
        # the latent space uses different parameter names
        pmin, pmax = self.input_bounds.get(param)
        pmin, pmax = self.param_to_latent(param, pmin), self.param_to_latent(param, pmax)
        return np.linspace(pmin, pmax, num=divisions)

class ConstantMap:
    '''A mapping for a single parameter between the latent and input spaces
    which does not change throughout the optimisation.
    '''
    def input_to_latent(self, param): raise NotImplementedError()
    def latent_to_input(self, param): raise NotImplementedError()



class NoLatentSpace(LatentSpace):
    ''' perform no modification to the input space so that input space == latent space '''
    def reset(self):
        self.input_bounds = None
    def set_input_bounds(self, input_bounds):
        self.input_bounds = input_bounds
    def linear_latent_range(self, param, divisions):
        pmin, pmax = self.input_bounds.get(param)
        return np.linspace(pmin, pmax, num=divisions)
    def get_latent_bounds(self):
        return self.input_bounds
    def get_latent_param_names(self, param_name):
        return [param_name]
    def get_param_bounds(self, param_name):
        return self.input_bounds.get(param_name)
    def param_to_latent(self, param_name, param_val):
        return param_val
    def param_from_latent(self, param_name, param_val):
        return param_val
    def to_latent(self, point):
        return point
    def from_latent(self, point):
        return point




class IdentityMap(ConstantMap):
    '''perform no modification so that latent space = input space for this parameter'''
    def input_to_latent(self, param): return param
    def latent_to_input(self, param): return param

class LogMap(ConstantMap):
    r'''perform a mapping such that latent space is logarithmic

    Note:
        this map is only defined for input space values of :math:`]zero_point, +\infty]`

    '''
    def __init__(self, zero_point=0):
        '''
        Args:
            zero_point: move the point at which the parameter value maps to
                infinity. This parameter is useful for moving the asymptote
                outside of the input space range to avoid crashing.
        '''
        self.zero_point = zero_point
    def input_to_latent(self, param): return math.log(param-self.zero_point)
    def latent_to_input(self, param): return math.exp(param)+self.zero_point

class LinearMap(ConstantMap):
    '''linearly remap the input space boundaries to the given boundaries

    eg if the input space bounds are 50 to 100.5 and you want to map this to
    `[0,1]` in the latent space, use:
    `LinearMap((50, 100.5), (0,1))`
    '''
    def __init__(self, input_space_range, latent_space_range):
        '''
        Args:
            input_space_range (tuple): (input_space_min, input_space_max)
            latent_space_range (tuple): (latent_space_min, latent_space_max)
        '''
        self.input_range = input_space_range
        self.latent_range = latent_space_range
    def input_to_latent(self, param):
        return remap(param, self.input_range, self.latent_range)
    def latent_to_input(self, param):
        return remap(param, self.latent_range, self.input_range)


class ConstantLatentSpace(LatentSpace):
    r''' A latent space where the mapping between the input and latent spaces
    does not change throughout the optimisation

    Assumptions:

    - the mappings are monotonic so that every point lies within the mapped min
      and max boundary values. This is so that `get_latent_bounds()` is correct
      and that `linear_latent_range()` is correct.
    '''
    def __init__(self, mappings):
        '''
        Args:
            mappings (dict): a dictionary of param_name to ConstantMap for every parameter
        '''
        self.mappings = mappings
        self.input_bounds = None
        self.latent_bounds = None

    def reset(self):
        self.input_bounds = None
        self.latent_bounds = None

    def set_input_bounds(self, input_bounds):
        assert set(self.mappings.keys()) == set(input_bounds.params), \
            'parameters with mappings differs from those of the bounds'
        assert all(isinstance(m, ConstantMap) for m in self.mappings.values()), \
            'mappings must be constant'
        self.input_bounds = input_bounds

        latent_bounds = []
        for b in input_bounds.ordered:
            name = b[0]
            pmin, pmax = self.param_to_latent(name, b[1]), self.param_to_latent(name, b[2])
            latent_bounds.append(('latent_' + name, pmin, pmax))

        self.latent_bounds = Bounds(latent_bounds)

    def get_latent_bounds(self):
        # for this latent space, the bounds can be precomputed
        return self.latent_bounds
    def get_latent_param_names(self, param_name):
        return ['latent_' + param_name]

    def _transform_point(self, point, to_latent):
        '''Convert a point to or from the latent space

        Args:
            to_latent (bool): True/False to determine the mapping direction. False => from_latent
        '''
        num_params = len(self.input_bounds)
        assert point.shape == (num_params,) or point.shape == (1, num_params), \
            'invalid point shape: {}'.format(point.shape)
        point = point.flatten() # makes a copy
        for i in range(num_params):
            name = self.input_bounds.ordered[i][0]
            point[i] = self.param_to_latent(name, point[i]) if to_latent \
                else self.param_from_latent(name, point[i])
        return point

    def param_to_latent(self, param_name, param_val):
        return self.mappings[param_name].input_to_latent(param_val)
    def param_from_latent(self, param_name, param_val):
        return self.mappings[param_name].latent_to_input(param_val)

    def to_latent(self, point):
        ''' transform the point from the input space to the latent space '''
        return self._transform_point(point, to_latent=True)
    def from_latent(self, point):
        ''' transform the point from the latent space to the input space '''
        return self._transform_point(point, to_latent=False)


