#!/usr/bin/env python3

import numpy as np

# local imports
from turbo.optimiser import Bounds

class FixedWarpingLatentSpace:
    def __init__(self, warped_params=[], warp=np.log, unwarp=np.exp):
        self.warped_params = set(warped_params)
        self.warp = warp
        self.unwarp = unwarp

    def set_bounds(self, bounds):
        assert all(p in bounds.params for p in self.warped_params), \
            'invalid warped parameter names'
        self.bounds = bounds

        latent_bounds = []
        for b in self.bounds.ordered:
            name = b[0]
            pmin, pmax = b[1], b[2]
            if name in self.warped_params:
                pmin, pmax = self.warp(pmin), self.warp(pmax)
            latent_bounds.append((name, pmin, pmax))

        self.latent_bounds = Bounds(latent_bounds)

    def evenly_spaced_range(self, param, divisions):
        '''Get a range of values evenly spaced in the latent space spanning the
        entire range of the given parameter.

        For example, if the warping function is the natural logarithm, then the
        evenly spaced range will be equivalent to
        `np.logspace(np.log(min), np.log(max), num=divisions, base=np.e)`

        Args:
            param (str): the parameter who's range will be returned
            divisions (int): the number of points to use for the list
        '''
        pmin, pmax = self.bounds.get(param)
        if param in self.warped_params:
            pmin, pmax = self.warp(pmin), self.warp(pmax)
        return np.linspace(pmin, pmax, num=divisions)

    def get_latent_bounds(self):
        # for this latent space, the bounds can be precomputed
        return self.latent_bounds

    def _convert_point(self, point, warp):
        '''Convert a point to or from the latent space

        Args:
            warp (bool): whether to `warp` or `unwarp` the warped parameters
        '''
        num_params = len(self.bounds)
        assert point.shape == (num_params,) or point.shape == (1, num_params), \
            'invalid point shape: {}'.format(point.shape)
        point = point.flatten()
        for i in range(num_params):
            name = self.bounds.ordered[i][0]
            if name in self.warped_params:
                point[i] = self.warp(point[i]) if warp else self.unwarp(point[i])
        return point

    def to_latent(self, point):
        return self._convert_point(point, warp=True)
    def from_latent(self, point):
        return self._convert_point(point, warp=False)


class NoLatentSpace(FixedWarpingLatentSpace):
    def __init__(self):
        id_fun = lambda x: x
        super().__init__(warped_params=[], warp=id_fun, unwarp=id_fun)

