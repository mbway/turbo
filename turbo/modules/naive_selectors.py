#!/usr/bin/env python3
'''
Modules for selecting points in the latent space for sampling which do not try
to make intelligent decisions, instead sampling randomly or quasi-randomly.

The interface is designed such that each call will generate the next N items in
a sequence. Any persistent data required to keep track of the sequence should be
handled internally by the module.
'''
import numpy as np


class random_selector:
    ''' select points uniform-randomly in the latent space
    '''
    def __init__(self):
        pass
    def __call__(self, num_points, latent_bounds):
        # generate values for each parameter
        cols = []
        for name, pmin, pmax in latent_bounds.ordered:
            cols.append(np.random.uniform(pmin, pmax, size=(num_points, 1)))
        return np.hstack(cols)

class random_selector_with_tolerance:
    def __init__(self, optimiser, close_tolerance=1e-8):
        self.optimiser = optimiser
        self.close_tolerance = close_tolerance
    def __call__(self, num_points, latent_bounds):
        pass

