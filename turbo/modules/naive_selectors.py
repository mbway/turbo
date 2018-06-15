#!/usr/bin/env python3
"""
Modules for selecting points in the latent space for sampling which do not try
to make intelligent decisions, instead sampling randomly or quasi-randomly.

The interface is designed such that each call will generate the next N items in
a sequence. Any persistent data required to keep track of the sequence should be
handled internally by the module.
"""
import numpy as np

#TODO: manual selector in input space and with dictionaries, get the conversion to latent space from the optimiser
#TODO: is convention in turbo that points should be rows?
#TODO: interactive manual selector which doesn't have a sequence preloaded, but instead prompts the user each time
#TODO: implement grid selector


#TODO: stop using __call__ and rename to be camel case
class manual_selector:
    def __init__(self, points):
        """
        Args:
            points: a list of points numpy arrays of shape (1, -1) (row) in latent space.
        """
        assert len(points) > 0, 'empty manual sequence'
        self.points = points
        num_elements = points[0].shape[1]
        assert all(p.shape == (1, num_elements) for p in points), \
            'invalid points in manual sequence'
        self.index = 0  # current index into points

    def __call__(self, num_points, latent_bounds):
        assert self.index + num_points <= len(self.points), 'Manual sequence exhausted!'
        samples = np.vstack(self.points[self.index:self.index+num_points])
        self.index += num_points
        return samples


class random_selector:
    """ select points uniform-randomly in the latent space """
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
        pass#TODO


class LHS_selector:
    """Latin Hypercube sampling selector """
    def __init__(self, num_total):
        self.num_total = num_total
        self.sequence = None
        self.index = 0 # index into the sequence

    def __call__(self, num_points, latent_bounds):
        if self.sequence is None:
            # first call, generate the sequence
            # fills points uniformly in each interval
            lower_bounds = np.array([b[1] for b in latent_bounds.ordered])
            ranges = np.array([b[2]-b[1] for b in latent_bounds.ordered])
            n = self.num_total # length of the sequence
            dims = len(latent_bounds)
            # rand is uniform random over [0,1)
            # each row of sequence is a sample
            self.sequence = lower_bounds + ranges * (np.arange(n).reshape(-1,1) + np.random.rand(n, dims)) / n
            # shuffle each dimension
            for i in range(dims):
                self.sequence[:, i] = np.random.permutation(self.sequence[:, i])

        assert self.index + num_points <= len(self.sequence), 'LHS sequence exhausted!'
        samples = self.sequence[self.index:self.index+num_points, :]
        self.index += num_points
        return samples


