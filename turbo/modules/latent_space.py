
import numpy as np

class FixedWarpingLatentSpace:
    def __init__(self, bounds, warped_params=[], warp=np.log, unwarp=np.exp):
        self.bounds = bounds
        self.param_names = set([b[0] for b in bounds])
        self.warped_params = set(warped_params)
        self.warp = warp
        self.unwarp = unwarp

        self.latent_bounds = []
        for b in self.bounds:
            name = b[0]
            pmin, pmax = b[1], b[2]
            if name in self.warped_params:
                pmin, pmax = self.warp(pmin), self.warp(pmax)
            self.latent_bounds.append((name, pmin, pmax))

    def get_latent_bounds(self):
        # for this latent space, the bounds can be precomputed
        return self.latent_bounds

    def from_latent(self, point):
        num_params = len(self.bounds)
        assert point.shape == (num_params,) or point.shape == (1, num_params), \
            'invalid point shape: {}'.format(point.shape)

        point = point.flatten()
        config = {}
        for i in range(num_params):
            name = self.bounds[i][0]
            val = point[i]
            if name in self.warped_params:
                val = self.unwarp(val)
            config[name] = val
        return config

    def to_latent(self, config):
        assert set(config.keys()) == self.param_names, 'invalid configuration'
        point = []
        for b in self.bounds:
            name = b[0]
            val = config[name]
            if name in self.warped_params:
                val = np.warp(val)
            point.append(val)
        return np.array(point)

