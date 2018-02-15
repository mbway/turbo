#!/usr/bin/env python3

import matplotlib as mpl
import numpy as np

class MidpointNorm(mpl.colors.Normalize):
    '''Warp the colormap so that more of the available colors are used on the range of interesting data.

    Half of the color map is used for values which fall below the midpoint,
    and half are used for values which fall above.
    This can be used to draw attention to smaller differences at the extreme
    ends of the observed values.

    based on:
        - http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
        - https://matplotlib.org/users/colormapnorms.html
    '''
    def __init__(self, vmin, vmax, midpoint, res=100, clip=False):
        '''
        Args:
            vmin: the minimum possible z/height value
            vmax: the maximum possible z/height value
            midpoint: the value to 'center' around
            res: the 'resolution' ie number of distinct levels in the colorbar
            clip: whether to clip the z values to [0,1] if they lie outside [vmin, vmax]

        Note: according to `mpl.colors.Normalize` documentation: If vmin or vmax
            is not given, they are initialized from the minimum and maximum
            value respectively of the first input processed.
        '''
        super().__init__(vmin, vmax, clip)
        self.midpoint = midpoint
        self.res = res

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

    def levels(self):
        '''
        Returns:
            a numpy array for the values where the boundaries between the colors
            should be placed.
        '''
        return np.concatenate((
            np.linspace(self.vmin, self.midpoint, num=self.res/2, endpoint=False),
            np.linspace(self.midpoint, self.vmax, num=self.res/2)))

