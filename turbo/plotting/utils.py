#!/usr/bin/env python3

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import os
import subprocess

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


def save_animation(figs, filename, time_per_frame=500):
    '''save a list of figures to an animated gif or a video format such as mp4

    Note: the figures will be closed by this function

    Note: requires the command line tool `convert` (part of ImageMagick)

    Tip: mp4 is roughly half the size of gif

    Tip: pass a generator instead of a list so that they can be rendered lazily
         (otherwise matplotlib gives a warning about having lots of figures open at
         once), for example:

            figs = (tp.plot_trial_1D(rec, param='x', trial_num=n) for n, _ in rec.get_sorted_trials())
            tp.save_animation(figs, 'out.mp4')

    Args:
        figs (iterable): a list of matplotlib figures
        filename (str): the filename to save to. Can be `.gif` or `.mp4` etc
        time_per_frame (int): milliseconds to display each frame for
    '''
    with tempfile.TemporaryDirectory() as tmp: # deleted after use
        filenames = []
        for i, fig in enumerate(figs):
            fig_filename = os.path.join(tmp, 'frame_{:02d}.png'.format(i))
            fig.savefig(fig_filename, bbox_inches='tight')
            plt.close(fig)
            filenames.append(fig_filename)
            print(fig_filename)

        delay = int(time_per_frame/10) # delay is in chunks of 10ms
        args = ['convert', '-delay', str(delay), '-background', 'white', '-alpha', 'remove']
        args += filenames
        args += [filename]
        subprocess.call(args)
        print('{} written'.format(filename))

