#!/usr/bin/env python3
'''
Sensible preset configurations for the optimiser
'''

import sklearn.gaussian_process as gp
from . import modules as tm


def load_optimiser_preset(optimiser, name):
    if name == 'default':
        # this default should hopefully provide reasonable results in most situations
        # the Optimiser.Plan defaults are left alone
        optimiser.latent_space = tm.NoLatentSpace()
        optimiser.pre_phase_select = tm.random_selector()
        optimiser.fallback = tm.Fallback()
        optimiser.maximise_acq = tm.random_quasi_newton()
        optimiser.async_eval = None
        optimiser.surrogate = tm.GPySurrogate()
        '''
        optimiser.surrogate = tm.SciKitGPSurrogate(model_params=dict(
            alpha = 1e-5, # larger => more noise. Default = 1e-10
            kernel = 1.0 * gp.kernels.Matern(nu=2.5) + gp.kernels.WhiteKernel(),
            n_restarts_optimizer = 10,
            normalize_Y = True,
        ))
        '''
        optimiser.acq_func_factory = tm.EI.Factory(xi=0.01)

    elif name == 'random_search':
        # a degenerate optimiser which never leaves the pre-phase and so
        # performs random search rather than Bayesian optimisation.
        optimiser.pre_phase_trials = float('inf')
        optimiser.latent_space = tm.NoLatentSpace()
        optimiser.pre_phase_select = tm.random_selector()
        optimiser.fallback = None
        optimiser.maximise_acq = None
        optimiser.async_eval = None
        optimiser.surrogate = None
        optimiser.acq_func_factory = None

    else:
        raise ValueError('unknown preset name: {}'.format(name))

